"""
Neo4j knowledge graph utilities.
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_driver = None


class GraphClient:
    """Thin wrapper around the Neo4j driver for knowledge-graph operations."""

    def __init__(self, driver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def _session(self):
        return self._driver.session(database=self._database)

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
    ) -> List[Dict[str, Any]]:
        """Return time-ordered facts for a named entity."""
        cypher = """
        MATCH (e:Entity {name: $entity_name})-[r:HAS_FACT]->(f:Fact)
        WHERE ($start_date IS NULL OR f.valid_at >= $start_date)
          AND ($end_date   IS NULL OR f.valid_at <= $end_date)
        RETURN f.content AS fact, f.valid_at AS valid_at, f.invalid_at AS invalid_at
        ORDER BY f.valid_at ASC
        """
        try:
            async with self._session() as session:
                result = await session.run(
                    cypher,
                    entity_name=entity_name,
                    start_date=str(start_date) if start_date else None,
                    end_date=str(end_date) if end_date else None,
                )
                records = await result.data()
            return records
        except Exception as exc:
            logger.error("Entity timeline query failed: %s", exc)
            return []


# Module-level singleton — populated by initialize_graph()
graph_client: Optional[GraphClient] = None


async def initialize_graph() -> None:
    """Open the Neo4j driver and expose the global graph_client."""
    global _driver, graph_client

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri:
        logger.warning("NEO4J_URI not set — graph database disabled")
        return

    try:
        from neo4j import AsyncGraphDatabase

        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        graph_client = GraphClient(_driver, database=database)
        logger.info("Neo4j graph database initialised (%s)", database)
    except ImportError:
        logger.warning("neo4j package not installed — graph database disabled")
    except Exception as exc:
        logger.error("Failed to initialise Neo4j: %s", exc)


async def close_graph() -> None:
    """Close the Neo4j driver."""
    global _driver, graph_client
    if _driver:
        await _driver.close()
        _driver = None
        graph_client = None
        logger.info("Neo4j connection closed")


async def test_graph_connection() -> bool:
    """Return True if Neo4j is reachable."""
    if not _driver:
        return False
    try:
        async with _driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            await session.run("RETURN 1")
        return True
    except Exception as exc:
        logger.error("Neo4j connection test failed: %s", exc)
        return False


async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """Full-text search over graph facts."""
    if not _driver:
        return []

    cypher = """
    CALL db.index.fulltext.queryNodes('factIndex', $query)
    YIELD node, score
    WHERE node:Fact
    RETURN
        node.content        AS fact,
        node.uuid           AS uuid,
        node.valid_at       AS valid_at,
        node.invalid_at     AS invalid_at,
        node.source_node_uuid AS source_node_uuid
    ORDER BY score DESC
    LIMIT 20
    """
    try:
        async with _driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            result = await session.run(cypher, query=query)
            records = await result.data()
        return records
    except Exception as exc:
        logger.error("Knowledge graph search failed: %s", exc)
        return []


async def get_entity_relationships(
    entity: str, depth: int = 2
) -> Dict[str, Any]:
    """Return entities and relationships within *depth* hops from *entity*."""
    if not _driver:
        return {"central_entity": entity, "related_entities": [], "relationships": []}

    cypher = f"""
    MATCH path = (e:Entity {{name: $entity}})-[*1..{depth}]-(related)
    RETURN
        e.name              AS central_entity,
        collect(DISTINCT related.name) AS related_entities,
        [r IN relationships(path) | type(r)] AS relationship_types
    LIMIT 50
    """
    try:
        async with _driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            result = await session.run(cypher, entity=entity)
            records = await result.data()
        if records:
            return {
                "central_entity": entity,
                "related_entities": records[0].get("related_entities", []),
                "relationships": records[0].get("relationship_types", []),
                "depth": depth,
            }
        return {"central_entity": entity, "related_entities": [], "relationships": [], "depth": depth}
    except Exception as exc:
        logger.error("Entity relationship query failed: %s", exc)
        return {
            "central_entity": entity,
            "related_entities": [],
            "relationships": [],
            "depth": depth,
            "error": str(exc),
        }
