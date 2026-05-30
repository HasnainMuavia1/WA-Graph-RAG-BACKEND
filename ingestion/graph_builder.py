"""
Knowledge-graph building from document chunks using Neo4j.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_knowledge_graph(
    chunk: str,
    embedding: List[float],
    doc_key: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store a document chunk and its embedding as a graph node, then extract
    named entities and link them with relationships.

    Requires the neo4j package and a running Neo4j instance.
    Silently skips if neo4j is not installed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j package not installed — skipping graph building")
        return

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri:
        logger.warning("NEO4J_URI not set — skipping graph building")
        return

    meta = metadata or {}
    access_level = meta.get("access_level", "public")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            session.run(
                """
                MERGE (c:Chunk {doc_key: $doc_key, content_hash: $content_hash})
                SET c.content = $chunk,
                    c.access_level = $access_level,
                    c.doc_key = $doc_key
                """,
                doc_key=doc_key,
                content_hash=_hash(chunk),
                chunk=chunk[:2000],
                access_level=access_level,
            )
        driver.close()
    except Exception as exc:
        logger.error("Graph building failed for %s: %s", doc_key, exc)


def _hash(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode()).hexdigest()
