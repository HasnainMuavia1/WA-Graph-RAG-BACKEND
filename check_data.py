#!/usr/bin/env python3
"""
Quick script to check ingested data in PostgreSQL and Neo4j.
"""

import asyncio
import asyncpg
import os
from pathlib import Path

from dotenv import load_dotenv

# Try Neo4j import (optional)
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  Neo4j driver not available. Install with: pip install neo4j")

load_dotenv(Path(__file__).resolve().parent / ".env")


async def check_postgresql():
    """Check PostgreSQL data."""
    print("=" * 60)
    print("📊 PostgreSQL Database Check")
    print("=" * 60)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found in .env file")
        return

    try:
        conn = await asyncpg.connect(database_url)

        # Count documents
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        print(f"\n📄 Documents: {doc_count}")

        # Count chunks
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        print(f"📦 Chunks: {chunk_count}")

        # Count chunks with embeddings
        embedding_count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        )
        print(f"🔢 Chunks with embeddings: {embedding_count}")

        if doc_count > 0:
            # List documents
            print("\n📋 Documents:")
            docs = await conn.fetch("""
                SELECT title, source, created_at, 
                       (SELECT COUNT(*) FROM chunks WHERE document_id = documents.id) as chunk_count
                FROM documents 
                ORDER BY created_at DESC
            """)
            for doc in docs:
                print(f"  • {doc['title']}")
                print(f"    └─ {doc['chunk_count']} chunks | Source: {doc['source']}")

            # Sample chunks
            print("\n📝 Sample Chunks (first 3):")
            chunks = await conn.fetch("""
                SELECT d.title, c.chunk_index, LEFT(c.content, 100) as preview
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY d.title, c.chunk_index
                LIMIT 3
            """)
            for chunk in chunks:
                preview = chunk["preview"].replace("\n", " ")
                print(
                    f"  [{chunk['title']}] Chunk {chunk['chunk_index']}: {preview}..."
                )
        else:
            print("\n⚠️  No documents found. Run ingestion first:")
            print("   python -m ingestion.ingest --documents data --verbose")

        await conn.close()

    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        print("   Make sure Docker containers are running: docker ps")


def check_neo4j():
    """Check Neo4j data."""
    print("\n" + "=" * 60)
    print("🕸️  Neo4j Knowledge Graph Check")
    print("=" * 60)

    if not NEO4J_AVAILABLE:
        print("⚠️  Neo4j driver not installed. Skipping Neo4j check.")
        return

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Neo4j credentials not found in .env file")
        return

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"\n🕸️  Nodes: {node_count}")

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            print(f"🔗 Relationships: {rel_count}")

            if node_count > 0:
                # Node types
                result = session.run("""
                    MATCH (n) 
                    RETURN labels(n)[0] as label, count(*) as count 
                    ORDER BY count DESC
                    LIMIT 10
                """)
                print("\n📊 Node Types:")
                for record in result:
                    print(f"  • {record['label']}: {record['count']}")

                # Relationship types
                result = session.run("""
                    MATCH ()-[r]->() 
                    RETURN type(r) as rel_type, count(*) as count 
                    ORDER BY count DESC
                    LIMIT 10
                """)
                print("\n🔗 Relationship Types:")
                for record in result:
                    print(f"  • {record['rel_type']}: {record['count']}")
            else:
                print("\n⚠️  No nodes found. Knowledge graph may not be built yet.")
                print("   Make sure ingestion completed with knowledge graph enabled.")

        driver.close()

    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("   Make sure Docker containers are running: docker ps")
        print("   Check Neo4j credentials in .env file")


async def main():
    """Main function."""
    await check_postgresql()
    check_neo4j()

    print("\n" + "=" * 60)
    print("✅ Check complete!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("  • View Neo4j graph: http://localhost:7474")
    print("  • Check Docker containers: docker ps")
    print("  • View full guide: cat CHECK_DATA.md")


if __name__ == "__main__":
    asyncio.run(main())
