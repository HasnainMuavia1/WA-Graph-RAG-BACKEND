"""
S3-based document ingestion pipeline.
Fetches documents from S3, parses, chunks, embeds, and stores them.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from .s3_utils import (
    list_documents_from_s3,
    download_document_from_s3,
    verify_s3_access,
)
from .file_parsers import parse_document
from .chunker import ChunkingConfig, create_chunker
from .embedder import create_embedder
from .graph_builder import build_knowledge_graph

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.access_control import assign_access_level

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

_DEFAULT_CHUNKING_CONFIG = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    use_semantic_splitting=True,
)


async def ingest_from_s3(
    bucket_type: str = "private",
    prefix: str = "",
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Ingest documents from a single S3 bucket.

    Returns a stats dict: {"success": int, "failed": int, "skipped": int}
    """
    if not verify_s3_access():
        logger.error("S3 access verification failed")
        return {"success": 0, "failed": 0, "skipped": 0}

    stats: Dict[str, int] = {"success": 0, "failed": 0, "skipped": 0}

    doc_keys = list_documents_from_s3(bucket_type=bucket_type, prefix=prefix)
    if not doc_keys:
        logger.warning("No documents found in '%s' bucket", bucket_type)
        return stats

    logger.info("Found %d documents in '%s' bucket", len(doc_keys), bucket_type)

    chunker = create_chunker(_DEFAULT_CHUNKING_CONFIG)
    embedder = create_embedder()
    access_level = assign_access_level(bucket_type)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for doc_key in doc_keys:
            try:
                local_path = str(Path(tmp_dir) / Path(doc_key).name)
                content_bytes = download_document_from_s3(
                    doc_key, bucket_type=bucket_type, save_path=local_path
                )
                if content_bytes is None:
                    logger.error("Failed to download '%s'", doc_key)
                    stats["failed"] += 1
                    continue

                parsed_content, file_metadata = parse_document(local_path)
                if not parsed_content:
                    logger.warning("Empty content after parsing '%s'", doc_key)
                    stats["failed"] += 1
                    continue

                doc_metadata = {
                    **file_metadata,
                    "bucket_type": bucket_type,
                    "access_level": access_level,
                    "s3_key": doc_key,
                }

                chunks = await chunker.chunk_document(
                    content=parsed_content,
                    title=Path(doc_key).stem,
                    source=doc_key,
                    metadata=doc_metadata,
                )

                embedded_chunks = await embedder.embed_chunks(chunks)

                for chunk in embedded_chunks:
                    embedding = getattr(chunk, "embedding", [])
                    build_knowledge_graph(
                        chunk=chunk.content,
                        embedding=embedding,
                        doc_key=doc_key,
                        metadata={
                            "bucket_type": bucket_type,
                            "access_level": access_level,
                        },
                    )

                logger.info(
                    "Processed '%s' → %d chunks [%s]",
                    doc_key,
                    len(embedded_chunks),
                    access_level,
                )
                stats["success"] += 1

            except Exception as exc:
                logger.error("Error processing '%s': %s", doc_key, exc)
                stats["failed"] += 1

    logger.info(
        "Ingestion complete — success=%d, failed=%d, skipped=%d",
        stats["success"],
        stats["failed"],
        stats["skipped"],
    )
    return stats


async def ingest_all_s3_buckets(skip_existing: bool = True) -> Dict[str, object]:
    """Ingest documents from both the private and public S3 buckets."""
    private = await ingest_from_s3(bucket_type="private", skip_existing=skip_existing)
    public = await ingest_from_s3(bucket_type="public", skip_existing=skip_existing)

    total_success = private["success"] + public["success"]
    total_failed = private["failed"] + public["failed"]
    logger.info("Total — success=%d, failed=%d", total_success, total_failed)

    return {"private": private, "public": public}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(ingest_all_s3_buckets())
    print(f"Results: {results}")
