"""
FastAPI application — all routes live under /api/v1.

Background ingestion (Celery)
-----------------------------
Ingestion runs on a Celery worker pool backed by Redis. Periodic full sweeps
are scheduled by Celery Beat (INGEST_INTERVAL_MINUTES, default 15). Manual and
webhook-driven ingests enqueue tasks and return a task id for status polling.

S3 webhook (real-time)
----------------------
POST /api/v1/ingest/webhook/s3 receives SNS-wrapped S3 event notifications and
enqueues a Celery task to ingest the changed object.

WhatsApp + voice
----------------
/api/v1/whatsapp/webhook receives Meta Cloud API events. Text and voice-note
messages are processed by a Celery task (Deepgram transcribes voice) and the
RAG agent's reply is sent back to the user.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi import APIRouter

from .agent import rag_agent, AgentDependencies
from .db_utils import (
    initialize_database,
    close_database,
    test_connection,
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection
from .models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthStatus,
    SearchRequest,
    SearchResponse,
    ToolCall,
)
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
INGEST_INTERVAL_MINUTES = int(os.getenv("INGEST_INTERVAL_MINUTES", "15"))

# ── LangChain session memory (no DB persistence) ──────────────────────────────
from .session_memory import memory_manager

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)

# ── Background ingestion ──────────────────────────────────────────────────────
# Periodic ingestion is now driven by Celery Beat (see worker/celery_app.py),
# not an in-process asyncio loop. This makes scheduling durable and lets us
# scale ingestion across multiple worker containers.


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Agentic RAG API …")
    try:
        await initialize_database()
        logger.info("PostgreSQL ready")
        await initialize_graph()
        logger.info("Neo4j ready")

        # Build BM25 index from existing chunks
        try:
            from .retriever import hybrid_retriever
            await hybrid_retriever.build_index()
            logger.info("BM25 index ready")
        except Exception as exc:
            logger.warning("BM25 index build skipped: %s", exc)

        logger.info("Agentic RAG API started (ingestion runs on Celery workers)")
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise
    yield
    logger.info("Shutting down …")
    try:
        await close_database()
        await close_graph()
        logger.info("Connections closed")
    except Exception as exc:
        logger.error("Shutdown error: %s", exc)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description=(
        "AI research assistant combining BM25 + vector hybrid search "
        "with a Neo4j knowledge graph. All endpoints are versioned under /api/v1."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Versioned router — all public endpoints live here
v1_router = APIRouter(prefix="/api/v1")

# ── Helper functions (also importable for testing) ────────────────────────────

def get_or_create_session(request: ChatRequest) -> str:
    """Return existing session_id or create a new one via LangChain memory."""
    return memory_manager.get_or_create(request.session_id)


def get_conversation_context(session_id: str) -> str:
    """Return the formatted conversation history string for this session."""
    return memory_manager.get_context_string(session_id)


def extract_tool_calls(result) -> List[ToolCall]:
    tools_used: List[ToolCall] = []
    try:
        for message in result.all_messages():
            if not hasattr(message, "parts"):
                continue
            for part in message.parts:
                if part.__class__.__name__ != "ToolCallPart":
                    continue
                try:
                    tool_name = str(getattr(part, "tool_name", "unknown"))
                    tool_args: Dict[str, Any] = {}
                    raw_args = getattr(part, "args", None)
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    elif isinstance(raw_args, dict):
                        tool_args = raw_args
                    tool_call_id = str(part.tool_call_id) if getattr(part, "tool_call_id", None) else None
                    tools_used.append(
                        ToolCall(tool_name=tool_name, args=tool_args, tool_call_id=tool_call_id)
                    )
                except Exception as exc:
                    logger.debug("Failed to parse tool call part: %s", exc)
    except Exception as exc:
        logger.warning("Failed to extract tool calls: %s", exc)
    return tools_used


def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a turn to LangChain memory (sliding window, no DB write)."""
    memory_manager.add_turn(session_id, user_message, assistant_message)


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> tuple[str, List[ToolCall]]:
    # ── Input guardrails (fail closed) ────────────────────────────────────────
    from .guardrails import check_input, apply_output_guardrails

    verdict = check_input(message)
    if not verdict.allowed:
        blocked = verdict.user_message or "Maazrat, main is sawal ka jawab nahi de sakta."
        save_conversation_turn(session_id, message, blocked)
        return blocked, []
    safe_message = verdict.sanitized_input or message

    try:
        deps = AgentDependencies(session_id=session_id, user_id=user_id)

        # Build prompt: prepend LangChain memory context if the session has history
        history = get_conversation_context(session_id)
        full_prompt = (
            f"Previous conversation:\n{history}\n\nCurrent question: {safe_message}"
            if history else safe_message
        )

        result = await rag_agent.run(full_prompt, deps=deps)
        # pydantic-ai >=1.0 exposes `.output`; older versions used `.data`.
        response: str = getattr(result, "output", None) or getattr(result, "data", "")
        tools_used = extract_tool_calls(result)

        # ── Output guardrails (leak scrub, PII redaction, Roman-Urdu) ─────────
        response = await apply_output_guardrails(response)

        # Persist to LangChain memory (in-process, no DB)
        save_conversation_turn(session_id, message, response)
        return response, tools_used

    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        error_response = "Maazrat — aap ki request process karte hue masla hua. Dobara koshish karein."
        save_conversation_turn(session_id, message, error_response)
        return error_response, []


# ── /api/v1/health ────────────────────────────────────────────────────────────

@v1_router.get("/health", response_model=HealthStatus, tags=["health"])
async def health_check():
    """Service health check — reports database and graph database connectivity."""
    try:
        db_status = await test_connection()
        graph_status = await test_graph_connection()
        if db_status and graph_status:
            status = "healthy"
        elif db_status or graph_status:
            status = "degraded"
        else:
            status = "unhealthy"
        return HealthStatus(
            status=status,
            database=db_status,
            graph_database=graph_status,
            llm_connection=True,
            version="1.0.0",
            timestamp=datetime.now(),
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Health check failed")


# ── /api/v1/chat ──────────────────────────────────────────────────────────────

@v1_router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """Non-streaming chat — returns the full agent response in one payload."""
    try:
        session_id = get_or_create_session(request)
        response, tools_used = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
        )
        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)},
        )
    except Exception as exc:
        logger.error("Chat endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/chat/stream", tags=["chat"])
async def chat_stream(request: ChatRequest):
    """Streaming chat via Server-Sent Events (SSE)."""
    try:
        session_id = get_or_create_session(request)

        async def generate_stream():
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

                # ── Input guardrails (fail closed) ────────────────────────────
                from .guardrails import check_input, apply_output_guardrails

                verdict = check_input(request.message)
                if not verdict.allowed:
                    blocked = verdict.user_message or "Maazrat, main is sawal ka jawab nahi de sakta."
                    yield f"data: {json.dumps({'type': 'text', 'content': blocked})}\n\n"
                    save_conversation_turn(session_id, request.message, blocked)
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return
                safe_message = verdict.sanitized_input or request.message

                deps = AgentDependencies(session_id=session_id, user_id=request.user_id)

                history = get_conversation_context(session_id)
                full_prompt = (
                    f"Previous conversation:\n{history}\n\nCurrent question: {safe_message}"
                    if history else safe_message
                )

                full_response = ""

                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
                                    if isinstance(event, PartStartEvent) and event.part.part_kind == "text":
                                        delta = event.part.content
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        delta = event.delta.content_delta
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta

                tools_used = extract_tool_calls(run.result)
                if tools_used:
                    yield f"data: {json.dumps({'type': 'tools', 'tools': [t.model_dump() for t in tools_used]})}\n\n"

                # ── Output guardrails on the assembled response ───────────────
                guarded = await apply_output_guardrails(full_response)
                if guarded != full_response:
                    # Streamed text needs correcting (e.g. Devanagari → Roman Urdu
                    # or PII redaction). Tell the client to replace the message.
                    yield f"data: {json.dumps({'type': 'replace', 'content': guarded})}\n\n"

                # Save the guarded version to LangChain memory (no DB)
                save_conversation_turn(session_id, request.message, guarded)
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except Exception as exc:
                logger.error("Stream error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as exc:
        logger.error("Streaming chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/search ────────────────────────────────────────────────────────────

@v1_router.post("/search/vector", response_model=SearchResponse, tags=["search"])
async def search_vector(request: SearchRequest):
    """Pure vector (semantic) search using pgvector cosine similarity."""
    try:
        t0 = datetime.now()
        results = await vector_search_tool(
            VectorSearchInput(query=request.query, limit=request.limit, user_id=request.filters.get("user_id"))
        )
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/search/graph", response_model=SearchResponse, tags=["search"])
async def search_graph(request: SearchRequest):
    """Knowledge graph full-text search."""
    try:
        t0 = datetime.now()
        results = await graph_search_tool(GraphSearchInput(query=request.query))
        return SearchResponse(
            graph_results=results,
            total_results=len(results),
            search_type="graph",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Graph search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/search/hybrid", response_model=SearchResponse, tags=["search"])
async def search_hybrid(request: SearchRequest):
    """Hybrid search: BM25 (LlamaIndex) + pgvector fused with Reciprocal Rank Fusion."""
    try:
        t0 = datetime.now()
        results = await hybrid_search_tool(
            HybridSearchInput(
                query=request.query,
                limit=request.limit,
                user_id=request.filters.get("user_id"),
            )
        )
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/documents ────────────────────────────────────────────────────────

@v1_router.get("/documents", tags=["documents"])
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
):
    """List ingested documents with optional user-based access filtering."""
    try:
        documents = await list_documents_tool(
            DocumentListInput(limit=limit, offset=offset, user_id=user_id)
        )
        return {"documents": documents, "total": len(documents), "limit": limit, "offset": offset}
    except Exception as exc:
        logger.error("Document listing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.get("/documents/{document_id}", tags=["documents"])
async def get_document_endpoint(document_id: str):
    """Retrieve a single document's full details and content."""
    try:
        from .tools import get_document_tool, DocumentInput
        doc = await get_document_tool(DocumentInput(document_id=document_id))
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document_endpoint(document_id: str):
    """Delete an ingested document and its chunks."""
    try:
        from .db_utils import delete_document, get_document
        doc = await get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        await delete_document(document_id)
        # Rebuild BM25 index after deletion!
        try:
            from agent.retriever import hybrid_retriever
            await hybrid_retriever.rebuild_index()
        except Exception as exc:
            logger.warning("BM25 rebuild skipped: %s", exc)
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document deletion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/documents/upload", tags=["documents"])
async def upload_document_endpoint(
    file: UploadFile = File(...),
    access_level: str = Form("public"),
):
    """Upload and ingest a document file directly."""
    try:
        import tempfile
        from pathlib import Path
        from ingestion.file_parsers import parse_document

        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            content, file_meta = parse_document(tmp_path)
            if not content.strip():
                raise HTTPException(status_code=400, detail="Document contains no parseable text")

            # Unique source
            source = f"upload://{uuid.uuid4()}/{file.filename}"
            title = Path(file.filename).stem

            # Parsing is fast and done inline; the heavy chunk→embed→upsert→graph
            # work is offloaded to a Celery worker so the request returns quickly.
            from worker.tasks import ingest_document_task

            async_result = ingest_document_task.delay(
                content=content,
                source=source,
                title=title,
                metadata={**file_meta, "uploaded": True, "file_name": file.filename},
                access_level=access_level,
            )

            return {
                "status": "queued",
                "task_id": async_result.id,
                "title": title,
                "source": source,
                "message": "Document parsed and queued for indexing. Poll /api/v1/ingest/status/{task_id}.",
            }
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document upload/ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/sessions ─────────────────────────────────────────────────────────

@v1_router.get("/sessions/{session_id}", tags=["sessions"])
async def get_session_info(session_id: str):
    """Return in-memory session metadata (turn count, window size)."""
    if not memory_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return memory_manager.session_info(session_id)


# ── /api/v1/ingest ───────────────────────────────────────────────────────────

@v1_router.post("/ingest/trigger", tags=["ingest"])
async def trigger_ingest():
    """
    Enqueue a full S3 ingestion run on the Celery worker pool.

    Returns immediately with the Celery task id — poll
    ``GET /api/v1/ingest/status/{task_id}`` for progress and the final result.
    """
    from worker.tasks import ingest_all_task

    async_result = ingest_all_task.delay()
    logger.info("Enqueued full ingest (task_id=%s)", async_result.id)
    return {
        "status": "queued",
        "task_id": async_result.id,
        "message": "Ingestion queued on Celery. Poll /api/v1/ingest/status/{task_id}.",
    }


@v1_router.post("/ingest/s3", tags=["ingest"])
async def ingest_s3_bucket(bucket_type: str = "private", prefix: str = ""):
    """Enqueue ingestion of a specific S3 bucket / prefix on the worker pool."""
    from worker.tasks import ingest_bucket_task

    async_result = ingest_bucket_task.delay(bucket_type=bucket_type, prefix=prefix)
    logger.info("Enqueued bucket ingest '%s/%s' (task_id=%s)", bucket_type, prefix, async_result.id)
    return {
        "status": "queued",
        "task_id": async_result.id,
        "bucket_type": bucket_type,
        "prefix": prefix,
    }


@v1_router.get("/ingest/status/{task_id}", tags=["ingest"])
async def ingest_status(task_id: str):
    """Return the live state and result of a queued ingestion task."""
    from worker.celery_app import celery_app

    res = celery_app.AsyncResult(task_id)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "state": res.state,  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
        "ready": res.ready(),
        "successful": res.successful() if res.ready() else None,
    }
    if res.ready():
        # `.result` is the return value on success or the exception on failure.
        payload["result"] = res.result if res.successful() else str(res.result)
    return payload


@v1_router.post("/ingest/webhook/s3", tags=["ingest"])
async def s3_event_webhook(request: Request):
    """
    Receive AWS S3 event notifications (delivered via SNS HTTP subscription).

    S3 → SNS topic → HTTPS subscription → this endpoint

    Handles both SNS SubscriptionConfirmation and Notification message types.
    New or modified objects are enqueued for ingestion on the Celery workers.
    """
    body = await request.json()

    # SNS subscription confirmation handshake
    if body.get("Type") == "SubscriptionConfirmation":
        import urllib.request
        urllib.request.urlopen(body["SubscribeURL"])
        return {"status": "confirmed"}

    # S3 event notification
    if body.get("Type") == "Notification":
        try:
            message = json.loads(body.get("Message", "{}"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid SNS message body")

        from worker.tasks import ingest_single_s3_task

        records = message.get("Records", [])
        queued = 0
        for record in records:
            s3_info = record.get("s3", {})
            bucket_name = s3_info.get("bucket", {}).get("name", "")
            object_key = s3_info.get("object", {}).get("key", "")
            event_name = record.get("eventName", "")

            if not object_key:
                continue

            if "ObjectCreated" in event_name or "ObjectModified" in event_name:
                logger.info("S3 event: %s → %s/%s (enqueuing)", event_name, bucket_name, object_key)
                ingest_single_s3_task.delay(s3_key=object_key, bucket_name=bucket_name)
                queued += 1

        return {"status": "accepted", "records_queued": queued}

    return {"status": "ignored"}


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4()),
    )


# ── Mount versioned router ────────────────────────────────────────────────────

app.include_router(v1_router)

from .auth_router import router as auth_router
from .users_router import router as users_router
from .whatsapp_router import router as whatsapp_router
from .conversations_router import router as conversations_router
from .dashboard_router import router as dashboard_router
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(whatsapp_router)
app.include_router(conversations_router)
app.include_router(dashboard_router)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
    )
