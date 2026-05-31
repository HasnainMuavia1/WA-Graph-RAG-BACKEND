# Uchenab RAG — Enterprise Backend Services

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DD0031?style=for-the-badge&logo=redis&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=celery&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Deepgram](https://img.shields.io/badge/Deepgram-13EF95?style=for-the-badge&logo=deepgram&logoColor=black)
![AWS S3](https://img.shields.io/badge/Amazon_S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

The **Uchenab University Assistant** backend is a state-of-the-art, enterprise-grade AI engine designed to assist university students and applicants. Powered by **Pydantic AI** and **FastAPI**, it leverages a dual-retrieval pipeline—combining semantic cosine searches in **Supabase pgvector** and **LlamaIndex BM25** keyword queries—fused with a structural **Neo4j Knowledge Graph**. 

The system features real-time asynchronous background pipelines running on **Celery + Redis**, integrating **Meta WhatsApp Cloud APIs**, **Deepgram** voice note translation, strict guardrails (Roman-Urdu translation, prompt injection filters, and CNIC/PII redaction), and robust connection caching.

---

## 🏗️ Architectural Blueprints

### 1. Detailed Agentic RAG Flow
The RAG pipeline retrieves and fuses context from both vector search and structured graph networks before serving the query to OpenAI GPT-4o.

```mermaid
graph TD
    User([User Query]) --> ScopeCheck{Is Query In-Scope?<br/>agent/guardrails.py}
    ScopeCheck -->|No - Out of Scope| OutOfScopeMsg["Graceful Refusal in Roman Urdu<br/>(Postgres logged)"]
    ScopeCheck -->|Yes - In Scope| IG["Input Guardrails Check<br/>(empty/too_long/injection/abuse)"]
    
    IG -->|Blocked| BlockedMsg["Return Guardrail Security Alert<br/>(Redis logged)"]
    IG -->|Allowed| Agent["Pydantic AI RAG Agent (agent/agent.py)"]
    
    Agent -->|1. Loads Context| Memory[(Redis Session Memory)]
    Agent -->|2. Agent Brain Pass: Analyzes Intent| ToolRouter{Which Tool is Needed?}
    
    %% Tool Routing
    ToolRouter -->|Factual / Entity Relations| GraphTool["search_knowledge_graph_facts (Tool)"]
    ToolRouter -->|Document / Policy Search| DocTool["search_documents (Tool)"]
    ToolRouter -->|Complex / Structural + Context| Both["Invoke Both Tools"]
    
    %% Graph Tool execution
    GraphTool --> Neo4jSearch["Neo4j Knowledge Graph (Cypher Queries)"]
    
    %% Doc Tool execution
    DocTool --> HybridSearch["Hybrid Search Engine"]
    Both --> Neo4jSearch
    Both --> HybridSearch
    
    %% Hybrid Search Breakdown
    subgraph Hybrid Retrieval Pipeline
        HybridSearch --> VectorSearch["pgvector Semantic Search (Supabase)"]
        HybridSearch --> BM25Search["BM25 Keyword Search (LlamaIndex)"]
        VectorSearch --> RERANK["Reciprocal Rank Fusion (RRF)"]
        BM25Search --> RERANK
    end
    
    %% Context synthesis
    RERANK --> FusedDocContext["Fused Documents Text Context"]
    Neo4jSearch --> GraphContext["Structured Entity Facts"]
    
    %% Relevance gate
    FusedDocContext --> ConfGate{Relevance Max Score >= 0.015?}
    GraphContext --> ConfGate
    
    %% Low Confidence branch
    ConfGate -->|No - Low Confidence| LowConf["Low Confidence Handling"]
    LowConf --> CeleryAlert["Dispatch Celery Task: notify_admin_weak_context_task"]
    LowConf --> SaveAlert["Post Red Warning Card to Admin Transcript"]
    LowConf --> ChannelSplit{Messaging Channel?}
    ChannelSplit -->|WhatsApp| Suppress["Suppress Automated Student Reply"]
    ChannelSplit -->|Web Chat| WebFallback["Return Polite Support Escalation Refusal"]
    
    %% High Confidence branch
    ConfGate -->|Yes - High Confidence| PromptBuilder["System Prompt Generator"]
    Memory --> PromptBuilder
    
    PromptBuilder --> LLM["OpenAI GPT-4o-mini / GPT-4o"]
    LLM --> RawOutput["Raw Agent Response"]
    RawOutput --> OG["Output Guardrails (PII & Leak Redactor)"]
    OG -->|Cleaned Response| SaveProvenance["Save Message + JSON Metadata (Provenance Logs)"]
    SaveProvenance --> UserResponse([Final Clean Response to Student])
```

---

### 2. Session Memory Lifecycle (Redis Cache)
Conversation history is preserved across stateless workers in a high-speed sliding-window Redis structure, failing back gracefully to in-process memory if Redis is offline.

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client (Web / WhatsApp)
    participant API as FastAPI Router / Celery
    participant Redis as Redis Cache
    participant Fallback as Local In-Process Dict

    Client->>API: Post Query /chat (session_id = "user_123")
    API->>Redis: Check Session History (key = "uchenab:session:user_123")
    alt Redis is connected & key exists
        Redis-->>API: Return JSON turns list (LangChain messages)
    else Redis is empty / missing key
        API->>API: Treat as new conversation (empty context window)
    else Redis is unreachable / connection failed
        API->>Fallback: Check local dictionary cache (_histories)
        Fallback-->>API: Return local memory list (if exists)
    end
    API->>API: Run Agent Turn (Combine history + new query)
    alt Redis is online
        API->>Redis: Append Turn (RPUSH User Query + Assistant Response)
        API->>Redis: Trim list to Context Window (LTRIM -20 -1)
        API->>Redis: Refresh rolling TTL (EXPIRE 24h)
    else Fallback mode
        API->>Fallback: Append Turn & trim list locally
    end
    API-->>Client: Deliver Message Response
```

---

### 3. WhatsApp Messaging & Voice Loop
Voice and text events from the Meta Graph API webhook are offloaded immediately to background queues to return a fast `200 OK` handshake response, logging transactions to Supabase and issuing read receipts.

```mermaid
graph TD
    User([WhatsApp Mobile User]) -->|Sends Text or Voice Note| Meta["Meta WhatsApp Cloud API"]
    Meta -->|HTTP POST Webhook Event| API["FastAPI Endpoint (/api/v1/whatsapp)"]
    API -->|Instantly returns 200 OK Handshake| Meta
    API -->|Asynchronously Enqueue Task| Broker[(Celery Redis Broker)]
    Broker -->|Picks up Task| Worker["Celery Messaging Worker"]
    
    %% Read receipt hook
    Worker -->|Mark message as read| ReadReceipt["whatsapp_client.mark_read (Meta API)"]
    ReadReceipt --> Meta
    
    subgraph Inbound processing
        Worker -->|Check Inbound Type| IsVoice{Is Message Voice Note?}
        IsVoice -->|Yes| Download["Download Voice File (.ogg)"]
        Download --> DG["Deepgram Speech-to-Text API"]
        DG -->|Transcribed Text| RecordInbound["conversation_store.record_inbound (Supabase: wa_messages)"]
        IsVoice -->|No| TextOnly["Get Text Content"]
        TextOnly --> RecordInbound
        RecordInbound --> Agent["Pydantic AI RAG Agent"]
    end
      
    Agent -->|Compute Context| Search[Vector / Graph Retrieve]
    Agent -->|Formulate Answer| Guard["Output Guardrails (Enforce Roman-Urdu)"]
    Guard --> SendMeta["Meta Graph API Outbound Call"]
    SendMeta --> User
    
    %% Save outbound
    SendMeta --> RecordOutbound["conversation_store.record_outbound (Supabase: wa_messages)"]
```

---

### 4. Agent Message vs. Admin Message Flows
AI automated workflows run independently from administrative manual intercepts inside the dashboard portal, unified through the outbound delivery endpoint.

```mermaid
graph TD
    UserStart([User Activity / Interaction])
    
    %% Flows
    UserStart -->|User inputs message| UserIn([User WhatsApp Input])
    UserStart -->|Admin views Alert & Types Message| AdminIn([Admin Dashboard UI])
    
    subgraph Auto Agent Reply Flow
        UserIn --> Webhook["FastAPI Webhook /api/v1/whatsapp"]
        Webhook --> Celery["Celery Task (process_whatsapp_message)"]
        Celery --> RAG["AI Agent (Autonomous Model)"]
        
        %% Splitting based on confidence
        RAG --> ScoreCheck{Confidence Score Check}
        ScoreCheck -->|Strong| SaveAuto["Save Message (sender = agent, direction = outbound)"]
        ScoreCheck -->|Weak / Low Confidence| AlertAdmin["Post Warning Banner to Admin Inbox + Suppress Reply"]
    end

    subgraph Manual Admin Intercept & Reply
        AlertAdmin -->|Visual Alert Card In Inbox| AdminIn
        AdminIn -->|Admin toggles Debug Mode| ViewProvenance["View Provenance Logs (retrieved chunks, scores, prompt summary)"]
        AdminIn -->|Admin types direct response| Auth["JWT Authenticated API Gated Router"]
        Auth --> SendEndpoint["POST /api/v1/conversations/{wa_id}/messages"]
        SendEndpoint --> Bypass["Bypass AI Processing & Guardrails"]
        Bypass --> SaveManual["Save Message (sender = admin, direction = outbound)"]
        SaveManual --> ResetUnread["Update wa_conversations: unread_count = 0"]
    end
    
    %% Output
    SaveAuto --> OutWA["WhatsApp Cloud API (Outbound Delivery)"]
    ResetUnread --> OutWA
    
    OutWA --> UserMobile([User Mobile Device])
```

---

### 5. Document Ingestion Pipeline
When files are added manually or synced periodically from S3, the pipeline processes and registers the metadata structure.

```mermaid
graph TD
    Doc([PDF, DOCX, XLSX, PPTX, CSV, HTML]) --> Upload["API Upload / S3 Bucket Webhook"]
    Upload --> Temp["Local Temp Directory"]
    Temp --> Parse["Parsers (pypdf, python-docx, openpyxl, beautifulsoup4)"]
    Parse --> CeleryTask["Celery Task (ingest_document_task)"]
    
    CeleryTask --> Chunk["Text Chunking (Overlap & Size splits)"]
    Chunk --> HashCheck{"Is Chunks Hash in PostgreSQL?"}
    HashCheck -->|Yes - Duplicate| Skip["Skip (Skip Embedding to avoid DB spam)"]
    HashCheck -->|No - New/Changed| Embed["Generate OpenAI vector embeddings"]
    
    Embed --> Postgres[("PostgreSQL (pgvector chunks upsert)")]
    Embed --> Neo4j[("Neo4j Knowledge Graph (Cypher nodes & links)")]
```

---

## 📚 References & Developer Documentation
To deep-dive into endpoints testing strategies or to check the QA test matrices, refer to our comprehensive internal documentation:
* 📄 **[API Testing Guide](apitesting.md)**: Detailed step-by-step specifications of all REST operations, Celery hooks, auth handshakes, and postman testing guidelines.
* 📋 **[QA Test Cases Log](testcases.md)**: Fully granular testing matrices, expected assertions, input/output boundary cases, and performance criteria log.

---

## 🛠️ Ingestion & Setup

### Service Map
* `api`: FastAPI application, serving all routes under `/api/v1/*` (port `8058`).
* `worker`: Celery worker performing voice processing, ingestion, and RAG execution.
* `beat`: Celery scheduler driving periodic 15-minute S3 synchronizations.
* `redis`: High-speed message broker, Celery backend, and endpoint/avatar cache store.

---

### 🚀 Running with Docker (Recommended)

1. Set up configurations:
   ```bash
   cp .env.example .env
   ```
2. Build and run all services:
   ```bash
   docker compose up --build
   ```
3. Access API Documentation at [http://localhost:8058/api/docs](http://localhost:8058/api/docs).

---

### 💻 Local (Non-Containerized) Setup

Ensure **PostgreSQL**, **Neo4j**, and **Redis** servers are running locally.

1. Initialize a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the services across separate terminals:
   * **Terminal 1 (API)**:
     ```bash
     python -m uvicorn agent.api:app --reload --port 8058
     ```
   * **Terminal 2 (Celery Worker)**:
     ```bash
     celery -A worker.celery_app:celery_app worker -Q ingestion,messaging --loglevel=info
     ```
   * **Terminal 3 (Celery Scheduler)**:
     ```bash
     celery -A worker.celery_app:celery_app beat --loglevel=info
     ```

---

## 🧪 Testing Suite
Execute the testing framework using a clean container build:
```bash
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest python -m pytest -q
```
