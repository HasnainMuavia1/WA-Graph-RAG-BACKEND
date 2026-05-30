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
    User([User Query]) --> IG["Input Guardrails (agent/guardrails.py)"]
    IG -->|Allowed| Agent["Pydantic AI RAG Agent (agent/agent.py)"]
    IG -->|Blocked| BlockedMsg["Return Guardrail Security Alert"]
    
    Agent -->|1. Loads Context| Memory[(Redis Session Memory)]
    
    %% Agent Brain Pass / Tool Decision
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
    
    FusedDocContext --> PromptBuilder["System Prompt Generator"]
    GraphContext --> PromptBuilder
    Memory --> PromptBuilder
    
    PromptBuilder --> LLM["OpenAI GPT-4o-mini / GPT-4o"]
    LLM --> RawOutput["Raw Agent Response"]
    RawOutput --> OG["Output Guardrails (PII & Leak Redactor)"]
    OG -->|Cleaned Response| UserResponse([Final Response])
```

---

### 2. Session Memory Lifecycle (Redis Cache)
Conversation history is preserved across stateless workers in a high-speed sliding-window Redis structure.

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client (Web / WhatsApp)
    participant API as FastAPI Router
    participant Redis as Redis Cache
    participant DB as Supabase PostgreSQL

    Client->>API: Post Query /chat (session_id = "user_123")
    API->>Redis: Check Session History (key = "uchenab:session:user_123")
    alt Session memory found in Redis
        Redis-->>API: Return JSON string (List of Turn objects)
    else Redis cache empty
        API->>DB: Fetch past transactions from message archives
        DB-->>API: Return past rows
        API->>Redis: Cache parsed turns list in Redis
    end
    API->>API: Execute RAG Prompt (Combine History + New Query)
    API->>Redis: Append Turn (RPUSH User Query + Assistant Response)
    API->>Redis: Trim list to sliding window limit (LTRIM -20 -1)
    API->>Redis: Refresh sliding expiry rolling TTL (EXPIRE 24h)
    API-->>Client: Deliver Message
```

---

### 3. WhatsApp Messaging & Voice Loop
Voice and text events from the Meta Graph API webhook are offloaded immediately to background queues to return a fast `200 OK` handshake response.

```mermaid
graph TD
    User([WhatsApp Mobile User]) -->|Sends Text or Voice Note| Meta["Meta WhatsApp Cloud API"]
    Meta -->|HTTP POST Webhook Event| API["FastAPI Endpoint (/api/v1/whatsapp)"]
    API -->|Instantly returns 200 OK Handshake| Meta
    API -->|Asynchronously Enqueue Task| Broker[(Celery Redis Broker)]
    Broker -->|Picks up Task| Worker["Celery Messaging Worker"]
    
    subgraph Inbound processing
        Worker -->|Check Inbound Type| IsVoice{Is Message Voice Note?}
        IsVoice -->|Yes| Download["Download Voice File (.ogg)"]
        Download --> DG["Deepgram Speech-to-Text API"]
        DG -->|Transcribed Text| Agent["Pydantic AI RAG Agent"]
        IsVoice -->|No| TextOnly["Get Text Content"]
        TextOnly --> Agent
      end
      
    Agent -->|Compute Context| Search[Vector / Graph Retrieve]
    Agent -->|Formulate Answer| Guard["Output Guardrails (Enforce Roman-Urdu)"]
    Guard --> Outbox["Save to wa_messages"]
    Outbox --> SendMeta["Meta Graph API Outbound Call"]
    SendMeta --> User
```

---

### 4. Agent Message vs. Admin Message Flows
AI automated workflows run independently from administrative manual intercepts inside the dashboard portal, unified through the outbound delivery endpoint.

```mermaid
graph TD
    UserStart([User Activity / Interaction])
    
    %% Flows
    UserStart -->|User inputs message| UserIn([User WhatsApp Input])
    UserStart -->|Admin opens chat panel| AdminIn([Admin Dashboard UI])
    
    subgraph Auto Agent Reply Flow
        UserIn --> Webhook["FastAPI Webhook /api/v1/whatsapp"]
        Webhook --> Celery["Celery Task (process_whatsapp_message)"]
        Celery --> RAG["AI Agent (Autonomous Model)"]
        RAG --> SaveAuto["Save Message (sender = agent, direction = outbound)"]
    end

    subgraph Manual Admin Reply Flow
        AdminIn --> Auth["JWT Authenticated API Gated Router"]
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
