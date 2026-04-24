# Government RFQ Capability Matcher

A production-architecture RAG application that maps government RFQ (Request for Quotation) task requirements to a service capability catalog — helping contracting officers and proposal managers find the right capabilities fast.

Built on a fork of the Azure Samples `rag-postgres-openai-python` reference architecture, adapted and re-engineered for a federal professional services domain.

---

## What This Does

Given a natural-language query like *"What capabilities do we have for cloud infrastructure and DevSecOps support?"*, the system:

1. Rewrites the query into structured search arguments using **Claude with tool use** (filtering by classification, category, or free text)
2. Runs a **hybrid search** against a PostgreSQL capability catalog — combining pgvector semantic search with PostgreSQL full-text search, fused via RRF (Reciprocal Rank Fusion)
3. Returns grounded, cited answers using **Claude claude-opus-4-7** with source attribution per capability

The service catalog covers 15 capability areas across IT Services and Professional Services — Custom Software Development, Cloud Infrastructure, DevSecOps, Data Analytics, ML/AI, Cybersecurity, Program Management, and more.

---

## Why This Architecture

### Hybrid search over pure vector search

Vector-only retrieval misses exact-match queries (e.g., a specific NAICS code or exact capability name). Full-text-only retrieval misses semantic variations ("AI/ML" vs "machine learning"). RRF fusion gets the best of both without requiring a re-ranker model — simpler, faster, and more explainable to stakeholders who need to audit retrieval decisions.

### Anthropic for chat, OpenAI for embeddings

Anthropic has no embeddings API. OpenAI's `text-embedding-3-large` at 1024 dimensions hits the right cost/quality point for a catalog this size. Claude handles answer generation and query rewriting — where nuanced instruction-following matters more than raw speed.

### Tool use for query rewriting (not a prompt hack)

The advanced flow uses Claude's native tool use with `tool_choice: forced` to extract structured filter arguments from natural-language queries. This is more reliable than few-shot prompting for filter extraction because it's type-constrained — the model can't hallucinate a filter format that breaks the query builder.

### What was deliberately left out

- **Azure deployment / managed identity** — the original template was Azure-native. Removed in favor of a portable setup that runs against any PostgreSQL instance (local, Docker, Supabase, AWS RDS).
- **OpenAI Agents SDK** — the original used `openai-agents` (OpenAIResponsesModel, Runner, function_tool). Replaced with direct Anthropic SDK calls. The abstraction layer added complexity without adding value for a single-model deployment.
- **Ollama / local model support** — descoped to keep the implementation clean. Adding it back is a one-file change in `openai_clients.py`.

---

## Architecture

```
User query
    │
    ▼
[Advanced Flow: Claude tool use]
    │  Rewrites query → search_query + optional filters
    │  (classification_name, category_name)
    ▼
[PostgresSearcher]
    ├── pgvector cosine similarity search  ─┐
    └── PostgreSQL full-text search        ─┴─► RRF fusion → top-N results
    ▼
[Claude claude-opus-4-7]
    │  System prompt: contracts specialist persona
    │  Sources: ranked capability rows with IDs
    └── Grounded answer with [ID] citations
```

**Stack:**
- Backend: Python / FastAPI
- Database: PostgreSQL + pgvector extension
- Embeddings: OpenAI `text-embedding-3-large` (1024 dimensions)
- Chat / query rewriting: Anthropic `claude-opus-4-7`
- Frontend: React / FluentUI

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- A PostgreSQL instance with pgvector (see options below)
- Anthropic API key
- OpenAI API key (embeddings only)

### Database options

**Option A — Supabase (free, no setup)**
1. Create a free account at [supabase.com](https://supabase.com)
2. Create a project
3. Copy the connection string from Settings → Database

**Option B — Local Docker**
```bash
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
```

### Setup

```bash
# Clone and install
git clone https://github.com/Belacqua/rag-postgres-openai-python
cd rag-postgres-openai-python
cd src/backend
pip install -e ".[dev]"

# Configure environment
cp ../../.env.sample ../../.env
# Edit .env: set POSTGRES_HOST, POSTGRES_PASSWORD, ANTHROPIC_API_KEY, OPENAICOM_KEY

# Initialize database and seed capability catalog
python -m fastapi_app.setup_postgres_database
python -m fastapi_app.setup_postgres_seeddata

# Generate embeddings for all 15 capabilities
python -m fastapi_app.update_embeddings

# Start backend
uvicorn fastapi_app:create_app --factory --reload
```

```bash
# In a second terminal — build and serve frontend
cd src/frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### Environment variables

```bash
# .env
POSTGRES_HOST=your-db-host
POSTGRES_USERNAME=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DATABASE=postgres
POSTGRES_SSL=require          # use 'disable' for local Docker

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_CHAT_MODEL=claude-opus-4-7

OPENAI_EMBED_HOST=openai
OPENAICOM_KEY=sk-...
OPENAICOM_EMBED_MODEL=text-embedding-3-large
OPENAICOM_EMBED_DIMENSIONS=1024
OPENAICOM_EMBEDDING_COLUMN=embedding_3l
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/capabilities/{id}` | Fetch a capability by ID |
| GET | `/similar?id=X&n=5` | Find similar capabilities by vector proximity |
| GET | `/search?query=...` | Hybrid search without chat generation |
| POST | `/chat` | Full RAG pipeline — returns answer + thought steps |
| POST | `/chat/stream` | Streaming version of `/chat` |

The `/chat` response includes a `context.thoughts` array showing the search arguments Claude chose, the retrieved capabilities, and the full prompt — useful for debugging retrieval quality.

---

## What I Would Build Next

- **Evaluation harness** — automated retrieval quality tests using a labeled query set, so chunking/embedding changes can be validated before deployment
- **Prompt caching** — the system prompt and capability catalog are stable across requests; adding Anthropic `cache_control` markers would cut token costs ~80% on repeated queries
- **Confidence scoring** — expose the vector similarity scores in the API response so calling applications can gate on retrieval confidence before generating an answer
- **Batch ingestion pipeline** — replace seed_data.json with a pipeline that reads from the source capability catalog (Excel/database) and handles incremental updates

---

## Background

This started as the [Azure Samples RAG on PostgreSQL](https://github.com/Azure-Samples/rag-postgres-openai-python) reference implementation (climbing gear demo). I adapted it to a government professional services use case and migrated the chat layer from OpenAI's Agents SDK to the Anthropic SDK — a non-trivial change since the original used OpenAI's Responses API (`ResponseFunctionToolCall`, `OpenAIResponsesModel`, `Runner`) which has no direct Anthropic equivalent. The query rewriting and answer generation flows were rewritten using Anthropic tool use and `messages.create()` directly.

The capability catalog is a representative slice of a service catalog I worked with while building AI-enabled proposal tooling for a federal contractor.
