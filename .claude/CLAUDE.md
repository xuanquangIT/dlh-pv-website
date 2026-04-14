# CLAUDE.md

## Purpose
This file documents the current scope and architecture of the PV Lakehouse website project as implemented today.

It is intentionally as-is documentation (no refactor roadmap, no redesign plan).


## Project Snapshot
- Project: PV Lakehouse Web
- Stack: FastAPI backend + Jinja2 server-rendered frontend + vanilla JavaScript widgets/pages
- Primary domain: Solar energy operations, analytics, model monitoring, and assistant chat
- Hosting model (local/dev): Single Python process serving APIs, HTML templates, and static assets


## High-Level Architecture
- Web/API framework: FastAPI app in main/backend/app/main.py
- UI rendering: Jinja2 templates from main/frontend/templates
- Static assets: main/frontend/static (CSS and JS)
- Auth/session model: HttpOnly cookie JWT (set by /auth/login)
- Metadata and chat persistence: PostgreSQL (Neon-compatible)
- Analytical data source: Databricks (Unity Catalog catalog/schema tables)
- Dashboard embedding: Power BI embed token flow (with mock fallback when credentials are absent)
- AI layer: Solar AI Chat with intent routing, LLM tool-calling, deterministic fallback summaries, optional RAG, optional web search grounding


## Runtime Data Sources
1. PostgreSQL (application metadata)
- auth_roles
- auth_users
- chat_sessions
- chat_messages
- rag_documents (pgvector)

2. Databricks (analytics and operations)
- Job orchestration APIs via databricks-sdk WorkspaceClient
- SQL data retrieval via databricks-sql-connector (Silver/Gold tables)

3. LLM providers (configurable)
- OpenAI-compatible
- Anthropic-compatible
- Gemini-compatible

4. Optional external APIs
- Power BI REST API (embed token + report URL)
- Tavily-compatible web search endpoint for concept grounding


## Backend Composition

### App Entry and Router Registration
FastAPI app factory is in main/backend/app/main.py.

Routers included by app:
- auth
- dashboard
- data_pipeline
- data_quality
- ml_training
- model_registry
- forecast
- analytics
- solar_ai_chat
- frontend

Global behavior:
- Static mount: /static -> main/frontend/static
- 401 HTML requests are redirected to /login


### Core Layers
- main/backend/app/api: HTTP routes and dependency guards
- main/backend/app/services: business logic and external integrations
- main/backend/app/repositories: data access abstractions
- main/backend/app/schemas: Pydantic request/response models
- main/backend/app/core: settings and security helpers
- main/backend/app/db: SQLAlchemy models and SessionLocal


### Authentication and Authorization
Auth flow:
- POST /auth/login validates credentials and sets JWT cookie
- GET/POST /auth/logout clears cookie and redirects to /login
- Token stored in configured cookie name (default pv_access_token)

Dependencies:
- get_current_user reads cookie, decodes JWT, loads user from repository
- require_role enforces role-based access per route

Roles in system:
- admin
- data_engineer
- ml_engineer
- analyst
- system

Chat role mapping nuance:
- analyst is mapped to data_analyst for chat role enum compatibility


## API Surface (Current)

### Frontend Page Routes
- GET /login
- GET /logout
- GET /dashboard
- GET /pipeline
- GET /quality
- GET /training
- GET /registry
- GET /forecast
- GET /analytics
- GET /settings/accounts
- GET /solar-chat
- GET /solar-ai-chat redirects to /solar-chat

Home route:
- GET / (authenticated) renders module card landing page


### Auth API
- POST /auth/login
- GET or POST /auth/logout
- GET /auth/me
- POST /auth/register (admin)
- GET /auth/users (admin)
- POST /auth/users (admin)
- PATCH /auth/users/{user_id}/status (admin)
- PATCH /auth/users/{user_id}/password (admin)


### Dashboard API
- GET /dashboard/summary
- GET /dashboard/embed-info


### Data Pipeline API
- GET /data-pipeline/status
- GET /data-pipeline/jobs
- GET /data-pipeline/jobs/runs
- POST /data-pipeline/jobs/run
- GET /data-pipeline/jobs/runs/{run_id}
- POST /data-pipeline/jobs/runs/{run_id}/cancel


### Data Quality API
- GET /data-quality/summary
- GET /data-quality/facility-scores
- GET /data-quality/recent-issues
- GET /data-quality/heatmap-data


### ML / Registry / Forecast / Analytics APIs
- GET /ml-training/monitoring
- GET /model-registry/models-list
- GET /forecast/summary-kpi
- GET /forecast/daily
- GET /analytics/query-history


### Solar AI Chat API
- GET /solar-ai-chat/topics
- POST /solar-ai-chat/query
- POST /solar-ai-chat/query/benchmark
- POST /solar-ai-chat/query/benchmark/model-only

Session management:
- POST /solar-ai-chat/sessions
- GET /solar-ai-chat/sessions
- GET /solar-ai-chat/sessions/{session_id}
- DELETE /solar-ai-chat/sessions/{session_id}
- PATCH /solar-ai-chat/sessions/{session_id}/title
- POST /solar-ai-chat/sessions/{session_id}/rename
- POST /solar-ai-chat/sessions/{session_id}/fork

RAG admin endpoints:
- POST /solar-ai-chat/documents/ingest
- GET /solar-ai-chat/documents/stats
- DELETE /solar-ai-chat/documents/{source_file}


## Solar AI Chat Internal Architecture

### Orchestration Flow
Core service: main/backend/app/services/solar_ai_chat/chat_service.py

Processing path:
1. Load chat history (if session provided)
2. Attempt tool-calling path if model router is available
3. If tool-calling unsupported/unavailable, switch to deterministic regex/intent routing
4. Retrieve metrics from repository/tool execution
5. Generate response via model or deterministic fallback summary
6. Persist user + assistant messages to history backend
7. Return response with metadata:
- topic
- key_metrics
- sources
- model_used
- fallback_used
- latency_ms
- intent_confidence
- warning_message
- thinking_trace (summary + steps + trace_id)


### Intent and Tooling
- Vietnamese/English keyword and semantic routing: intent_service.py
- Extreme metric parser (AQI/energy/weather + timeframe/date/hour): nlp_parser.py
- Tool registry: schemas/solar_ai_chat/tools.py
- Tool dispatch and RBAC enforcement: tool_executor.py

Tool-supported analytics include:
- system overview
- energy performance
- ML model info
- pipeline status
- forecast 72h
- data quality issues
- facility info
- extreme AQI/energy/weather
- station daily report
- search_documents (RAG)


### LLM Router
LLM client: llm_client.py

Capabilities:
- Provider-agnostic payload conversion (openai, anthropic, gemini)
- Primary/fallback model routing
- Temporary unavailable cooldown handling
- Tool-call parsing across providers
- Explicit detection of invalid tool invocation errors


### History Backends
Selectable via SOLAR_CHAT_HISTORY_BACKEND:
- postgres -> PostgresChatHistoryRepository
- databricks (default) -> DatabricksChatHistoryRepository

Postgres history:
- Uses SQLAlchemy models chat_sessions/chat_messages
- Ensures owner user exists in local auth_users for FK integrity

Databricks history:
- Writes/reads chat tables in Unity Catalog app schema
- Resolves candidate app catalog names and uses quoted identifiers


### RAG Subsystem
- Embeddings client: GeminiEmbeddingClient
- Ingestion service: RagIngestionService
- Vector store: VectorRepository (pgvector in PostgreSQL)

RAG constraints:
- Ingestion endpoint restricted to files inside configured data root boundary
- Doc types accepted:
  - incident_report
  - equipment_manual
  - model_changelog


### Optional Web Search Grounding
- WebSearchClient uses Tavily-compatible POST API
- Used selectively for external definition/grounding scenarios
- Returns vetted snippets/citations, then blends into final response path


## Frontend Architecture

### Template Structure
- Base landing shell: main/frontend/templates/base.html
- Portal shell: main/frontend/templates/platform_portal/base.html
- Feature pages under main/frontend/templates/platform_portal

Portal shell includes:
- Sidebar and topbar components
- Theme initialization and persisted light/dark mode
- Shared scripts for charts, chat, and common behaviors


### Frontend Page Modules
Page JS modules in main/frontend/static/js/platform_portal:
- data_pipeline.js
- data_quality.js
- training.js
- registry.js
- forecast.js
- accounts_page.js
- solar_chat_page.js
- chatbot_widget.js
- chatbot-bubble.js (legacy/alternate panel behavior)
- common.js
- charts.js


### Chat UI Surfaces
There are two chat experiences in frontend assets:
1. Full Solar Chat page (/solar-chat)
- Rich conversation list
- Project grouping in localStorage
- Rename/delete/fork workflows
- Session-scoped API calls

2. Floating chatbot panel (portal-level)
- Lightweight assistant panel across non-chat pages
- Reuses shared SolarChatApi from solar_chat_page.js


## Data Schema and Bootstrap
Bootstrap SQL file:
- main/002-create-lakehouse-tables.sql

Creates/extensions:
- CREATE EXTENSION vector
- CREATE EXTENSION pgcrypto
- rag_documents (vector(3072), source_file + chunk_index unique)
- auth_roles and auth_users (seeded demo accounts)
- chat_sessions and chat_messages (+ indexes)


## Configuration Model
Primary settings in main/backend/app/core/settings.py.

Environment files searched:
- .env at project root
- main/docker/.env
- plus environment-specific files for Power BI settings

Key config groups:
- Auth: AUTH_SECRET_KEY, AUTH_COOKIE_* settings
- Database: DATABASE_URL or POSTGRES_*
- Databricks: DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_SQL_HTTP_PATH or DATABRICKS_WAREHOUSE_ID
- UC naming: UC_CATALOG, UC_SILVER_SCHEMA, UC_GOLD_SCHEMA, UC_APP_CATALOG, UC_APP_SCHEMA
- Solar chat LLM: SOLAR_CHAT_LLM_API_FORMAT, SOLAR_CHAT_LLM_API_KEY, SOLAR_CHAT_PRIMARY_MODEL, SOLAR_CHAT_FALLBACK_MODEL, SOLAR_CHAT_LLM_BASE_URL
- Embeddings/RAG: SOLAR_CHAT_EMBEDDING_*, SOLAR_CHAT_RAG_*
- History backend: SOLAR_CHAT_HISTORY_BACKEND
- Web search: SOLAR_CHAT_WEBSEARCH_*
- Power BI: POWERBI_*


## Operational Scripts and Test Assets

### Utility Scripts
- main/backend/scripts/solar_chat_perf_cli.py
  - Benchmarks chat, full pipeline, or model-only latency

- main/backend/scripts/solar_chat_accuracy_suite.py
  - Bilingual and multi-turn accuracy/regression suite
  - Optional Databricks cross-validation
  - Markdown/JSON report output


### Test Layout
- Unit tests: main/backend/tests/unit
- Integration tests: main/backend/tests/integration

Current test focus areas include:
- auth login flow and cookie behavior
- dashboard embed API guardrails
- role permission matrix
- intent routing and parser logic
- llm client/tool-calling behavior
- RAG and vector repository behavior
- prompt builder and fallback response guards


## Local Runtime Notes
- Standard local run command uses uvicorn with app-dir main/backend
- Frontend templates/static are served by the same FastAPI process
- Local default in README expects PostgreSQL-backed history and Databricks-backed analytics
- Docker and Trino are not required for the default local flow in this project README


## Scope Boundary
This document captures current implementation scope and architecture only.
It does not include refactor planning, migration strategy, or redesign proposals.