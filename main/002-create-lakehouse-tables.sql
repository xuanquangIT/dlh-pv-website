-- Local PostgreSQL bootstrap for authentication and RAG features.
-- Silver and Gold datasets are queried from Iceberg via Trino and are not created here.
-- RAG: pgvector extension and document chunks
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS rag_documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type     VARCHAR(50)  NOT NULL,
    source_file  VARCHAR(500) NOT NULL,
    chunk_index  INTEGER      NOT NULL,
    content      TEXT         NOT NULL,
    embedding    vector(3072),
    created_at   TIMESTAMPTZ  DEFAULT now(),
    UNIQUE(source_file, chunk_index)
);

-- NOTE: Neon pgvector currently does not allow ANN index creation for 3072 dimensions.
-- Keep table functional without ANN index; similarity search will use sequential scan.

-- Authentication Roles
CREATE TABLE IF NOT EXISTS auth_roles (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

-- Authentication Users 
CREATE TABLE IF NOT EXISTS auth_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    hashed_password VARCHAR(200) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    role_id VARCHAR(50) NOT NULL REFERENCES auth_roles(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Seed Default Roles
INSERT INTO auth_roles (id, name, description) VALUES
    ('admin', 'Manager', 'Manager / Approver'),
    ('data_engineer', 'Data Engineer', 'Pipeline Owner'),
    ('ml_engineer', 'ML Engineer', 'Model Developer'),
    ('analyst', 'Analyst', 'Data Consumer'),
    ('system', 'System', 'Auto Scheduler')
ON CONFLICT (id) DO NOTHING;

-- Seed Admin & Demo Users (password for all is 'admin123' generated with bcrypt)
INSERT INTO auth_users (id, username, email, hashed_password, full_name, role_id) VALUES
    (gen_random_uuid(), 'admin', 'admin@pvlakehouse.local', '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2', 'Platform Admin', 'admin'),
    (gen_random_uuid(), 'data_eng', 'de@pvlakehouse.local', '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2', 'Data Engineer', 'data_engineer'),
    (gen_random_uuid(), 'ml_eng', 'ml@pvlakehouse.local', '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2', 'ML Engineer', 'ml_engineer'),
    (gen_random_uuid(), 'analyst1', 'analyst@pvlakehouse.local', '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2', 'Data Analyst', 'analyst'),
    (gen_random_uuid(), 'system_bot', 'system@pvlakehouse.local', '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2', 'Auto Scheduler', 'system')
ON CONFLICT (username) DO NOTHING;

-- Solar AI Chat history
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id     VARCHAR(12) PRIMARY KEY,
    title          VARCHAR(200) NOT NULL,
    role           VARCHAR(50) NOT NULL,
    owner_user_id  UUID NOT NULL REFERENCES auth_users(id),
    created_at     TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at     TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_user_id
    ON chat_sessions (owner_user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         VARCHAR(12) PRIMARY KEY,
    session_id VARCHAR(12) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    sender     VARCHAR(20) NOT NULL,
    content    TEXT NOT NULL,
    "timestamp" TIMESTAMPTZ DEFAULT now() NOT NULL,
    topic      VARCHAR(50),
    sources    JSONB,
    thinking_trace JSONB,  -- persisted ThinkingTrace: {summary, steps, trace_id}
    key_metrics    JSONB,  -- compact tool outputs used to rebuild viz (data_table/chart/kpi_cards) on fetch
    viz_requested  BOOLEAN DEFAULT FALSE NOT NULL  -- whether user asked for a chart (gates chart rendering on hydrate)
);

-- Forward-compatible add for databases created before key_metrics/viz_requested existed.
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS key_metrics JSONB;
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS viz_requested BOOLEAN DEFAULT FALSE NOT NULL;
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS viz_payload JSONB;  -- exact rendered viz snapshot (data_table/chart/kpi_cards) for faithful reload

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id);
