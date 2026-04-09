-- Solar AI Chat history (PostgreSQL backend)
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(12) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    role VARCHAR(50) NOT NULL,
    owner_user_id UUID NOT NULL REFERENCES auth_users(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_user_id
    ON chat_sessions(owner_user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR(12) PRIMARY KEY,
    session_id VARCHAR(12) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    sender VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now(),
    topic VARCHAR(50),
    sources JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_timestamp
    ON chat_messages(session_id, timestamp DESC);
