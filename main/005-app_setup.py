# Databricks notebook source
"""Create Databricks Unity Catalog objects for app metadata tables.
Owner: Data Engineer
"""

# COMMAND ----------
# MAGIC %md
# MAGIC ### App Metadata Setup
# MAGIC Initializes Databricks catalog/schema and app operational tables.
# MAGIC
# MAGIC **Creates**
# MAGIC - `dlh_web.app.auth_roles`
# MAGIC - `dlh_web.app.auth_users`
# MAGIC - `dlh_web.app.chat_sessions`
# MAGIC - `dlh_web.app.chat_messages`
# MAGIC - `dlh_web.app.rag_documents`
# MAGIC
# MAGIC This notebook is idempotent and safe to rerun.

from __future__ import annotations

from pyspark.sql import SparkSession


TABLE_DDLS = [
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.auth_roles (
      id STRING NOT NULL,
      name STRING NOT NULL,
      description STRING,
      created_at TIMESTAMP NOT NULL DEFAULT current_timestamp()
    )
    USING DELTA
    COMMENT 'Application role dictionary for auth and RBAC'
    """,
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.auth_users (
      id STRING NOT NULL,
      username STRING NOT NULL,
      email STRING NOT NULL,
      hashed_password STRING NOT NULL,
      full_name STRING,
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      role_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT current_timestamp()
    )
    USING DELTA
    COMMENT 'Application users for portal authentication'
    """,
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.chat_sessions (
      session_id STRING NOT NULL,
      title STRING NOT NULL,
      role STRING NOT NULL,
      owner_user_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
      updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp()
    )
    USING DELTA
    COMMENT 'Solar AI Chat conversation sessions'
    """,
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.chat_messages (
      id STRING NOT NULL,
      session_id STRING NOT NULL,
      sender STRING NOT NULL,
      content STRING NOT NULL,
      timestamp TIMESTAMP NOT NULL DEFAULT current_timestamp(),
      topic STRING,
      sources ARRAY<STRUCT<
        layer: STRING,
        dataset: STRING,
        data_source: STRING
      >>,
      created_at TIMESTAMP NOT NULL DEFAULT current_timestamp()
    )
    USING DELTA
    COMMENT 'Solar AI Chat message history'
    """,
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.rag_documents (
      id STRING,
      doc_type STRING NOT NULL,
      source_file STRING NOT NULL,
      chunk_index INT NOT NULL,
      content STRING NOT NULL,
      embedding ARRAY<FLOAT>,
      created_at TIMESTAMP NOT NULL DEFAULT current_timestamp()
    )
    USING DELTA
    COMMENT 'RAG document chunks and embeddings for Solar AI Chat'
    """,
]

SEED_ROLES_SQL = """
MERGE INTO dlh_web.app.auth_roles AS t
USING (
  SELECT * FROM VALUES
    ('admin', 'Manager', 'Manager / Approver'),
    ('data_engineer', 'Data Engineer', 'Pipeline Owner'),
    ('ml_engineer', 'ML Engineer', 'Model Developer'),
    ('analyst', 'Analyst', 'Data Consumer'),
    ('system', 'System', 'Auto Scheduler')
) AS s(id, name, description)
ON t.id = s.id
WHEN MATCHED THEN UPDATE SET
  t.name = s.name,
  t.description = s.description
WHEN NOT MATCHED THEN INSERT (id, name, description)
VALUES (s.id, s.name, s.description)
"""

SEED_USERS_SQL = """
MERGE INTO dlh_web.app.auth_users AS t
USING (
  SELECT * FROM VALUES
    (
      '00000000-0000-0000-0000-000000000001',
      'admin',
      'admin@pvlakehouse.local',
      '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2',
      'Platform Admin',
      TRUE,
      'admin'
    ),
    (
      '00000000-0000-0000-0000-000000000002',
      'data_eng',
      'de@pvlakehouse.local',
      '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2',
      'Data Engineer',
      TRUE,
      'data_engineer'
    ),
    (
      '00000000-0000-0000-0000-000000000003',
      'ml_eng',
      'ml@pvlakehouse.local',
      '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2',
      'ML Engineer',
      TRUE,
      'ml_engineer'
    ),
    (
      '00000000-0000-0000-0000-000000000004',
      'analyst1',
      'analyst@pvlakehouse.local',
      '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2',
      'Data Analyst',
      TRUE,
      'analyst'
    ),
    (
      '00000000-0000-0000-0000-000000000005',
      'system_bot',
      'system@pvlakehouse.local',
      '$2b$12$wEg.VLiJ8wJINLWXogZSIuC.Q56IhGQ1i7cj87vBuDbnTsYx.wTz2',
      'Auto Scheduler',
      TRUE,
      'system'
    )
) AS s(id, username, email, hashed_password, full_name, is_active, role_id)
ON t.username = s.username
WHEN MATCHED THEN UPDATE SET
  t.email = s.email,
  t.hashed_password = s.hashed_password,
  t.full_name = s.full_name,
  t.is_active = s.is_active,
  t.role_id = s.role_id
WHEN NOT MATCHED THEN INSERT (
  id,
  username,
  email,
  hashed_password,
  full_name,
  is_active,
  role_id
)
VALUES (
  s.id,
  s.username,
  s.email,
  s.hashed_password,
  s.full_name,
  s.is_active,
  s.role_id
)
"""


def get_spark_session(app_name: str) -> SparkSession:
    """Return an active SparkSession for notebook or local Databricks Connect runs."""
    return SparkSession.builder.appName(app_name).getOrCreate()


def main() -> None:
    spark_session = get_spark_session("pv-app-metadata-setup")

    spark_session.sql("CREATE CATALOG IF NOT EXISTS dlh_web")
    spark_session.sql("CREATE SCHEMA IF NOT EXISTS dlh_web.app")
    spark_session.sql("USE CATALOG dlh_web")

    for ddl in TABLE_DDLS:
        spark_session.sql(ddl)

    spark_session.sql(SEED_ROLES_SQL)
    spark_session.sql(SEED_USERS_SQL)
    print("App metadata tables created successfully")


# COMMAND ----------
if __name__ == "__main__":
    main()
