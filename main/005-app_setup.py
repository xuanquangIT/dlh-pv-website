# Databricks notebook source
"""Create Databricks Unity Catalog objects for app metadata tables.
Owner: Data Engineer
"""

# COMMAND ----------

# DBTITLE 1,Cell 2
from __future__ import annotations

from pyspark.sql import SparkSession


TABLE_DDLS = [
    """
    CREATE TABLE IF NOT EXISTS dlh_web.app.auth_roles (
      id STRING NOT NULL,
      name STRING NOT NULL,
      description STRING,
      created_at TIMESTAMP NOT NULL
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
      is_active BOOLEAN NOT NULL,
      role_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL
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
      created_at TIMESTAMP NOT NULL,
      updated_at TIMESTAMP NOT NULL
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
      timestamp TIMESTAMP NOT NULL,
      topic STRING,
      sources ARRAY<STRUCT<
        layer: STRING,
        dataset: STRING,
        data_source: STRING
      >>,
      created_at TIMESTAMP NOT NULL
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
      created_at TIMESTAMP NOT NULL
    )
    USING DELTA
    COMMENT 'RAG document chunks and embeddings for Solar AI Chat'
    """,
]

SEED_ROLES_SQL = """
MERGE INTO dlh_web.app.auth_roles AS t
USING (
  SELECT col1 as id, col2 as name, col3 as description, current_timestamp() as created_at FROM VALUES
    ('admin', 'Manager', 'Manager / Approver'),
    ('data_engineer', 'Data Engineer', 'Pipeline Owner'),
    ('ml_engineer', 'ML Engineer', 'Model Developer'),
    ('analyst', 'Analyst', 'Data Consumer'),
    ('system', 'System', 'Auto Scheduler')
) AS s
ON t.id = s.id
WHEN MATCHED THEN UPDATE SET
  t.name = s.name,
  t.description = s.description
WHEN NOT MATCHED THEN INSERT *
"""

SEED_USERS_SQL = """
MERGE INTO dlh_web.app.auth_users AS t
USING (
  SELECT 
    col1 as id,
    col2 as username,
    col3 as email,
    col4 as hashed_password,
    col5 as full_name,
    col6 as is_active,
    col7 as role_id,
    current_timestamp() as created_at
  FROM VALUES
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
) AS s
ON t.username = s.username
WHEN MATCHED THEN UPDATE SET
  t.email = s.email,
  t.hashed_password = s.hashed_password,
  t.full_name = s.full_name,
  t.is_active = s.is_active,
  t.role_id = s.role_id
WHEN NOT MATCHED THEN INSERT *
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
