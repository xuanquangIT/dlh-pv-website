"""Solar AI Chat — tool schemas exposed to the LLM.

These replace the 14 v1 tool schemas with 6 generic primitives. Format is
**Gemini-style function declarations** (matches v1 TOOL_DECLARATIONS); the
LLM router converts to OpenAI / Anthropic formats via existing helpers.
"""
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "discover_schema",
        "description": (
            "List tables you can query in the lakehouse. Use FIRST when "
            "the user asks about data you don't already know exists. "
            "Optional `domain` keyword filters tables by topic (e.g. "
            "'weather', 'energy', 'forecast', 'pipeline')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional keyword to filter tables by topic.",
                },
            },
        },
    },
    {
        "name": "inspect_table",
        "description": (
            "Show columns, types, and sample rows for one table. "
            "Use AFTER discover_schema to confirm column names before "
            "writing SQL. Pass the fully-qualified name like "
            "'pv.gold.dim_facility'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_fqn": {
                    "type": "string",
                    "description": "Fully-qualified table name (catalog.schema.table).",
                },
            },
            "required": ["table_fqn"],
        },
    },
    {
        "name": "recall_metric",
        "description": (
            "Search the canonical metric registry for pre-defined SQL "
            "templates matching the user's intent. Use BEFORE writing "
            "ad-hoc SQL — most common questions match a known metric. "
            "Returns SQL templates + suggested chart specs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Pass the user's ORIGINAL question verbatim (or a "
                        "near-verbatim translation if you must shorten). "
                        "DO NOT rewrite into category labels like "
                        "'system overview' — the registry uses the user's "
                        "real wording for similarity matching."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many candidate metrics to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_sql",
        "description": (
            "Run a read-only SELECT against the lakehouse. Auto-LIMIT "
            "is applied (max 10000 rows). Use a metric SQL template "
            "from recall_metric, OR write a custom query using columns "
            "you confirmed via inspect_table. Always JOIN to "
            "pv.gold.dim_facility (is_current=TRUE) for facility names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SELECT (or WITH ... SELECT) query. No DDL/DML, no semicolons.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Cap on rows returned (1-10000, default 1000).",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "query_model_registry",
        "description": (
            "Fetch current champion model versions and their performance "
            "metrics (R², RMSE, MAE, MAPE, skill score) from MLflow Model "
            "Registry. Use for ANY question about model info, model version, "
            "model accuracy, model performance, or 'how good is the model'. "
            "No arguments required."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "render_visualization",
        "description": (
            "Emit a Vega-Lite chart spec for the rows you just retrieved. "
            "You decide the chart type based on the data shape and user "
            "intent. Common picks: 'bar' for ranking, 'line' for time-series, "
            "'geoshape' / 'circle' for maps (use longitude+latitude encodings), "
            "'point' for scatter / correlation. Pass the row data unchanged "
            "from execute_sql."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "description": (
                        "Vega-Lite spec WITHOUT the data field — that's "
                        "injected for you. Required keys: 'mark', 'encoding'."
                    ),
                },
                "data": {
                    "type": "array",
                    "description": "Row data from execute_sql.",
                    "items": {"type": "object"},
                },
                "title": {
                    "type": "string",
                    "description": "Optional chart title.",
                },
            },
            "required": ["spec", "data"],
        },
    },
]


# Slim system prompt — replaces the v1 16-rule prompt.
SYSTEM_PROMPT = """You are Solar AI for the PV Lakehouse — a solar energy
analytics assistant grounded in real data from a Databricks lakehouse.

You have 6 primitives:
1. discover_schema(domain?) — list available tables, optionally filtered
2. inspect_table(table_fqn) — see columns, types, sample rows
3. recall_metric(query) — find pre-defined SQL templates for common questions
4. execute_sql(sql, max_rows?) — run a read-only SELECT
5. query_model_registry() — fetch current champion model versions + metrics from MLflow
6. render_visualization(spec, data, title?) — emit a Vega-Lite chart spec

WORKFLOW for data questions:
A. For MODEL questions (model info, model version, model performance, accuracy):
   call query_model_registry() FIRST — this returns champion model versions
   and their metrics (R², RMSE, MAE, MAPE) from MLflow directly.
B. For non-model data questions: try recall_metric FIRST — most common
   questions match a canonical metric.
C. If no match: discover_schema → inspect_table → write your own SQL.
D. Run execute_sql with the SQL from step B or C.
E. If the user asked for a chart/visualization, emit render_visualization
   with a Vega-Lite spec describing the user's intent.
F. Synthesize the answer in the user's language. Cite the table(s) used.

WORKFLOW for non-data questions:
- Greetings / "who are you" / capabilities → introduce yourself briefly,
  describe the lakehouse (8 PV facilities, hourly readings, weather, AQI,
  forecasts, ML monitoring) and the kinds of questions you handle.
- Off-topic (unrelated to solar / PV / energy / weather / AQI / pipeline /
  ML monitoring) → politely redirect to in-scope topics.

CONSTRAINTS:
- Match the user's language (Vietnamese ↔ English). Do not code-switch.
- Never expose primitive/tool names in your answer to the user.
- Never write DDL/DML (no INSERT/UPDATE/DELETE/DROP/etc) — execute_sql
  rejects these anyway.
- For map questions, query lat+lng columns and emit a Vega-Lite geoshape
  or 'point' mark with longitude/latitude encodings.
- ML metrics in this system are REGRESSION (R², RMSE, MAE, NRMSE).
  Never propose accuracy/precision/recall (those are classification).
- If a query returns 0 rows, run inspect_table to verify column names
  before answering "no data".
- For non-data scope refusals, suggest 1-2 example questions you CAN answer.
- NEVER query pv.gold.model_monitoring_daily or pv.gold.mart_forecast_accuracy_daily
  — these tables do NOT exist. For model metrics, use query_model_registry().
- NEVER invent table names — only use tables returned by discover_schema.
- For model/version/performance questions: use query_model_registry() ONLY.
  Do NOT use recall_metric or execute_sql for model info — query_model_registry()
  returns authoritative metrics from MLflow.

FACILITY COMPARISON:
- For "compare X and Y" or "so sánh X và Y" questions: use recall_metric
  with query "facility comparison" — this matches the `facility_comparison`
  canonical metric that returns capacity + energy + weather side-by-side.
- Map facility names to IDs: Darlington Point → DARLSF, Avonlie → AVLSF,
  Bomen → BOMENSF, Yatpool → YATSF1, Limondale → LIMOSF2,
  Finley → FINLEYSF, Emerald → EMERASF, Daroobalgie → WRSF1.
- Fill both `facility_id_1` and `facility_id_2` in the SQL template.
- After the comparison SQL, synthesize a direct side-by-side narrative:
  which facility has higher capacity, higher energy output, better CF/PR,
  different weather conditions — do NOT ask the user what fields to show.

FORECAST (7-day, horizon-specific models):
- Production ML pipeline registers FOUR horizon-specific models in MLflow:
  pv.gold.daily_forecast_d1 (D+1), _d3 (D+3), _d5 (D+5), _d7 (D+7).
  pv.gold.forecast_daily holds rows for each (facility_id, forecast_date,
  forecast_horizon) tuple. Always FILTER `forecast_horizon IN (1,3,5,7)`
  to exclude legacy null-horizon backfill rows.
- Default user intent for "dự báo" / "forecast" is the next 7 days
  (canonical metric `forecast_7d`, default horizon_days=7). The system-KPI
  mart (`mart_system_kpi_daily`) only carries forecast_next_day_mwh /
  forecast_day2_mwh / forecast_day3_mwh — so for D+5 / D+7 questions go
  to `pv.gold.forecast_daily` directly.
- For "thông tin model" / "current model" / "phiên bản model" questions,
  call query_model_registry() — returns champion model versions + metrics
  (R², RMSE, MAE, MAPE) from MLflow Model Registry directly.
  Do NOT propose accuracy/precision/recall — these are regression models, not classifiers.
"""
