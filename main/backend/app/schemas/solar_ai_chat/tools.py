TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "get_system_overview",
        "description": (
            "Retrieve the PV Lakehouse system overview: total energy production (MWh), "
            "ML model R-squared score, average data quality score, "
            "and total number of facilities/solar stations. "
            "Use this when the user asks about the overall system status, "
            "general situation, or system-wide metrics. If a timeframe is specified (e.g. 'today', 'last week'), pass timeframe_days."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe_days": {
                    "type": "integer",
                    "description": "Number of days to look back (default is 30). For example, 1 for 'today', 7 for 'last week'."
                }
            }
        },
    },
    {
        "name": "get_energy_performance",
        "description": (
            "Retrieve solar energy performance: top producing facilities, "
            "peak hour energy production, and tomorrow's energy forecast (MWh). "
            "Use this when the user asks about performance, comparing stations, "
            "peak hours, or short-term forecasts. If the user specifies a timeframe (e.g. 'last week'), pass the number of days. "
            "Choose `focus` to tailor which metrics are returned: "
            "'energy' → only energy production per facility (use for 'top N by energy', 'ranked by output'); "
            "'capacity' → only capacity factor per facility (use for 'capacity factor', 'efficiency'); "
            "'overview' (default) → full summary with energy + capacity + peaks + forecast."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe_days": {
                    "type": "integer",
                    "description": "Number of days to look back (default is 30). For example, 7 for 'tuần qua' or 'last week'."
                },
                "focus": {
                    "type": "string",
                    "enum": ["overview", "energy", "capacity"],
                    "description": (
                        "Which subset of metrics to return. "
                        "'energy' when the user asks about energy output / MWh ranking only. "
                        "'capacity' when the user asks about capacity factor / efficiency only. "
                        "'overview' (default) when the user asks for a general summary."
                    )
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of facilities to return in the breakdown lists. "
                        "Set this whenever the user asks for 'top N' / 'bottom N' / 'top 5' etc. "
                        "Omit for a full 8-facility list."
                    )
                }
            }
        },
    },
    {
        "name": "get_ml_model_info",
        "description": (
            "Retrieve ML energy forecast model information: GBT-v4.2 parameters "
            "(max_depth, learning_rate, estimators), R-squared comparison with v4.1, "
            "and stability score. Use this when the user asks about the ML model, "
            "algorithms, parameters, accuracy, or version comparisons."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "Retrieve the data pipeline status: progress of each layer (Bronze, Silver, Gold), "
            "estimated time of arrival (ETA), and a list of data quality alerts. "
            "Use this when the user asks about the pipeline status, "
            "data errors, alerts, or processing progress."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forecast_72h",
        "description": (
            "Retrieve the 72-hour (3 days) solar energy forecast: "
            "expected production by day and confidence interval. "
            "Use this when the user asks about the forecast, tomorrow's production, "
            "the next 3 days, or production trends."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_data_quality_issues",
        "description": (
            "Retrieve data quality issues: facilities with the lowest quality scores, "
            "and possible causes (missing data, equipment failure, anomalies). "
            "Use this when the user asks about data quality, errors, "
            "which facilities have issues, or abnormal AQI metrics. If a timeframe is specified, pass timeframe_days."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe_days": {
                    "type": "integer",
                    "description": "Number of days to look back (default is 30)."
                }
            }
        },
    },
    {
        "name": "get_extreme_aqi",
        "description": (
            "Retrieve the highest or lowest AQI (Air Quality Index) "
            "for a specified timeframe. Use this when the user asks which station "
            "has the highest/lowest AQI in a day, week, month, or year."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Highest or lowest.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Timeframe for the query.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": (
                        "Anchor date (YYYY-MM-DD). Leave empty to use the latest date."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Specific hour (0-23). Only used when timeframe=hour."
                    ),
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_energy",
        "description": (
            "Retrieve the highest or lowest energy production "
            "for a specified timeframe. Use this when the user asks which station "
            "produced the most/least energy, or to compare production between stations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Highest or lowest.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Timeframe for the query.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": (
                        "Anchor date (YYYY-MM-DD). Leave empty to use the latest date."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Specific hour (0-23). Only used when timeframe=hour."
                    ),
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_weather",
        "description": (
            "Retrieve the highest or lowest weather metric: "
            "temperature, wind speed, wind gusts, shortwave radiation, cloud cover. "
            "Use this when the user asks about the weather at the stations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Highest or lowest.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Timeframe for the query.",
                },
                "weather_metric": {
                    "type": "string",
                    "enum": [
                        "temperature_2m",
                        "wind_speed_10m",
                        "wind_gusts_10m",
                        "shortwave_radiation",
                        "cloud_cover",
                    ],
                    "description": "Weather metric to query.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": (
                        "Anchor date (YYYY-MM-DD). Leave empty to use the latest date."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Specific hour (0-23). Only used when timeframe=hour."
                    ),
                },
            },
            "required": ["query_type", "timeframe", "weather_metric"],
        },
    },
    {
        "name": "get_station_daily_report",
        "description": (
            "Retrieve a snapshot report for a SPECIFIC SINGLE DAY: "
            "energy (MWh), shortwave radiation (W/m2), AQI, temperature, wind. "
            "Can return data for ALL stations or a SINGLE station. "
            "ONLY USE when the user explicitly names a date and wants a snapshot of that day, "
            "e.g. 'báo cáo trạm ngày 19/04', 'daily report for April 19'. "
            "\n\n🚫 DO NOT USE THIS TOOL WHEN: "
            "(a) the user asks about correlation / relationship between metrics "
            "('mối liên hệ giữa X và Y', 'X vs Y', 'how does X affect Y', "
            "'PR vs temperature') — use `query_gold_kpi(table_name='energy')` instead. "
            "(b) the user asks for trends over multiple days / last N days — use "
            "`query_gold_kpi(table_name='energy')` to get multi-day data. "
            "(c) the user asks about `performance_ratio` / `PR` — this tool does NOT "
            "return PR; only `query_gold_kpi('energy')` has `performance_ratio_pct`. "
            "If the user mentions a station name or code, pass it as station_name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "anchor_date": {
                    "type": "string",
                    "description": "Date for the report (YYYY-MM-DD). Leave empty for the latest data date.",
                },
                "station_name": {
                    "type": "string",
                    "description": (
                        "Name or code of a specific station to filter. "
                        "Leave empty to retrieve data for all stations. "
                        "Supports partial matching (e.g. 'Solar' will match 'Solar Farm Alpha')."
                    ),
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "energy_mwh",
                            "shortwave_radiation",
                            "aqi_value",
                            "temperature_2m",
                            "wind_speed_10m",
                            "cloud_cover",
                        ],
                    },
                    "description": (
                        "List of metrics to retrieve. "
                        "Leave empty to retrieve all."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_station_hourly_report",
        "description": (
            "Retrieve HOURLY energy generation for one or more stations on a specific date. "
            "Returns rows with hour (0-23), facility, energy_mwh, capacity_factor_pct. "
            "Use this ONLY when the user asks for an hour-by-hour breakdown within a day "
            "('theo giờ', 'từng giờ', 'hourly', 'by hour'). "
            "DO NOT use this for cross-facility comparisons of totals, averages, or capacity "
            "factor summaries — use `get_energy_performance` instead (with focus='capacity' "
            "for capacity-factor questions, focus='energy' for energy-output rankings). "
            "If anchor_date is omitted, the latest available date for the station is used. "
            "Always pass station_name for single-station requests; leave empty for all stations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station_name": {
                    "type": "string",
                    "description": (
                        "Name or code of a specific station (partial match supported). "
                        "Leave empty to aggregate across all stations."
                    ),
                },
                "anchor_date": {
                    "type": "string",
                    "description": (
                        "Date for the hourly breakdown (YYYY-MM-DD). "
                        "Leave empty to use the latest date with data for that station."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_facility_info",
        "description": (
            "Retrieve detailed information about solar energy facilities: "
            "station name, station code, geographical location (latitude, longitude), "
            "timezone and UTC offset of the station, "
            "country, state/province, installed capacity (MW). "
            "Use this when the user asks about the location of a station, where it is, "
            "its timezone, UTC offset, country, GPS coordinates, facility info, "
            "or the capacity of the station/plant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facility_name": {
                    "type": "string",
                    "description": (
                        "Name or code of a specific station. "
                        "Leave empty to retrieve all stations."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Semantic search over TEXT documents (incident reports, equipment "
            "manuals, model changelogs). Returns text chunks only — NO numeric "
            "data, NO charts will be generated from this tool. \n\n"
            "✅ USE for: conceptual questions ('what is PR?', 'define capacity "
            "factor'), troubleshooting references, changelog lookups. \n"
            "🚫 DO NOT USE for: correlation / ranking / comparison of metrics "
            "('mối liên hệ giữa X và Y', 'so sánh trạm', 'PR vs temperature', "
            "'trạm nào cao nhất'). For those use `query_gold_kpi` or "
            "`get_energy_performance` — they return structured data that "
            "renders charts + tables."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for documents.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": [
                        "incident_report",
                        "equipment_manual",
                        "model_changelog",
                    ],
                    "description": (
                        "Type of document to search for. Leave empty to search all."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_gold_kpi",
        "description": (
            "Dynamically query Gold-layer KPI mart tables. There are EXACTLY 5 valid tables — "
            "you MUST pick one of them, do not invent new names. "
            "Pick based on the user's question: "
            "`energy` → `gold.mart_energy_daily`: the MOST COMPLETE per-facility per-day table. "
            "Contains energy_mwh_daily, performance_ratio_pct, weighted_capacity_factor_pct, "
            "avg_temperature_c, avg_cloud_cover_pct, daily_insolation_kwh_m2, and more. "
            "USE THIS for ANY correlation involving performance ratio (PR) or capacity factor "
            "against weather features — e.g., 'mối liên hệ giữa performance ratio và nhiệt độ', "
            "'PR vs temperature', 'how does temperature affect PR', 'daily energy ranking'. "
            "`weather_impact` → `gold.mart_weather_impact_daily`: weather-band aggregates "
            "(Cloud Band × Temperature Band × Rain Band). Use ONLY when the user asks about "
            "weather conditions broken down by band (e.g., 'performance under cloudy vs clear "
            "weather'), NOT for general PR-vs-temperature correlation. "
            "`aqi_impact` → `gold.mart_aqi_impact_daily`: correlation between air quality (PM2.5, "
            "PM10, AQI) and energy. "
            "`forecast_accuracy` → `gold.mart_forecast_accuracy_daily`: actual vs forecast, MAPE, "
            "MAE, RMSE per day. "
            "`system_kpi` → `gold.mart_system_kpi_daily`: system-wide daily KPIs. "
            "The schema is dynamically discovered — interpret whatever columns are returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "enum": [
                        "aqi_impact",
                        "energy",
                        "forecast_accuracy",
                        "system_kpi",
                        "weather_impact",
                    ],
                    "description": (
                        "MUST be one of: aqi_impact, energy, forecast_accuracy, system_kpi, "
                        "weather_impact. Do NOT pass custom names like "
                        "'performance_ratio_vs_temperature' — use 'weather_impact' instead."
                    ),
                },
                "anchor_date": {
                    "type": "string",
                    "description": (
                        "Optional YYYY-MM-DD date filter. OMIT for correlation / "
                        "relationship / trend queries — they need many days of data, "
                        "and filtering to a single day reduces a scatter to one "
                        "point per facility. Only set when the user explicitly "
                        "names a date (e.g. 'on April 19', 'ngày 19/04')."
                    ),
                },
                "station_name": {
                    "type": "string",
                    "description": "Optional station/facility filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum rows to return (default 100, max 500). For "
                        "correlation / trend queries use 100+ to get enough "
                        "data points across multiple days."
                    ),
                },
            },
            "required": ["table_name"],
        },
    },
]

TOOL_NAME_TO_TOPIC: dict[str, str] = {
    "get_system_overview": "system_overview",
    "get_energy_performance": "energy_performance",
    "get_ml_model_info": "ml_model",
    "get_pipeline_status": "pipeline_status",
    "get_forecast_72h": "forecast_72h",
    "get_data_quality_issues": "data_quality_issues",
    "get_extreme_aqi": "data_quality_issues",
    "get_extreme_energy": "energy_performance",
    "get_extreme_weather": "energy_performance",
    "get_station_daily_report": "energy_performance",
    "get_station_hourly_report": "energy_performance",
    "get_facility_info": "facility_info",
    "search_documents": "general",
    "query_gold_kpi": "energy_performance",
}

