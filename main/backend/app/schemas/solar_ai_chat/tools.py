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
            "peak hours, or short-term forecasts. If the user specifies a timeframe (e.g. 'last week'), pass the number of days."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe_days": {
                    "type": "integer",
                    "description": "Number of days to look back (default is 30). For example, 7 for 'tuần qua' or 'last week'."
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
            "Retrieve a comprehensive report for a specific day: "
            "energy (MWh), shortwave radiation (W/m2), AQI, "
            "temperature, wind. Can return data for ALL stations or a SINGLE station. "
            "Use this when the user requests a report for a specific day, "
            "or asks about a specific station's data on a particular date. "
            "If the user mentions a station name or code, pass it as station_name. "
            "DO NOT use this tool for requests over the last X days/3 days/general trends."
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
            "Search for incident reports, equipment manuals, "
            "or model changelogs. Use this when the query is related "
            "to documentation, reports, manuals, or changelog history."
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
            "Dynamically query Gold-layer KPI mart tables. "
            "Use this when the user asks about specific KPIs such as AQI impact, "
            "daily energy KPIs, forecast accuracy, system-wide KPIs, or weather impact. "
            "The schema is dynamically discovered, so you must interpret whatever columns are returned."
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
                    "description": "Short name of the KPI mart table.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": "Optional date filter (YYYY-MM-DD).",
                },
                "station_name": {
                    "type": "string",
                    "description": "Optional station/facility filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default 30, max 100).",
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
    "get_facility_info": "facility_info",
    "search_documents": "general",
    "query_gold_kpi": "energy_performance",
}

