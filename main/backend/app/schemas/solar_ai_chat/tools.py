TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "get_system_overview",
        "description": (
            "Lay tong quan he thong: san luong, R-squared, "
            "diem chat luong du lieu, so co so."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_energy_performance",
        "description": (
            "Lay hieu suat nang luong: top co so, gio cao diem, "
            "du bao ngay mai."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_ml_model_info",
        "description": (
            "Lay thong tin mo hinh ML: tham so GBT-v4.2, "
            "so sanh voi v4.1."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "Lay trang thai pipeline: tien do tung stage, "
            "ETA va canh bao."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forecast_72h",
        "description": (
            "Lay du bao 72 gio: san luong theo ngay va "
            "khoang tin cay."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_data_quality_issues",
        "description": (
            "Lay van de chat luong du lieu: co so diem thap, "
            "nguyen nhan kha di."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_extreme_aqi",
        "description": (
            "Lay chi so AQI cao nhat hoac thap nhat theo khoang thoi gian."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Cao nhat hoac thap nhat.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Khoang thoi gian truy van.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat.",
                },
                "specific_hour": {
                    "type": "integer",
                    "description": "Gio cu the (0-23). Chi dung khi timeframe=hour.",
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_energy",
        "description": (
            "Lay san luong nang luong cao nhat hoac thap nhat "
            "theo khoang thoi gian."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Cao nhat hoac thap nhat.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Khoang thoi gian truy van.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat.",
                },
                "specific_hour": {
                    "type": "integer",
                    "description": "Gio cu the (0-23). Chi dung khi timeframe=hour.",
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_weather",
        "description": (
            "Lay chi so thoi tiet cao nhat hoac thap nhat: "
            "nhiet do, gio, buc xa, may."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["highest", "lowest"],
                    "description": "Cao nhat hoac thap nhat.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["hour", "day", "24h", "week", "month", "year"],
                    "description": "Khoang thoi gian truy van.",
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
                    "description": "Chi so thoi tiet can truy van.",
                },
                "anchor_date": {
                    "type": "string",
                    "description": "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat.",
                },
                "specific_hour": {
                    "type": "integer",
                    "description": "Gio cu the (0-23). Chi dung khi timeframe=hour.",
                },
            },
            "required": ["query_type", "timeframe", "weather_metric"],
        },
    },
    {
        "name": "get_station_daily_report",
        "description": (
            "Lay bao cao tong hop theo ngay cua tat ca tram: "
            "nang luong (MWh), buc xa mat troi (W/m2), AQI, "
            "nhiet do, gio. Dung khi nguoi dung hoi bao cao, "
            "tong hop, so lieu cac tram theo ngay cu the."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "anchor_date": {
                    "type": "string",
                    "description": "Ngay can bao cao (YYYY-MM-DD).",
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
                        "Danh sach chi so can lay. "
                        "Bo trong de lay tat ca."
                    ),
                },
            },
            "required": ["anchor_date"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Tim kiem tai lieu bao cao su co, huong dan thiet bi, "
            "hoac changelog mo hinh. Dung khi cau hoi lien quan den "
            "tai lieu, bao cao, huong dan, hoac lich su thay doi."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Truy van tim kiem tai lieu.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["incident_report", "equipment_manual", "model_changelog"],
                    "description": "Loai tai lieu can tim. Bo trong de tim tat ca.",
                },
            },
            "required": ["query"],
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
    "search_documents": "general",
}
