TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "get_system_overview",
        "description": (
            "Lay tong quan he thong PV Lakehouse: tong san luong dien (MWh), "
            "he so R-squared cua mo hinh ML, diem chat luong du lieu trung binh, "
            "va tong so co so/tram nang luong mat troi. "
            "Dung khi nguoi dung hoi ve tong quan, tinh hinh chung, "
            "hoac so lieu toan he thong."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_energy_performance",
        "description": (
            "Lay hieu suat nang luong mat troi: top co so san xuat nhieu nhat, "
            "cac gio cao diem san xuat dien, du bao san luong ngay mai (MWh). "
            "Dung khi nguoi dung hoi ve hieu suat, so sanh cac tram, "
            "gio nao san xuat nhieu nhat, hoac du bao ngan han."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_ml_model_info",
        "description": (
            "Lay thong tin mo hinh ML du bao nang luong: tham so GBT-v4.2 "
            "(max_depth, learning_rate, estimators), so sanh R-squared voi v4.1, "
            "va diem on dinh. Dung khi nguoi dung hoi ve mo hinh, "
            "thuat toan, tham so, do chinh xac, hoac so sanh phien ban."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "Lay trang thai pipeline du lieu: tien do tung tang (Bronze, Silver, Gold), "
            "thoi gian con lai (ETA), va danh sach canh bao chat luong du lieu. "
            "Dung khi nguoi dung hoi ve trang thai pipeline, "
            "loi du lieu, canh bao, hoac tien do xu ly."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forecast_72h",
        "description": (
            "Lay du bao san luong dien mat troi 72 gio toi (3 ngay): "
            "san luong du kien theo ngay va khoang tin cay (confidence interval). "
            "Dung khi nguoi dung hoi ve du bao, san luong ngay mai, "
            "3 ngay toi, hoac xu huong san xuat."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_data_quality_issues",
        "description": (
            "Lay cac van de chat luong du lieu: co so co diem chat luong thap nhat, "
            "nguyen nhan co the (thieu du lieu, thiet bi loi, gia tri bat thuong). "
            "Dung khi nguoi dung hoi ve chat luong du lieu, loi, "
            "co so nao co van de, hoac chi so AQI bat thuong."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_extreme_aqi",
        "description": (
            "Lay chi so AQI (chat luong khong khi) cao nhat hoac thap nhat "
            "theo khoang thoi gian. Dung khi nguoi dung hoi tram nao "
            "co AQI cao nhat/thap nhat trong ngay, tuan, thang, hoac nam."
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
                    "description": (
                        "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Gio cu the (0-23). Chi dung khi timeframe=hour."
                    ),
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_energy",
        "description": (
            "Lay san luong nang luong cao nhat hoac thap nhat "
            "theo khoang thoi gian. Dung khi nguoi dung hoi tram nao "
            "san xuat nhieu/it nhat, hoac so sanh san luong giua cac tram."
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
                    "description": (
                        "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Gio cu the (0-23). Chi dung khi timeframe=hour."
                    ),
                },
            },
            "required": ["query_type", "timeframe"],
        },
    },
    {
        "name": "get_extreme_weather",
        "description": (
            "Lay chi so thoi tiet cao nhat hoac thap nhat: "
            "nhiet do, toc do gio, gio giat, buc xa mat troi, may. "
            "Dung khi nguoi dung hoi ve thoi tiet tai cac tram."
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
                    "description": (
                        "Ngay neo (YYYY-MM-DD). Bo trong de dung ngay moi nhat."
                    ),
                },
                "specific_hour": {
                    "type": "integer",
                    "description": (
                        "Gio cu the (0-23). Chi dung khi timeframe=hour."
                    ),
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
            "nhiet do, gio. Dung khi nguoi dung yeu cau bao cao, "
            "tong hop so lieu tat ca cac tram theo mot ngay cu the, "
            "hoac muon so sanh hieu suat cac tram trong mot ngay."
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
        "name": "get_facility_info",
        "description": (
            "Lay thong tin chi tiet ve cac tram nang luong mat troi: "
            "ten tram, ma tram, vi tri dia ly (latitude, longitude), "
            "quoc gia, bang/tinh, cong suat lap dat (MW). "
            "Dung khi nguoi dung hoi ve vi tri tram, tram nam o dau, "
            "quoc gia nao, toa do GPS, thong tin co so, "
            "hoac cong suat cua tram/nha may."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facility_name": {
                    "type": "string",
                    "description": (
                        "Ten hoac ma cua tram cu the. "
                        "Bo trong de lay tat ca cac tram."
                    ),
                },
            },
            "required": [],
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
                    "enum": [
                        "incident_report",
                        "equipment_manual",
                        "model_changelog",
                    ],
                    "description": (
                        "Loai tai lieu can tim. Bo trong de tim tat ca."
                    ),
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
    "get_facility_info": "facility_info",
    "search_documents": "general",
}
