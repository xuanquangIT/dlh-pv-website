"""Prompt and summary builder for Solar AI Chat.

Responsible for constructing:
- Tool-call message history for Gemini function-calling API
- RAG/regex-path prompt strings
- Deterministic fallback summaries when LLM is unavailable
"""
from __future__ import annotations

import json
from typing import Any

from app.schemas.solar_ai_chat import ChatMessage, ChatRole, ChatTopic, SourceMetadata

# Maximum messages from history to include in any prompt
MAX_HISTORY_IN_PROMPT = 10
# Recent messages get full content; older ones get truncated
MAX_RECENT_FULL_LENGTH = 500
# Older messages get a shorter preview
MAX_OLD_PREVIEW_LENGTH = 150
# Number of recent messages to keep at full length
NUM_RECENT_FULL = 4


def format_source_text(sources: list[SourceMetadata]) -> str:
    return ", ".join(f"{s.layer}:{s.dataset}" for s in sources)


def _build_system_prompt(role_value: str) -> str:
    """Build a rich system prompt with domain context.

    Does NOT hardcode country/region — the chatbot discovers
    station locations dynamically via get_facility_info tool.
    """
    return (
        "Ban la Solar AI, tro ly thong minh cua he thong PV Lakehouse "
        "- nen tang phan tich du lieu nang luong mat troi.\n\n"
        "HE THONG PV LAKEHOUSE:\n"
        "- Kien truc du lieu: Bronze (raw) -> Silver (clean) -> Gold (dimensional)\n"
        "- Luu tru: Apache Iceberg tren MinIO, truy van qua Trino\n"
        "- Du lieu: nang luong (MWh), thoi tiet (buc xa, nhiet do, gio, may), "
        "chat luong khong khi (AQI), mo hinh ML du bao\n"
        "- Cac tram nang luong mat troi duoc luu trong bang dim_facility "
        "voi toa do GPS (latitude, longitude) va cong suat (MW)\n\n"
        "QUY TAC TRA LOI:\n"
        "1. Tra loi bang tieng Viet, ngan gon, ro rang, toi da 8 cau\n"
        "2. LUON su dung tool de lay du lieu thuc truoc khi tra loi "
        "- KHONG bao gio doan so lieu\n"
        "3. Khi nguoi dung hoi ve vi tri, quoc gia, toa do, "
        "o dau -> dung tool get_facility_info\n"
        "4. Chi dung get_station_daily_report khi nguoi dung yeu cau "
        "bao cao cho MOT ngay cu the (vi du: 2026-04-09, 09/04/2026, hom qua). "
        "Neu cau hoi la x ngay gan nhat/3 ngay/toan bo xu huong, uu tien "
        "get_energy_performance hoac get_forecast_72h va KHONG tu dat ngay.\n"
        "5. Khi nguoi dung hoi follow-up, tham khao lich su "
        "hoi thoai de hieu ngu canh\n"
        "6. Khi nhan duoc toa do GPS (lat/lng), ban bat buoc phai "
        "chuyen doi toa do do thanh thong tin dia ly cu the (bang, "
        "khu vuc, quoc gia...) bang kien thuc san co cua ban. Khong duoc "
        "tu choi hoac tra loi rang ban chi co the cung cap toa do.\n"
        "7. Tuye doi KHONG de cap den cac kien truc du lieu noi bo "
        "(nhu: bang Gold, bang Silver, nguon Bronze, dim_facility, "
        "lh_silver_*, lh_gold_*) hay bat ky tu ngu ky thuat nao lien quan "
        "den CSDL trong cau tra loi cho nguoi dung.\n"
        "8. Neu khong tim thay du lieu, thong bao ro rang thay vi doan\n"
        f"9. Vai tro nguoi dung hien tai: {role_value}\n"
    )


def _format_history_messages(
    history: list[ChatMessage],
) -> list[dict[str, object]]:
    """Format chat history with smart truncation.

    Recent messages get full content, older ones are truncated.
    """
    trimmed = history[-MAX_HISTORY_IN_PROMPT:]
    messages: list[dict[str, object]] = []
    total = len(trimmed)
    for idx, msg in enumerate(trimmed):
        role = "user" if msg.sender == "user" else "model"
        is_recent = (total - idx) <= NUM_RECENT_FULL
        max_len = (
            MAX_RECENT_FULL_LENGTH if is_recent
            else MAX_OLD_PREVIEW_LENGTH
        )
        content = msg.content[:max_len]
        if len(msg.content) > max_len:
            content += "..."
        messages.append(
            {"role": role, "parts": [{"text": content}]},
        )
    return messages


def build_tool_messages(
    request_message: str,
    role_value: str,
    history: list[ChatMessage] | None = None,
) -> list[dict[str, object]]:
    """Build the Gemini messages list for a tool-calling request."""
    messages: list[dict[str, object]] = []
    system_text = _build_system_prompt(role_value)
    messages.append(
        {"role": "user", "parts": [{"text": system_text}]},
    )
    messages.append(
        {"role": "model", "parts": [{
            "text": (
                "Da hieu. Toi la Solar AI, tro ly PV Lakehouse. "
                "Toi se su dung tool de lay du lieu thuc "
                "va tra loi chinh xac bang tieng Viet."
            ),
        }]},
    )

    if history:
        messages.extend(_format_history_messages(history))

    messages.append(
        {"role": "user", "parts": [{"text": request_message}]},
    )
    return messages


def build_prompt(
    user_message: str,
    role: ChatRole,
    topic: ChatTopic,
    metrics: dict[str, Any],
    sources: list[SourceMetadata],
    history: list[ChatMessage] | None = None,
) -> str:
    """Build a plain-text prompt for the RAG/regex path."""
    source_text = format_source_text(sources)
    metrics_json = json.dumps(metrics, ensure_ascii=False)

    parts = [
        _build_system_prompt(role.value),
        "Hay tra loi dua tren du lieu da truy xuat ben duoi. "
        "Can neu chi so chinh va y nghia ngu canh. "
        "Khong hien thi nguon du lieu noi bo trong cau tra loi.",
    ]

    if history:
        parts.append("Lich su hoi thoai gan day:")
        trimmed = history[-MAX_HISTORY_IN_PROMPT:]
        total = len(trimmed)
        for idx, msg in enumerate(trimmed):
            label = "Nguoi dung" if msg.sender == "user" else "Tro ly"
            is_recent = (total - idx) <= NUM_RECENT_FULL
            max_len = (
                MAX_RECENT_FULL_LENGTH if is_recent
                else MAX_OLD_PREVIEW_LENGTH
            )
            content = msg.content[:max_len]
            if len(msg.content) > max_len:
                content += "..."
            parts.append(f"  {label}: {content}")

    parts.extend([
        f"Vai tro nguoi dung: {role.value}",
        f"Chu de: {topic.value}",
        f"Cau hoi: {user_message}",
        f"Nguon du lieu: {source_text}",
        f"Chi so da truy xuat: {metrics_json}",
    ])
    return "\n".join(parts)


def build_fallback_summary(
    topic: ChatTopic,
    metrics: dict[str, Any],
    sources: list[SourceMetadata],
) -> str:
    """Return a deterministic summary when the LLM is unavailable."""

    if topic is ChatTopic.GENERAL:
        return (
            "Xin chao! Toi la tro ly Solar AI Chat. Ban co the hoi toi ve: "
            "tong quan he thong, hieu suat nang luong, mo hinh ML, "
            "trang thai pipeline, du bao 72 gio, hoac chat luong du lieu."
        )

    if topic is ChatTopic.SYSTEM_OVERVIEW:
        return (
            "Tổng quan hệ thống: sản lượng hiện tại "
            f"{metrics.get('production_output_mwh', 0)} MWh, R-squared "
            f"{metrics.get('r_squared', 0)}, điểm chất lượng dữ liệu "
            f"{metrics.get('data_quality_score', 0)} và số cơ sở "
            f"{metrics.get('facility_count', 0)}."
        )

    if topic is ChatTopic.ENERGY_PERFORMANCE and "extreme_metric" not in metrics:
        top_rows = metrics.get("top_facilities", [])
        top_line = ""
        if isinstance(top_rows, list) and top_rows:
            top = top_rows[0] if isinstance(top_rows[0], dict) else {}
            top_name = top.get("facility", "Unknown")
            top_energy = top.get("energy_mwh", 0)
            top_line = f"Trạm có hiệu suất tốt nhất hiện tại là {top_name} ({top_energy} MWh). "
        return (
            top_line
            + "Hiệu suất năng lượng: top cơ sở theo sản lượng và các giờ cao điểm đã được tổng hợp, "
            f"dự báo ngày mai khoảng {metrics.get('tomorrow_forecast_mwh', 0)} MWh."
        )

    if topic is ChatTopic.ML_MODEL:
        comparison = metrics.get("comparison", {})
        return (
            "Mô hình GBT-v4.2 đang dùng bộ tham số chuẩn, so sánh với v4.1 cho thấy delta R-squared "
            f"{comparison.get('delta_r_squared', 0)}."
        )

    if topic is ChatTopic.PIPELINE_STATUS:
        return (
            "Pipeline đang được theo dõi theo từng stage với ETA và cảnh báo chất lượng dữ liệu. "
            f"Số cảnh báo hiện tại: {len(metrics.get('alerts', []))}."
        )

    if topic is ChatTopic.FORECAST_72H:
        return (
            "Dự báo 72 giờ đã sẵn sàng theo từng ngày với khoảng tin cậy. "
            f"Số mốc dự báo: {len(metrics.get('daily_forecast', []))}."
        )

    extreme_metric = metrics.get("extreme_metric")
    if extreme_metric is not None:
        return format_extreme_fallback(metrics, extreme_metric)

    if topic is ChatTopic.DATA_QUALITY_ISSUES:
        return (
            "Các cơ sở có điểm chất lượng thấp đã được xác định "
            "kèm nguyên nhân khả dĩ từ cờ chất lượng."
        )

    if topic is ChatTopic.FACILITY_INFO:
        facilities = metrics.get("facilities", [])
        count = metrics.get("facility_count", len(facilities))
        if not facilities:
            return "Không tìm thấy thông tin trạm phù hợp."
        top_station_name = None
        top_station_capacity = None
        timezone_labels: list[str] = []
        for row in facilities:
            try:
                cap = float(row.get("total_capacity_mw", 0) or 0)
            except (TypeError, ValueError):
                cap = 0.0
            if top_station_capacity is None or cap > top_station_capacity:
                top_station_capacity = cap
                top_station_name = str(row.get("facility_name", "Unknown"))

            tz_name = str(row.get("timezone_name", "")).strip()
            tz_offset = str(row.get("timezone_utc_offset", "")).strip()
            if tz_name and tz_offset:
                timezone_labels.append(f"{tz_name} ({tz_offset})")

        lines = []
        if timezone_labels:
            timezone_text = ", ".join(sorted(set(timezone_labels)))
            lines.append(f"Múi giờ các trạm hiện tại: {timezone_text}.")
        if top_station_name is not None and top_station_capacity is not None:
            lines.append(
                f"Trạm có công suất lớn nhất hiện tại là {top_station_name} ({round(top_station_capacity, 2)} MW)."
            )
        lines.append(f"Tìm thấy {count} trạm năng lượng mặt trời:")
        for f in facilities[:8]:
            name = f.get("facility_name", "Unknown")
            lat = f.get("location_lat", 0)
            lng = f.get("location_lng", 0)
            cap = f.get("total_capacity_mw", 0)
            tz_name = f.get("timezone_name", "N/A")
            tz_offset = f.get("timezone_utc_offset", "N/A")
            lines.append(
                f"  - {name}: tọa độ ({lat}, {lng}), "
                f"Công suất {cap} MW, Múi giờ {tz_name} ({tz_offset})"
            )
        return "\n".join(lines)

    raise ValueError(
        f"No fallback summary for topic '{topic.value}'."
    )


def format_extreme_fallback(
    metrics: dict[str, Any],
    extreme_metric: str,
) -> str:
    query_type = metrics.get("query_type", "")
    station = metrics.get(f"{query_type}_station", "Unknown")
    timeframe_text = describe_timeframe(metrics)

    if extreme_metric == "aqi":
        return (
            f"AQI {query_type} theo truy vấn là "
            f"{metrics.get(f'{query_type}_aqi_value', 0)} tại trạm {station} "
            f"{timeframe_text}. "
            f"Phân loại AQI: {metrics.get(f'{query_type}_aqi_category', 'Unknown')}."
        )

    if extreme_metric == "energy":
        return (
            f"Sản lượng năng lượng {query_type} là "
            f"{metrics.get(f'{query_type}_energy_mwh', 0)} MWh "
            f"tại trạm {station} {timeframe_text}."
        )

    return (
        f"Chỉ số thời tiết {metrics.get('weather_metric_label', 'weather')} {query_type} là "
        f"{metrics.get(f'{query_type}_weather_value', 0)} {metrics.get('weather_unit', '')} "
        f"tại trạm {station} {timeframe_text}."
    )


def describe_timeframe(metrics: dict[str, Any]) -> str:
    timeframe = metrics.get("timeframe", "day")
    period_label = str(metrics.get("period_label", metrics.get("query_date", "N/A")))
    if timeframe == "hour":
        if metrics.get("specific_hour") is not None:
            return f"vào lúc {period_label}"
        return f"trong 1 giờ tại mốc {period_label}"
    if timeframe == "24h":
        return f"trong 24 giờ với mốc {period_label}"
    if timeframe == "week":
        return f"trong tuần {period_label}"
    if timeframe == "month":
        return f"trong tháng {period_label}"
    if timeframe == "year":
        return f"trong năm {period_label}"
    return f"vào ngày {period_label}"
