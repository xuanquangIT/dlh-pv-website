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
MAX_HISTORY_IN_PROMPT = 6
# Maximum characters from a single past message to include
MAX_MESSAGE_PREVIEW_LENGTH = 300


def format_source_text(sources: list[SourceMetadata]) -> str:
    return ", ".join(f"{s.layer}:{s.dataset}" for s in sources)


def build_tool_messages(
    request_message: str,
    role_value: str,
    history: list[ChatMessage] | None = None,
) -> list[dict[str, object]]:
    """Build the Gemini messages list for a tool-calling request."""
    messages: list[dict[str, object]] = []
    system_text = (
        "Ban la tro ly Solar AI Chat cho nguoi dung khong ky thuat. "
        "Hay tra loi bang tieng Viet ngan gon, ro rang, toi da 5 cau. "
        f"Vai tro nguoi dung: {role_value}. "
        "Su dung tool de lay du lieu truoc khi tra loi."
    )
    messages.append({"role": "user", "parts": [{"text": system_text}]})
    messages.append({"role": "model", "parts": [{"text": "Da hieu. Toi se su dung tool de tra loi."}]})

    if history:
        for msg in history[-MAX_HISTORY_IN_PROMPT:]:
            role = "user" if msg.sender == "user" else "model"
            messages.append({"role": role, "parts": [{"text": msg.content[:MAX_MESSAGE_PREVIEW_LENGTH]}]})

    messages.append({"role": "user", "parts": [{"text": request_message}]})
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
        "Ban la tro ly Solar AI Chat cho nguoi dung khong ky thuat. "
        "Hay tra loi bang tieng Viet ngan gon, ro rang, toi da 5 cau. "
        "Can neu chi so chinh, y nghia ngu canh va nhac nguon du lieu Silver/Gold.",
    ]

    if history:
        parts.append("Lich su hoi thoai gan day:")
        for msg in history[-MAX_HISTORY_IN_PROMPT:]:
            label = "Nguoi dung" if msg.sender == "user" else "Tro ly"
            parts.append(f"  {label}: {msg.content[:MAX_MESSAGE_PREVIEW_LENGTH]}")

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
    source_text = format_source_text(sources)

    if topic is ChatTopic.GENERAL:
        return (
            "Xin chao! Toi la tro ly Solar AI Chat. Ban co the hoi toi ve: "
            "tong quan he thong, hieu suat nang luong, mo hinh ML, "
            "trang thai pipeline, du bao 72 gio, hoac chat luong du lieu. "
            f"Nguon: {source_text}."
        )

    if topic is ChatTopic.SYSTEM_OVERVIEW:
        return (
            "Tổng quan hệ thống: sản lượng hiện tại "
            f"{metrics.get('production_output_mwh', 0)} MWh, R-squared "
            f"{metrics.get('r_squared', 0)}, điểm chất lượng dữ liệu "
            f"{metrics.get('data_quality_score', 0)} và số cơ sở "
            f"{metrics.get('facility_count', 0)}. Nguồn: {source_text}."
        )

    if topic is ChatTopic.ENERGY_PERFORMANCE and "extreme_metric" not in metrics:
        return (
            "Hiệu suất năng lượng: top cơ sở theo sản lượng và các giờ cao điểm đã được tổng hợp, "
            f"dự báo ngày mai khoảng {metrics.get('tomorrow_forecast_mwh', 0)} MWh. "
            f"Nguồn: {source_text}."
        )

    if topic is ChatTopic.ML_MODEL:
        comparison = metrics.get("comparison", {})
        return (
            "Mô hình GBT-v4.2 đang dùng bộ tham số chuẩn, so sánh với v4.1 cho thấy delta R-squared "
            f"{comparison.get('delta_r_squared', 0)}. Nguồn: {source_text}."
        )

    if topic is ChatTopic.PIPELINE_STATUS:
        return (
            "Pipeline đang được theo dõi theo từng stage với ETA và cảnh báo chất lượng dữ liệu. "
            f"Số cảnh báo hiện tại: {len(metrics.get('alerts', []))}. Nguồn: {source_text}."
        )

    if topic is ChatTopic.FORECAST_72H:
        return (
            "Dự báo 72 giờ đã sẵn sàng theo từng ngày với khoảng tin cậy. "
            f"Số mốc dự báo: {len(metrics.get('daily_forecast', []))}. Nguồn: {source_text}."
        )

    extreme_metric = metrics.get("extreme_metric")
    if extreme_metric is not None:
        return format_extreme_fallback(metrics, extreme_metric, source_text)

    if topic is ChatTopic.DATA_QUALITY_ISSUES:
        return (
            "Các cơ sở có điểm chất lượng thấp đã được xác định kèm nguyên nhân khả dĩ từ cờ chất lượng. "
            f"Nguồn: {source_text}."
        )

    raise ValueError(f"No fallback summary defined for topic '{topic.value}'.")


def format_extreme_fallback(
    metrics: dict[str, Any],
    extreme_metric: str,
    source_text: str,
) -> str:
    query_type = metrics.get("query_type", "")
    station = metrics.get(f"{query_type}_station", "Unknown")
    timeframe_text = describe_timeframe(metrics)

    if extreme_metric == "aqi":
        return (
            f"AQI {query_type} theo truy vấn là "
            f"{metrics.get(f'{query_type}_aqi_value', 0)} tại trạm {station} "
            f"{timeframe_text}. "
            f"Phân loại AQI: {metrics.get(f'{query_type}_aqi_category', 'Unknown')}. "
            f"Nguồn: {source_text}."
        )

    if extreme_metric == "energy":
        return (
            f"Sản lượng năng lượng {query_type} là "
            f"{metrics.get(f'{query_type}_energy_mwh', 0)} MWh "
            f"tại trạm {station} {timeframe_text}. Nguồn: {source_text}."
        )

    return (
        f"Chỉ số thời tiết {metrics.get('weather_metric_label', 'weather')} {query_type} là "
        f"{metrics.get(f'{query_type}_weather_value', 0)} {metrics.get('weather_unit', '')} "
        f"tại trạm {station} {timeframe_text}. Nguồn: {source_text}."
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
