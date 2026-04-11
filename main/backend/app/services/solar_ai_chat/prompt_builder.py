"""Prompt and summary builder for Solar AI Chat.

Responsible for constructing:
- Tool-call message history for Gemini function-calling API
- RAG/regex-path prompt strings
- Deterministic fallback summaries when LLM is unavailable
"""
from __future__ import annotations

import json
from unicodedata import normalize
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


def _normalize_message_for_matching(message: str) -> str:
    lowered = str(message or "").strip().lower()
    without_marks = normalize("NFD", lowered)
    return "".join(character for character in without_marks if ord(character) < 128)


def _is_performance_ratio_request(message: str | None) -> bool:
    normalized = _normalize_message_for_matching(message or "")
    if not normalized:
        return False
    performance_markers = (
        "performance ratio",
        "capacity factor",
        "he so cong suat",
        "ti le hieu suat",
        "chi so hieu suat",
    )
    return any(marker in normalized for marker in performance_markers)


def _is_definition_request(message: str | None) -> bool:
    normalized = _normalize_message_for_matching(message or "")
    if not normalized:
        return False
    markers = (
        "la gi",
        "nghia la gi",
        "dinh nghia",
        "giai thich",
        "what is",
        "define",
        "definition",
        "meaning",
    )
    return any(marker in normalized for marker in markers)


def _is_ranking_request(message: str | None) -> bool:
    normalized = _normalize_message_for_matching(message or "")
    if not normalized:
        return False
    markers = (
        "top",
        "xep hang",
        "rank",
        "cao nhat",
        "thap nhat",
        "best",
        "worst",
        "so sanh",
        "compare",
    )
    return any(marker in normalized for marker in markers)


def _is_metric_evaluation_request(message: str | None) -> bool:
    normalized = _normalize_message_for_matching(message or "")
    if not normalized:
        return False

    metric_markers = (
        "chi so",
        "tieu chi",
        "metric",
        "metrics",
        "kpi",
        "indicator",
        "indicators",
    )
    evaluation_markers = (
        "danh gia",
        "evaluate",
        "evaluation",
        "assess",
        "assessment",
        "hoat dong tot",
        "tot khong",
        "hieu qua",
        "plant health",
    )

    has_metric_marker = any(marker in normalized for marker in metric_markers)
    has_evaluation_marker = any(marker in normalized for marker in evaluation_markers)
    return has_metric_marker and has_evaluation_marker


def _is_top_facility_comparison_request(message: str | None) -> bool:
    normalized = _normalize_message_for_matching(message or "")
    if not normalized:
        return False

    compare_markers = ("so sanh", "compare", "comparison", "versus", " vs ")
    facility_markers = ("facility", "facilities", "tram", "co so", "nha may")
    ranking_markers = (
        "top",
        "top 2",
        "largest",
        "highest",
        "lon nhat",
        "2 facilities",
        "2 facility",
        "hai tram",
        "hai co so",
    )

    has_compare = any(marker in normalized for marker in compare_markers)
    has_facility = any(marker in normalized for marker in facility_markers)
    has_rank = any(marker in normalized for marker in ranking_markers)
    return has_compare and has_facility and has_rank


def _is_vietnamese_request(message: str | None) -> bool:
    raw = str(message or "").strip().lower()
    if not raw:
        return False

    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(character in vietnamese_chars for character in raw):
        return True

    normalized = _normalize_message_for_matching(raw)
    vietnamese_markers = (
        "so sanh",
        "tram",
        "he thong",
        "lon nhat",
        "nho nhat",
        "co so",
        "san luong",
        "du bao",
        "hieu suat",
        "cho toi",
        "thong tin",
    )
    return any(marker in normalized for marker in vietnamese_markers)


def format_source_text(sources: list[SourceMetadata]) -> str:
    return ", ".join(f"{s.layer}:{s.dataset}" for s in sources)


def _build_system_prompt(role_value: str) -> str:
    """Build a rich system prompt with domain context.

    Does NOT hardcode country/region — the chatbot discovers
    station locations dynamically via get_facility_info tool.
    """
    return (
        "You are Solar AI, the intelligent assistant for the PV Lakehouse system "
        "- a solar energy data analytics platform.\n\n"
        "PV LAKEHOUSE SYSTEM ARCHITECTURE:\n"
        "- Data Architecture: Bronze (raw) -> Silver (clean) -> Gold (dimensional)\n"
        "- Storage: Databricks Delta Lake, managed via Unity Catalog (pv.bronze, pv.silver, pv.gold)\n"
        "- Fact/Reference Data: energy (MWh), weather (radiation, temperature, wind, cloud cover), "
        "air quality (AQI), ML power generation models\n"
        "- Solar facility details are stored in the dim_facility table "
        "with GPS coordinates (latitude, longitude) and capacity (MW).\n\n"
        "YOUR RESPONSE GUIDELINES:\n"
        "1. Respond naturally and professionally in English by default. HOWEVER, if the user asks in Vietnamese, YOU MUST REPLY IN VIETNAMESE with proper diacritics. You may use appropriate emojis. Keep it concise and clear (max 8-10 sentences) and format the final answer in GitHub-flavored Markdown (headings/lists/table/code when appropriate).\n"
        "2. ALWAYS use tools to fetch real technical metrics.\n"
        "- NEVER hallucinate or make up data/forecasts under any circumstances.\n"
        "3. When asked about locations, countries, GPS coordinates, or 'where' "
        "-> prioritize calling get_facility_info.\n"
        "4. ONLY call get_station_daily_report when the user requests a report "
        "for ONE SPECIFIC DAY (e.g., 2026-04-09, yesterday). "
        "If they ask for the last X days/trends/multiple days, use get_energy_performance "
        "or get_forecast_72h.\n"
        "5. Carefully analyze the conversational context to maintain the chat flow.\n"
        "6. For location questions, prioritize exact fields returned by tools (facility_name, coordinates, region/state/timezone). "
        "If country/state is missing from the data payload, state that clearly instead of guessing.\n"
        "7. ABSOLUTELY HIDE THESE KEYWORDS: 'dim_facility', 'Bronze', 'Silver', 'Gold', "
        "'pv.bronze', 'pv.silver', 'pv.gold', 'Databricks', 'Delta Lake', 'Unity Catalog'. Do not mention internal architecture.\n"
        "8. Tactfully inform the user if a tool returns no data.\n"
        f"9. The user's recognized role/persona is: {role_value}\n"
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
                "Acknowledged. I am Solar AI, the data assistant for the PV Lakehouse system. "
                "I will automatically search through data tools "
                "and respond intelligently and accurately in English (or Vietnamese if requested)."
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
        "Please provide an intelligent answer based on the successfully retrieved data below. "
        "Summarize important metrics, compare correlations (if any), and provide clear Insights. "
        "Absolutely do not display internal keys or raw JSON from the system in your answer.",
    ]

    if history:
        parts.append("Recent conversation history between you and the user:")
        trimmed = history[-MAX_HISTORY_IN_PROMPT:]
        total = len(trimmed)
        for idx, msg in enumerate(trimmed):
            label = "User" if msg.sender == "user" else "Solar AI Assistant"
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
        f"User Role: {role.value}",
        f"Recognized Intent Topic: {topic.value}",
        f"Original Question: {user_message}",
        f"Reference Data Sources: {source_text}",
        f"Tool Data/Metrics [Consider this as your own knowledge, do not say 'the tool returned']: {metrics_json}",
    ])
    return "\n".join(parts)


def build_fallback_summary(
    topic: ChatTopic,
    metrics: dict[str, Any],
    sources: list[SourceMetadata],
    user_message: str | None = None,
) -> str:
    """Return a deterministic summary when the LLM is unavailable."""

    if topic is ChatTopic.GENERAL:
        if user_message:
            return (
                "Mình chưa có dữ liệu hoặc công cụ phù hợp để trả lời chính xác câu hỏi này. "
                "Bạn có thể hỏi về: tổng quan hệ thống, hiệu suất năng lượng, mô hình ML, "
                "trạng thái pipeline, dự báo 72 giờ, chất lượng dữ liệu, hoặc thông tin trạm."
            )
        return (
            "Hello! I am the Solar AI Chat assistant. You can ask me about: "
            "system overview, energy performance, ML models, "
            "pipeline status, 72-hour forecasts, or data quality."
        )

    if topic is ChatTopic.SYSTEM_OVERVIEW:
        return (
            "System overview: current production is "
            f"{metrics.get('production_output_mwh', 0)} MWh, R-squared is "
            f"{metrics.get('r_squared', 0)}, data quality score is "
            f"{metrics.get('data_quality_score', 0)} and total facilities "
            f"{metrics.get('facility_count', 0)}."
        )

    if topic is ChatTopic.ENERGY_PERFORMANCE and "extreme_metric" not in metrics:
        if _is_performance_ratio_request(user_message):
            definition_requested = _is_definition_request(user_message)
            ranking_requested = _is_ranking_request(user_message)

            if definition_requested and not ranking_requested:
                return (
                    "Performance Ratio (PR) là chỉ số phản ánh hiệu suất vận hành thực tế của hệ PV so với mức kỳ vọng theo điều kiện bức xạ. "
                    "PR càng cao thì hệ thống càng ít tổn thất (nhiệt độ, inverter, dây dẫn, bụi bẩn, suy hao thiết bị). "
                    "Trong hệ thống hiện tại, chưa có cột PR chuẩn trực tiếp cho mọi trạm; "
                    "vì vậy đang dùng capacity_factor_pct như chỉ số proxy để so sánh tương đối giữa các trạm."
                )

            ratio_rows = metrics.get("top_performance_ratio_facilities", [])
            if isinstance(ratio_rows, list) and ratio_rows:
                ratio_fragments: list[str] = []
                for row in ratio_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    ratio_fragments.append(
                        f"{row.get('facility', 'Unknown')} ({row.get('performance_ratio_pct', 0)}%)"
                    )
                if ratio_fragments:
                    return (
                        "Top trạm theo performance ratio (dùng capacity factor trung bình): "
                        + ", ".join(ratio_fragments)
                        + "."
                    )

            return (
                "Hiện chưa có dữ liệu performance ratio/capacity factor đủ để xếp hạng theo yêu cầu này."
            )

        if _is_top_facility_comparison_request(user_message):
            top_rows = metrics.get("top_facilities", [])
            if isinstance(top_rows, list):
                valid_rows = [row for row in top_rows if isinstance(row, dict)]
            else:
                valid_rows = []

            if len(valid_rows) >= 2:
                first = valid_rows[0]
                second = valid_rows[1]

                first_name = str(first.get("facility") or "Facility A")
                second_name = str(second.get("facility") or "Facility B")

                try:
                    first_energy = float(first.get("energy_mwh") or 0.0)
                except (TypeError, ValueError):
                    first_energy = 0.0

                try:
                    second_energy = float(second.get("energy_mwh") or 0.0)
                except (TypeError, ValueError):
                    second_energy = 0.0

                delta = first_energy - second_energy
                ratio_text = "N/A"
                ratio_multiple_text = "N/A"
                if second_energy > 0:
                    ratio_text = f"{(delta / second_energy) * 100:.2f}%"
                    ratio_multiple_text = f"{(first_energy / second_energy):.2f}x"

                combined_top2 = first_energy + second_energy
                first_share_text = "N/A"
                second_share_text = "N/A"
                if combined_top2 > 0:
                    first_share_text = f"{(first_energy / combined_top2) * 100:.2f}%"
                    second_share_text = f"{(second_energy / combined_top2) * 100:.2f}%"

                window_days = metrics.get("window_days")
                if window_days is not None:
                    period_text_vi = f"{window_days} ngày gần nhất"
                    period_text_en = f"last {window_days} days"
                else:
                    period_text_vi = "kỳ dữ liệu hiện có"
                    period_text_en = "current available data window"

                peak_rows = metrics.get("peak_hours", []) if isinstance(metrics, dict) else []
                top_peak_text = "N/A"
                if isinstance(peak_rows, list) and peak_rows:
                    first_peak = peak_rows[0]
                    if isinstance(first_peak, dict):
                        try:
                            peak_energy = float(first_peak.get("energy_mwh") or 0.0)
                        except (TypeError, ValueError):
                            peak_energy = 0.0
                        top_peak_text = f"{first_peak.get('hour')}:00 ({peak_energy:.2f} MWh)"

                try:
                    tomorrow_forecast_value = float(metrics.get("tomorrow_forecast_mwh") or 0.0)
                except (TypeError, ValueError):
                    tomorrow_forecast_value = 0.0

                third_text_vi = ""
                third_text_en = ""
                if len(valid_rows) >= 3:
                    third = valid_rows[2]
                    third_name = str(third.get("facility") or "Facility C")
                    try:
                        third_energy = float(third.get("energy_mwh") or 0.0)
                    except (TypeError, ValueError):
                        third_energy = 0.0
                    gap_vs_third = second_energy - third_energy
                    third_text_vi = (
                        f"\n- Facility đứng thứ 3 là **{third_name}** với **{third_energy:.2f} MWh** "
                        f"(kém {second_name} **{gap_vs_third:.2f} MWh**)."
                    )
                    third_text_en = (
                        f"\n- **3rd place**: {third_name} with {third_energy:.2f} MWh "
                        f"({gap_vs_third:.2f} MWh below {second_name})."
                    )

                if _is_vietnamese_request(user_message):
                    return (
                        "## So sánh Top 2 Facilities (theo tổng sản lượng)\n"
                        f"- Kỳ phân tích: **{period_text_vi}**\n"
                        "- Chỉ số xếp hạng: **Tổng sản lượng điện (MWh)**, không phải capacity lắp đặt.\n"
                        "\n"
                        "| Xếp hạng | Facility | Sản lượng (MWh) | Tỷ trọng trong Top 2 |\n"
                        "|---|---|---:|---:|\n"
                        f"| 1 | {first_name} | {first_energy:.2f} | {first_share_text} |\n"
                        f"| 2 | {second_name} | {second_energy:.2f} | {second_share_text} |\n"
                        "\n"
                        f"- Chênh lệch tuyệt đối: **{delta:.2f} MWh**.\n"
                        f"- {first_name} đang cao hơn {second_name} **{ratio_text}** (tương đương **{ratio_multiple_text}**)."
                        f"{third_text_vi}\n"
                        f"- Peak hour toàn hệ (tham chiếu): **{top_peak_text}**.\n"
                        f"- Dự báo ngày mai toàn hệ: **{tomorrow_forecast_value:.2f} MWh**."
                    )

                return (
                    "## Top 2 Facilities Comparison\n"
                    f"- Analysis window: **{period_text_en}**\n"
                    "- Ranking metric: **Total generated energy (MWh)** (not installed capacity).\n"
                    "\n"
                    "| Rank | Facility | Energy (MWh) | Share within Top 2 |\n"
                    "|---|---|---:|---:|\n"
                    f"| 1 | {first_name} | {first_energy:.2f} | {first_share_text} |\n"
                    f"| 2 | {second_name} | {second_energy:.2f} | {second_share_text} |\n"
                    "\n"
                    f"- **Absolute gap**: {delta:.2f} MWh.\n"
                    f"- **Relative lead**: {ratio_text} vs {second_name} ({ratio_multiple_text}).\n"
                    f"{third_text_en}\n"
                    f"- **System peak hour reference**: {top_peak_text}.\n"
                    f"- **Tomorrow forecast (system-wide)**: {tomorrow_forecast_value:.2f} MWh."
                )

            return "Top-2 facility comparison is unavailable because there is not enough facility performance data."

        if (
            _is_metric_evaluation_request(user_message)
            and not _is_ranking_request(user_message)
        ):
            return (
                "Các chỉ số thường dùng để đánh giá một nhà máy điện mặt trời hoạt động tốt gồm:\n"
                "- Performance Ratio (PR): mức hiệu suất thực tế so với mức kỳ vọng theo bức xạ.\n"
                "- Capacity Factor (CF): tỷ lệ sản lượng thực tế so với công suất định mức theo thời gian.\n"
                "- Specific Yield (kWh/kWp): sản lượng tạo ra trên mỗi kWp lắp đặt.\n"
                "- Availability/Uptime: tỷ lệ thời gian hệ thống sẵn sàng phát điện.\n"
                "- System losses: tổn thất do nhiệt độ, inverter clipping, cáp, bụi bẩn và suy hao thiết bị.\n"
                "- Forecast error (nếu có mô hình dự báo): mức chênh giữa dự báo và sản lượng thực tế."
            )

        top_rows = metrics.get("top_facilities", [])
        peak_rows = metrics.get("peak_hours", [])

        top_text = "No top-facility data is available."
        if isinstance(top_rows, list) and top_rows:
            top_fragments: list[str] = []
            for row in top_rows[:3]:
                if not isinstance(row, dict):
                    continue
                top_fragments.append(
                    f"{row.get('facility', 'Unknown')} ({row.get('energy_mwh', 0)} MWh)"
                )
            if top_fragments:
                top_text = "Top facilities: " + ", ".join(top_fragments) + "."

        peak_text = ""
        if isinstance(peak_rows, list) and peak_rows:
            peak_fragments: list[str] = []
            for row in peak_rows[:3]:
                if not isinstance(row, dict):
                    continue
                peak_fragments.append(f"{row.get('hour')}:00 ({row.get('energy_mwh', 0)} MWh)")
            if peak_fragments:
                peak_text = " Peak hours: " + ", ".join(peak_fragments) + "."

        return (
            f"{top_text}{peak_text} "
            f"Tomorrow's forecast is approximately {metrics.get('tomorrow_forecast_mwh', 0)} MWh."
        )

    if topic is ChatTopic.ML_MODEL:
        model_name = str(metrics.get("model_name") or "Unknown")
        model_version = str(metrics.get("model_version") or "N/A")
        parameters = metrics.get("parameters", {}) if isinstance(metrics, dict) else {}
        comparison = metrics.get("comparison", {}) if isinstance(metrics, dict) else {}

        approach = str(parameters.get("approach") or "N/A")
        current_r2 = comparison.get("current_r_squared")
        previous_r2 = comparison.get("previous_r_squared")
        delta_r2 = comparison.get("delta_r_squared")
        skill_score = comparison.get("skill_score")
        nrmse_pct = comparison.get("nrmse_pct")
        evaluated_on = str(comparison.get("evaluated_on") or "N/A")

        current_text = f"{current_r2}" if current_r2 is not None else "N/A"
        previous_text = f"{previous_r2}" if previous_r2 is not None else "N/A"
        delta_text = f"{delta_r2}" if delta_r2 is not None else "N/A"
        skill_text = f"{skill_score}" if skill_score is not None else "N/A"
        nrmse_text = f"{nrmse_pct}" if nrmse_pct is not None else "N/A"

        summary = (
            f"Current model: {model_name} (version {model_version}, approach {approach}). "
            f"Eval date: {evaluated_on}. "
            f"R-squared: {current_text}, previous: {previous_text}, delta: {delta_text}. "
            f"Skill score: {skill_text}, NRMSE: {nrmse_text}%."
        )

        if bool(metrics.get("is_fallback_model", False)):
            latest_non_fallback_r2 = comparison.get("latest_non_fallback_r_squared")
            latest_non_fallback_model = str(comparison.get("latest_non_fallback_model") or "")
            if latest_non_fallback_r2 is not None:
                fallback_label = latest_non_fallback_model or "latest non-fallback model"
                summary += (
                    f" Latest run is fallback. {fallback_label} has R-squared {latest_non_fallback_r2}."
                )

        return summary

    if topic is ChatTopic.PIPELINE_STATUS:
        return (
            "The pipeline is being tracked by stage with ETAs and data quality alerts. "
            f"Current alerts count: {len(metrics.get('alerts', []))}."
        )

    if topic is ChatTopic.FORECAST_72H:
        return (
            "The 72-hour forecast is ready for each day along with confidence intervals. "
            f"Forecast milestones: {len(metrics.get('daily_forecast', []))}."
        )

    extreme_metric = metrics.get("extreme_metric")
    if extreme_metric is not None:
        return format_extreme_fallback(metrics, extreme_metric)

    if topic is ChatTopic.DATA_QUALITY_ISSUES:
        facility_scores = metrics.get("facility_quality_scores", [])
        if isinstance(facility_scores, list) and facility_scores:
            lines: list[str] = ["Data quality score by facility (average completeness):"]
            for row in facility_scores[:20]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"  - {row.get('facility', 'Unknown')}: {row.get('quality_score', 0)}%"
                )

            low_score = metrics.get("low_score_facilities", [])
            if isinstance(low_score, list) and low_score:
                low_fragments: list[str] = []
                for row in low_score[:5]:
                    if not isinstance(row, dict):
                        continue
                    low_fragments.append(
                        f"{row.get('facility', 'Unknown')} ({row.get('quality_score', 0)}%)"
                    )
                if low_fragments:
                    lines.append("Facilities below 95%: " + ", ".join(low_fragments) + ".")
            else:
                lines.append("All facilities have quality score >= 95%.")

            return "\n".join(lines)

        low_score = metrics.get("low_score_facilities", [])
        if isinstance(low_score, list) and low_score:
            return (
                "Low-score facilities detected: "
                + ", ".join(
                    f"{row.get('facility', 'Unknown')} ({row.get('quality_score', 0)}%)"
                    for row in low_score[:5]
                    if isinstance(row, dict)
                )
                + "."
            )

        summary_text = str(metrics.get("summary") or "").strip()
        if summary_text:
            return summary_text

        return "No data-quality issue was detected from the latest dataset."

    if topic is ChatTopic.FACILITY_INFO:
        facilities = metrics.get("facilities", [])
        count = metrics.get("facility_count", len(facilities))
        if not facilities:
            return "No matching station information found."
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
                f"The station with the largest capacity currently is {top_station_name} ({round(top_station_capacity, 2)} MW)."
            )
        lines.append(f"Found {count} solar energy stations:")
        for f in facilities[:8]:
            name = f.get("facility_name", "Unknown")
            lat = f.get("location_lat", 0)
            lng = f.get("location_lng", 0)
            cap = f.get("total_capacity_mw", 0)
            tz_name = f.get("timezone_name", "N/A")
            tz_offset = f.get("timezone_utc_offset", "N/A")
            lines.append(
                f"  - {name}: coordinates ({lat}, {lng}), "
                f"Capacity {cap} MW, Timezone {tz_name} ({tz_offset})"
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
            f"The {query_type} AQI query "
            f"returned {metrics.get(f'{query_type}_aqi_value', 0)} at station {station} "
            f"{timeframe_text}. "
            f"AQI Category: {metrics.get(f'{query_type}_aqi_category', 'Unknown')}."
        )

    if extreme_metric == "energy":
        return (
            f"The {query_type} energy production is "
            f"{metrics.get(f'{query_type}_energy_mwh', 0)} MWh "
            f"at station {station} {timeframe_text}."
        )

    return (
        f"The {query_type} weather metric {metrics.get('weather_metric_label', 'weather')} is "
        f"{metrics.get(f'{query_type}_weather_value', 0)} {metrics.get('weather_unit', '')} "
        f"at station {station} {timeframe_text}."
    )


def describe_timeframe(metrics: dict[str, Any]) -> str:
    timeframe = metrics.get("timeframe", "day")
    period_label = str(metrics.get("period_label", metrics.get("query_date", "N/A")))
    if timeframe == "hour":
        if metrics.get("specific_hour") is not None:
            return f"at {period_label}"
        return f"for 1 hour at {period_label}"
    if timeframe == "24h":
        return f"for 24 hours anchored at {period_label}"
    if timeframe == "week":
        return f"for the week {period_label}"
    if timeframe == "month":
        return f"for the month {period_label}"
    if timeframe == "year":
        return f"for the year {period_label}"
    return f"on {period_label}"
