import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any
from unicodedata import normalize

from app.repositories.solar_chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat import (
    ChatRole,
    ChatTopic,
    SolarChatRequest,
    SolarChatResponse,
    SourceMetadata,
)
from app.services.gemini_client import GeminiModelRouter, ModelUnavailableError
from app.services.solar_chat_intent_service import VietnameseIntentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtremeMetricQuery:
    metric_name: str
    query_type: str
    timeframe: str
    specific_hour: int | None = None


class SolarAIChatService:
    """Business service that orchestrates intent routing, RBAC, data, and LLM response."""

    _WEATHER_METRIC_CATALOG: tuple[dict[str, Any], ...] = (
        {
            "key": "temperature_2m",
            "label": "temperature",
            "unit": "C",
            "keywords": ("nhiet do", "temperature", "nong", "lanh"),
        },
        {
            "key": "wind_speed_10m",
            "label": "wind speed",
            "unit": "m/s",
            "keywords": ("toc do gio", "wind speed", "gio", "wind"),
        },
        {
            "key": "wind_gusts_10m",
            "label": "wind gust",
            "unit": "m/s",
            "keywords": ("gio giat", "wind gust"),
        },
        {
            "key": "shortwave_radiation",
            "label": "shortwave radiation",
            "unit": "W/m2",
            "keywords": (
                "buc xa mat troi",
                "solar radiation",
                "irradiance",
                "buc xa",
                "radiation",
            ),
        },
        {
            "key": "cloud_cover",
            "label": "cloud cover",
            "unit": "%",
            "keywords": ("do phu may", "cloud cover", "may", "cloud"),
        },
    )

    _TIMEFRAME_NOISE_PATTERNS: tuple[str, ...] = (
        r"\b24\s*h\b",
        r"\b24\s*gio\b",
        r"\b1\s*h\b",
        r"\b1\s*gio\b",
        r"\btheo\s+gio\b",
        r"\bmoi\s+gio\b",
    )

    _ROLE_PERMISSIONS: dict[ChatRole, set[ChatTopic]] = {
        ChatRole.DATA_ENGINEER: {
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.PIPELINE_STATUS,
            ChatTopic.FORECAST_72H,
            ChatTopic.DATA_QUALITY_ISSUES,
        },
        ChatRole.ML_ENGINEER: {
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.ML_MODEL,
            ChatTopic.FORECAST_72H,
        },
        ChatRole.DATA_ANALYST: {
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.ML_MODEL,
            ChatTopic.FORECAST_72H,
            ChatTopic.DATA_QUALITY_ISSUES,
        },
        ChatRole.VIEWER: {
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.FORECAST_72H,
        },
        ChatRole.ADMIN: {
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.ML_MODEL,
            ChatTopic.PIPELINE_STATUS,
            ChatTopic.FORECAST_72H,
            ChatTopic.DATA_QUALITY_ISSUES,
        },
    }

    def __init__(
        self,
        repository: SolarChatRepository,
        intent_service: VietnameseIntentService,
        model_router: GeminiModelRouter | None,
    ) -> None:
        self._repository = repository
        self._intent_service = intent_service
        self._model_router = model_router

    def handle_query(self, request: SolarChatRequest) -> SolarChatResponse:
        started = time.perf_counter()

        response_topic: ChatTopic
        intent_confidence: float

        extreme_query = self._extract_extreme_metric_query(request.message)
        if extreme_query is not None:
            response_topic = self._topic_for_extreme_metric(extreme_query.metric_name)
            self._validate_role(topic=response_topic, role=request.role)

            query_date = self._extract_query_date(request.message)

            metrics, source_rows = self._fetch_extreme_metrics(
                extreme_query=extreme_query,
                query_date=query_date,
                message=request.message,
            )
            intent_confidence = 0.92
        else:
            detection = self._intent_service.detect_intent(request.message)
            response_topic = detection.topic
            intent_confidence = detection.confidence

            self._validate_role(topic=response_topic, role=request.role)
            metrics, source_rows = self._repository.fetch_topic_metrics(response_topic)
        sources = [SourceMetadata(**source_row) for source_row in source_rows]

        warning_message: str | None = None
        fallback_used = False
        model_used = "deterministic-summary"

        prompt = self._build_prompt(
            user_message=request.message,
            role=request.role,
            topic=response_topic,
            metrics=metrics,
            sources=sources,
        )

        if self._model_router is not None:
            try:
                model_result = self._model_router.generate(prompt)
                answer = model_result.text
                fallback_used = model_result.fallback_used
                model_used = model_result.model_used
            except ModelUnavailableError:
                answer = self._build_fallback_summary(
                    topic=response_topic,
                    metrics=metrics,
                    sources=sources,
                )
                warning_message = (
                    "The AI model is temporarily unavailable. Returned a data-backed summary instead."
                )
                fallback_used = True
        else:
            answer = self._build_fallback_summary(
                topic=response_topic,
                metrics=metrics,
                sources=sources,
            )
            warning_message = "Gemini API key is not configured. Returned a data-backed summary."

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "solar_chat_request_completed",
            extra={
                "topic": response_topic.value,
                "role": request.role.value,
                "model_used": model_used,
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
            },
        )

        return SolarChatResponse(
            answer=answer,
            topic=response_topic,
            role=request.role,
            sources=sources,
            key_metrics=metrics,
            model_used=model_used,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            intent_confidence=intent_confidence,
            warning_message=warning_message,
        )

    def _validate_role(self, topic: ChatTopic, role: ChatRole) -> None:
        allowed_topics = self._ROLE_PERMISSIONS.get(role, set())
        if topic not in allowed_topics:
            raise PermissionError(
                f"Role '{role.value}' is not allowed to access topic '{topic.value}'."
            )

    @staticmethod
    def _build_prompt(
        user_message: str,
        role: ChatRole,
        topic: ChatTopic,
        metrics: dict[str, Any],
        sources: list[SourceMetadata],
    ) -> str:
        source_text = ", ".join(f"{source.layer}:{source.dataset}" for source in sources)
        metrics_json = json.dumps(metrics, ensure_ascii=False)

        return (
            "Bạn là trợ lý Solar AI Chat cho người dùng không kỹ thuật. "
            "Hãy trả lời bằng tiếng Việt ngắn gọn, rõ ràng, tối đa 5 câu. "
            "Cần nêu chỉ số chính, ý nghĩa ngữ cảnh và nhắc nguồn dữ liệu Silver/Gold.\n"
            f"Vai trò người dùng: {role.value}\n"
            f"Chủ đề: {topic.value}\n"
            f"Câu hỏi: {user_message}\n"
            f"Nguồn dữ liệu: {source_text}\n"
            f"Chỉ số đã truy xuất: {metrics_json}"
        )

    @staticmethod
    def _build_fallback_summary(
        topic: ChatTopic,
        metrics: dict[str, Any],
        sources: list[SourceMetadata],
    ) -> str:
        source_text = ", ".join(f"{source.layer}:{source.dataset}" for source in sources)

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

        if metrics.get("extreme_metric") == "aqi":
            query_type = metrics.get("query_type", "")
            station_key = f"{query_type}_station"
            value_key = f"{query_type}_aqi_value"
            category_key = f"{query_type}_aqi_category"
            timeframe_text = SolarAIChatService._describe_timeframe(metrics)
            return (
                f"AQI {query_type} theo truy vấn là "
                f"{metrics.get(value_key, 0)} tại trạm {metrics.get(station_key, 'Unknown')} "
                f"{timeframe_text}. "
                f"Phân loại AQI: {metrics.get(category_key, 'Unknown')}. "
                f"Nguồn: {source_text}."
            )

        if metrics.get("extreme_metric") == "energy":
            query_type = metrics.get("query_type", "")
            station_key = f"{query_type}_station"
            value_key = f"{query_type}_energy_mwh"
            timeframe_text = SolarAIChatService._describe_timeframe(metrics)
            return (
                f"Sản lượng năng lượng {query_type} là {metrics.get(value_key, 0)} MWh "
                f"tại trạm {metrics.get(station_key, 'Unknown')} {timeframe_text}. "
                f"Nguồn: {source_text}."
            )

        if metrics.get("extreme_metric") == "weather":
            query_type = metrics.get("query_type", "")
            station_key = f"{query_type}_station"
            value_key = f"{query_type}_weather_value"
            timeframe_text = SolarAIChatService._describe_timeframe(metrics)
            return (
                f"Chỉ số thời tiết {metrics.get('weather_metric_label', 'weather')} {query_type} là "
                f"{metrics.get(value_key, 0)} {metrics.get('weather_unit', '')} "
                f"tại trạm {metrics.get(station_key, 'Unknown')} {timeframe_text}. "
                f"Nguồn: {source_text}."
            )

        return (
            "Các cơ sở có điểm chất lượng thấp đã được xác định kèm nguyên nhân khả dĩ từ cờ chất lượng. "
            f"Nguồn: {source_text}."
        )

    def _fetch_extreme_metrics(
        self,
        extreme_query: ExtremeMetricQuery,
        query_date: date | None,
        message: str,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if extreme_query.metric_name == "aqi":
            return self._repository.fetch_extreme_aqi(
                query_type=extreme_query.query_type,
                timeframe=extreme_query.timeframe,
                anchor_date=query_date,
                specific_hour=extreme_query.specific_hour,
            )

        if extreme_query.metric_name == "energy":
            return self._repository.fetch_extreme_energy(
                query_type=extreme_query.query_type,
                timeframe=extreme_query.timeframe,
                anchor_date=query_date,
                specific_hour=extreme_query.specific_hour,
            )

        weather_metric = self._resolve_weather_metric(message)
        return self._repository.fetch_extreme_weather(
            query_type=extreme_query.query_type,
            timeframe=extreme_query.timeframe,
            anchor_date=query_date,
            specific_hour=extreme_query.specific_hour,
            weather_metric=weather_metric["key"],
            weather_metric_label=weather_metric["label"],
            weather_unit=weather_metric["unit"],
        )

    @staticmethod
    def _topic_for_extreme_metric(metric_name: str) -> ChatTopic:
        if metric_name == "aqi":
            return ChatTopic.DATA_QUALITY_ISSUES
        return ChatTopic.ENERGY_PERFORMANCE

    @staticmethod
    def _extract_extreme_metric_query(message: str) -> ExtremeMetricQuery | None:
        normalized_message = SolarAIChatService._normalize_text(message)

        query_type: str | None = None
        if any(marker in normalized_message for marker in ("thap nhat", "nho nhat", "min")):
            query_type = "lowest"
        if any(marker in normalized_message for marker in ("cao nhat", "lon nhat", "max")):
            query_type = "highest"
        if query_type is None:
            return None

        specific_hour = SolarAIChatService._extract_specific_hour(normalized_message)
        timeframe = SolarAIChatService._extract_timeframe(
            normalized_message,
            specific_hour=specific_hour,
        )

        if "aqi" in normalized_message:
            return ExtremeMetricQuery(
                metric_name="aqi",
                query_type=query_type,
                timeframe=timeframe,
                specific_hour=specific_hour,
            )

        energy_markers = ("energy", "nang luong", "san luong", "dien nang", "mwh")
        if any(marker in normalized_message for marker in energy_markers):
            return ExtremeMetricQuery(
                metric_name="energy",
                query_type=query_type,
                timeframe=timeframe,
                specific_hour=specific_hour,
            )

        weather_markers = (
            "weather",
            "thoi tiet",
            "nhiet do",
            "wind",
            "gio",
            "buc xa",
            "cloud",
            "may",
            "mua",
        )
        if any(marker in normalized_message for marker in weather_markers):
            return ExtremeMetricQuery(
                metric_name="weather",
                query_type=query_type,
                timeframe=timeframe,
                specific_hour=specific_hour,
            )

        return None

    def _resolve_weather_metric(self, message: str) -> dict[str, Any]:
        normalized_message = self._normalize_text(message)
        sanitized_message = self._strip_timeframe_noise(normalized_message)

        selected_metric = self._WEATHER_METRIC_CATALOG[0]
        selected_score = 0

        for metric in self._WEATHER_METRIC_CATALOG:
            score = self._score_metric_keywords(sanitized_message, metric["keywords"])
            if score > selected_score:
                selected_metric = metric
                selected_score = score

        return selected_metric

    @classmethod
    def _strip_timeframe_noise(cls, normalized_message: str) -> str:
        sanitized = normalized_message
        for pattern in cls._TIMEFRAME_NOISE_PATTERNS:
            sanitized = re.sub(pattern, " ", sanitized)
        return re.sub(r"\s+", " ", sanitized).strip()

    @staticmethod
    def _score_metric_keywords(message: str, keywords: tuple[str, ...]) -> int:
        score = 0
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword)}\b"
            matches = re.findall(pattern, message)
            if not matches:
                continue

            token_weight = max(1, keyword.count(" ") + 1)
            score += len(matches) * token_weight
        return score

    @staticmethod
    def _extract_query_date(message: str) -> date | None:
        normalized_message = SolarAIChatService._normalize_text(message)

        ddmmyyyy_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", normalized_message)
        if ddmmyyyy_match:
            day, month, year = (int(value) for value in ddmmyyyy_match.groups())
            try:
                return date(year=year, month=month, day=day)
            except ValueError:
                return None

        yyyymmdd_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", normalized_message)
        if yyyymmdd_match:
            year, month, day = (int(value) for value in yyyymmdd_match.groups())
            try:
                return date(year=year, month=month, day=day)
            except ValueError:
                return None

        mmyyyy_match = re.search(r"(?<!\d)(\d{1,2})[/-](\d{4})(?!\d)", normalized_message)
        if mmyyyy_match:
            month, year = (int(value) for value in mmyyyy_match.groups())
            try:
                return date(year=year, month=month, day=1)
            except ValueError:
                return None

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", normalized_message)
        if year_match:
            year = int(year_match.group(1))
            return date(year=year, month=1, day=1)

        return None

    @staticmethod
    def _extract_timeframe(
        normalized_message: str,
        specific_hour: int | None = None,
    ) -> str:
        if specific_hour is not None:
            return "hour"
        if any(marker in normalized_message for marker in ("24 gio", "24h", "24 h")):
            return "24h"
        if any(marker in normalized_message for marker in ("1 gio", "1h", "1 h", "theo gio", "moi gio")):
            return "hour"
        if "tuan" in normalized_message:
            return "week"
        if "thang" in normalized_message:
            return "month"
        if "nam" in normalized_message:
            return "year"
        return "day"

    @staticmethod
    def _extract_specific_hour(normalized_message: str) -> int | None:
        patterns = (
            re.compile(
                r"\b(?:vao\s+luc|luc|vao)\s*(?P<hour>\d{1,2})(?:\s*[:h]\s*(?P<minute>\d{1,2}))?\s*(?:gio)?\s*(?P<period>sang|chieu|toi|dem|am|pm)?\b"
            ),
            re.compile(r"\b(?P<hour>\d{1,2})\s*gio\s*(?P<period>sang|chieu|toi|dem)\b"),
            re.compile(r"\b(?P<hour>\d{1,2})\s*(?::|h)\s*(?P<minute>\d{1,2})\s*(?P<period>am|pm)?\b"),
        )

        for pattern in patterns:
            match = pattern.search(normalized_message)
            if not match:
                continue

            hour_text = match.groupdict().get("hour")
            minute_text = match.groupdict().get("minute")
            day_period = match.groupdict().get("period")
            if hour_text is None:
                continue

            try:
                hour = int(hour_text)
            except ValueError:
                continue

            if minute_text is not None:
                try:
                    minute = int(minute_text)
                except ValueError:
                    continue
                if minute >= 60:
                    continue

            if day_period in ("chieu", "toi", "dem", "pm") and 1 <= hour <= 11:
                hour += 12
            elif day_period in ("sang", "am") and hour == 12:
                hour = 0

            if hour == 24:
                hour = 0

            if 0 <= hour <= 23:
                return hour

        return None

    @staticmethod
    def _describe_timeframe(metrics: dict[str, Any]) -> str:
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

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = value.strip().lower()
        without_marks = normalize("NFD", lowered)
        return "".join(character for character in without_marks if ord(character) < 128)
