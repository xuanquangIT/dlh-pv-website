"""NLP parsing utilities for the Solar AI Chat service.

Extracts intent signals from Vietnamese user messages:
- Extreme metric query detection (highest/lowest AQI/energy/weather)
- Time frame extraction (hour, day, week, month, year)
- Specific hour extraction with AM/PM handling
- Date extraction with multiple format support
- Weather metric resolution by keyword scoring
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.schemas.solar_ai_chat.enums import ChatTopic
from app.services.solar_ai_chat.intent_service import normalize_vietnamese_text


@dataclass(frozen=True)
class ExtremeMetricQuery:
    metric_name: str
    query_type: str
    timeframe: str
    specific_hour: int | None = None


# Confidence constants for intent detection scoring
INTENT_CONFIDENCE_BASE = 0.5
INTENT_CONFIDENCE_PER_MATCH = 0.15
INTENT_CONFIDENCE_MAX = 0.99

# Weather metric catalog: key, display label, unit, and matching keywords
WEATHER_METRIC_CATALOG: tuple[dict[str, Any], ...] = (
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
        "keywords": ("buc xa mat troi", "solar radiation", "irradiance", "buc xa", "radiation"),
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

_HOUR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:vao\s+luc|luc|vao)\s*(?P<hour>\d{1,2})(?:\s*[:h]\s*(?P<minute>\d{1,2}))?\s*(?:gio)?\s*(?P<period>sang|chieu|toi|dem|am|pm)?\b"
    ),
    re.compile(r"\b(?P<hour>\d{1,2})\s*gio\s*(?P<period>sang|chieu|toi|dem)\b"),
    re.compile(r"\b(?P<hour>\d{1,2})\s*(?::|h)\s*(?P<minute>\d{1,2})\s*(?P<period>am|pm)?\b"),
)

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"), ("day", "month", "year")),
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("year", "month", "day")),
    (re.compile(r"(?<!\d)(\d{1,2})[/-](\d{4})(?!\d)"), ("month", "year")),
    (re.compile(r"\b(19\d{2}|20\d{2})\b"), ("year",)),
)


def topic_for_extreme_metric(metric_name: str) -> ChatTopic:
    if metric_name == "aqi":
        return ChatTopic.DATA_QUALITY_ISSUES
    return ChatTopic.ENERGY_PERFORMANCE


# ---------------------------------------------------------------------------
# Facility ID ↔ name mapping (Bug #5)
# Maps the short facility codes used in the data platform to human-readable
# station names that the LLM and tool executor understand.
# ---------------------------------------------------------------------------
FACILITY_ID_MAP: dict[str, str] = {
    "WRSF1": "White Rock Solar Farm",
    "AVLSF": "Avonlie Solar Farm",
    "BOMENSF": "Bomen Solar Farm",
    "YATSF1": "Yatpool Solar Farm",
    "LIMOSF2": "Limondale Solar Farm",
    "FINLEYSF": "Finley Solar Farm",
    "EMERASF": "Emerald Solar Farm",
    "DARLSF": "Darlington Point Solar Farm",
    # Common short-forms / aliases
    "WRSF": "White Rock Solar Farm",
    "EMERA": "Emerald Solar Farm",
    "DARL": "Darlington Point Solar Farm",
    "AVON": "Avonlie Solar Farm",
    "BOMEN": "Bomen Solar Farm",
    "YATPOOL": "Yatpool Solar Farm",
    "LIMO": "Limondale Solar Farm",
    "FINLEY": "Finley Solar Farm",
}

# Lower-cased facility codes for case-insensitive lookup
_FACILITY_ID_MAP_LOWER: dict[str, str] = {k.lower(): v for k, v in FACILITY_ID_MAP.items()}


def resolve_facility_code(token: str) -> str | None:
    """Return the full facility name for a facility code token, or None if unknown."""
    return _FACILITY_ID_MAP_LOWER.get(token.strip().lower())


def expand_facility_codes_in_message(message: str) -> str:
    """Replace any facility ID codes in *message* with their full names.

    This ensures the LLM and downstream tools receive understandable station
    names even when the user types abbreviated codes such as WRSF1 or EMERASF.
    """
    words = re.split(r"(\s+)", message)
    expanded: list[str] = []
    for word in words:
        clean = word.strip().rstrip(".,;:?!\"'")
        suffix = word[len(clean):]
        full_name = _FACILITY_ID_MAP_LOWER.get(clean.lower())
        if full_name:
            expanded.append(full_name + suffix)
        else:
            expanded.append(word)
    return "".join(expanded)


def make_extreme_query(
    metric_name: str, query_type: str, timeframe: str, specific_hour: int | None,
) -> ExtremeMetricQuery:
    return ExtremeMetricQuery(
        metric_name=metric_name,
        query_type=query_type,
        timeframe=timeframe,
        specific_hour=specific_hour,
    )


def extract_extreme_metric_query(message: str) -> ExtremeMetricQuery | None:
    normalized = normalize_vietnamese_text(message)

    # Ranking requests (top-N/listing) should follow topic-intent path,
    # not the single-station extreme metric path.
    if _is_ranking_request(normalized):
        return None

    query_type: str | None = None
    if any(m in normalized for m in ("thap nhat", "nho nhat", "min")):
        query_type = "lowest"
    elif any(m in normalized for m in ("cao nhat", "lon nhat", "max")):
        query_type = "highest"
    if query_type is None:
        return None

    specific_hour = extract_specific_hour(normalized)
    timeframe = extract_timeframe(normalized, specific_hour=specific_hour)

    if "aqi" in normalized:
        return make_extreme_query("aqi", query_type, timeframe, specific_hour)

    if any(m in normalized for m in ("energy", "nang luong", "san luong", "dien nang", "mwh")):
        return make_extreme_query("energy", query_type, timeframe, specific_hour)

    if any(m in normalized for m in ("weather", "thoi tiet", "nhiet do", "wind", "gio", "buc xa", "cloud", "may", "mua")):
        return make_extreme_query("weather", query_type, timeframe, specific_hour)

    return None


def _is_ranking_request(normalized_message: str) -> bool:
    if re.search(r"\btop\s*\d+\b", normalized_message):
        return True

    ranking_markers = (
        "top ",
        "xep hang",
        "danh sach",
        "cac tram",
        "nhung tram",
    )
    return any(marker in normalized_message for marker in ranking_markers)


def extract_timeframe(normalized_message: str, specific_hour: int | None = None) -> str:
    if specific_hour is not None:
        return "hour"
    # Bug #6: detect "all time" / "historical record" requests
    if any(m in normalized_message for m in (
        "lich su", "all time", "alltime", "moi thoi gian",
        "tat ca thoi gian", "lich su toan bo", "record lich su",
        "toan bo lich su", "ever", "all-time", "historical",
    )):
        return "all_time"
    if any(m in normalized_message for m in ("24 gio", "24h", "24 h")):
        return "24h"
    if any(m in normalized_message for m in ("1 gio", "1h", "1 h", "theo gio", "moi gio")):
        return "hour"
    if "tuan" in normalized_message:
        return "week"
    if "thang" in normalized_message:
        return "month"
    if (
        any(m in normalized_message for m in ("theo nam", "trong nam", "ca nam", "nam nay", "nam ngoai"))
        or re.search(r"\bnam\s+\d{4}\b", normalized_message)
    ):
        return "year"
    return "day"

def parse_timeframe_days(message: str) -> int | None:
    """Extract a timeframe in days from common user query phrases."""
    norm = normalize_vietnamese_text(message)
    # Most specific to least specific
    if any(m in norm for m in ("hom nay", "today", "ngay nay")): return 0
    if any(m in norm for m in ("hom qua", "yesterday", "1 ngay", "mot ngay", "24h", "24 gio")): return 1
    if any(m in norm for m in ("tuan", "week", "7 ngay", "bay ngay")): return 7
    if any(m in norm for m in ("thang", "month", "30 ngay", "ba muoi ngay")): return 30
    if any(m in norm for m in ("nam", "year", "365 ngay")): return 365
    return None


def extract_specific_hour(normalized_message: str) -> int | None:
    for pattern in _HOUR_PATTERNS:
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


def extract_query_date(message: str, base_date: date | None = None) -> date | None:
    normalized = normalize_vietnamese_text(message)
    # Relative-day keywords (Vietnamese + English). Ordered longest-first.
    today = base_date if base_date is not None else date.today()
    if any(kw in normalized for kw in ("hom nay", "ngay nay", "today")):
        return today
    if any(kw in normalized for kw in ("hom qua", "yesterday")):
        from datetime import timedelta
        return today - timedelta(days=1)
    if any(kw in normalized for kw in ("ngay mai", "hom mai", "tomorrow")):
        from datetime import timedelta
        return today + timedelta(days=1)
    for pattern, field_names in _DATE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        parts = {name: int(value) for name, value in zip(field_names, match.groups())}
        try:
            return date(
                year=parts["year"],
                month=parts.get("month", 1),
                day=parts.get("day", 1),
            )
        except ValueError:
            continue
    return None


def strip_timeframe_noise(normalized_message: str) -> str:
    sanitized = normalized_message
    for pattern in _TIMEFRAME_NOISE_PATTERNS:
        sanitized = re.sub(pattern, " ", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


def score_metric_keywords(message: str, keywords: tuple[str, ...]) -> int:
    score = 0
    for keyword in keywords:
        matches = re.findall(rf"\b{re.escape(keyword)}\b", message)
        if matches:
            score += len(matches) * max(1, keyword.count(" ") + 1)
    return score


def resolve_weather_metric(message: str) -> dict[str, Any]:
    normalized = normalize_vietnamese_text(message)
    sanitized = strip_timeframe_noise(normalized)
    selected_metric = WEATHER_METRIC_CATALOG[0]
    selected_score = 0
    for metric in WEATHER_METRIC_CATALOG:
        s = score_metric_keywords(sanitized, metric["keywords"])
        if s > selected_score:
            selected_metric = metric
            selected_score = s
    return selected_metric
