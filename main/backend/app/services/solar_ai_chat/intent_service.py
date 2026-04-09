from dataclasses import dataclass
from unicodedata import normalize

from app.schemas.solar_ai_chat import ChatTopic


def normalize_vietnamese_text(value: str) -> str:
    """Normalize Vietnamese text: lowercase, strip diacritics, ASCII-only."""
    lowered = value.strip().lower()
    without_marks = normalize("NFD", lowered)
    return "".join(character for character in without_marks if ord(character) < 128)


@dataclass(frozen=True)
class IntentDetectionResult:
    topic: ChatTopic
    confidence: float


class VietnameseIntentService:
    """Rule-based Vietnamese intent detection for supported chatbot topics."""

    _TOPIC_KEYWORDS: dict[ChatTopic, tuple[str, ...]] = {
        ChatTopic.DATA_QUALITY_ISSUES: (
            "chat luong du lieu",
            "chat luong khong khi",
            "du lieu loi",
            "chi so aqi",
            "aqi",
            "aqi thap nhat",
            "tram nao",
            "low score",
            "co so diem thap",
            "nguyen nhan",
            "canh bao du lieu",
        ),
        ChatTopic.FORECAST_72H: (
            "72h",
            "72 gio",
            "ba ngay",
            "3 ngay",
            "khoang tin cay",
            "confidence interval",
        ),
        ChatTopic.PIPELINE_STATUS: (
            "pipeline",
            "tien do",
            "eta",
            "trang thai",
            "alerts",
            "canh bao",
        ),
        ChatTopic.ML_MODEL: (
            "mo hinh",
            "model",
            "gbt",
            "v4.2",
            "v4.1",
            "tham so",
            "r-squared",
        ),
        ChatTopic.ENERGY_PERFORMANCE: (
            "hieu suat",
            "top nha may",
            "top facility",
            "gio cao diem",
            "peak hour",
            "du bao ngay mai",
            "tomorrow forecast",
            "energy",
            "nang luong",
            "san luong cao nhat",
            "san luong thap nhat",
            "energy cao nhat",
            "energy thap nhat",
            "weather",
            "thoi tiet",
            "nhiet do",
            "wind",
            "toc do gio",
        ),
        ChatTopic.SYSTEM_OVERVIEW: (
            "tong quan",
            "he thong",
            "san luong",
            "r2",
            "r-squared",
            "so co so",
            "facility count",
        ),
        ChatTopic.FACILITY_INFO: (
            "vi tri",
            "o dau",
            "toa do",
            "quoc gia",
            "nuoc nao",
            "bang nao",
            "tinh nao",
            "gps",
            "location",
            "latitude",
            "longitude",
            "cong suat",
            "capacity",
            "thong tin tram",
            "thong tin co so",
            "thong tin nha may",
            "dia ly",
            "dia chi",
            "nam o",
            "dat o",
            "tram nao o",
            "co so o",
        ),
    }

    def detect_intent(self, message: str) -> IntentDetectionResult:
        normalized_message = normalize_vietnamese_text(message)
        if not normalized_message:
            raise ValueError("The question cannot be empty.")

        matched_topic: ChatTopic | None = None
        matched_score = 0

        for topic, keywords in self._TOPIC_KEYWORDS.items():
            topic_score = sum(1 for keyword in keywords if keyword in normalized_message)
            if topic_score > matched_score:
                matched_topic = topic
                matched_score = topic_score

        if matched_topic is None:
            return IntentDetectionResult(topic=ChatTopic.GENERAL, confidence=0.3)

        confidence = min(0.99, 0.5 + (matched_score * 0.15))
        return IntentDetectionResult(topic=matched_topic, confidence=round(confidence, 2))
