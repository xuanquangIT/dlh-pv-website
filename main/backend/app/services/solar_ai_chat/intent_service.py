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
            "performance",
            "best performance",
            "tot nhat",
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
            "mui gio",
            "timezone",
            "time zone",
            "utc",
            "gio dia phuong",
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

    _TOPIC_CANONICAL_PHRASES: dict[ChatTopic, list[str]] = {
        ChatTopic.DATA_QUALITY_ISSUES: [
            "Chất lượng dữ liệu hôm nay có tốt không",
            "Có trạm nào bị cảnh báo lỗi dữ liệu hay AQI thấp không",
            "Nguyên nhân trạm bị rớt dữ liệu là gì",
        ],
        ChatTopic.FORECAST_72H: [
            "Dự báo sản lượng năng lượng trong 72 giờ tới",
            "Năng lượng khoảng tin cậy 3 ngày tiếp theo",
            "Xu hướng thời tiết 3 ngày tới ra sao",
        ],
        ChatTopic.PIPELINE_STATUS: [
            "Tiến độ chạy ETL pipeline dạo này tốt không",
            "Pipeline trạng thái báo lỗi hay cảnh báo gì không",
        ],
        ChatTopic.ML_MODEL: [
            "Thông số mô hình GBT đang dùng bản v4.1 hay v4.2",
            "Mô hình học máy dự báo có độ chính xác R-squared bao nhiêu",
        ],
        ChatTopic.ENERGY_PERFORMANCE: [
            "Cho xem sản lượng cao nhất và thấp nhất hôm nay",
            "Top trạm có hiệu suất năng lượng tốt nhất là gì",
            "Tình hình thời tiết tác động thế nào đến công suất",
        ],
        ChatTopic.SYSTEM_OVERVIEW: [
            "Tổng quan toàn bộ hệ thống các cơ sở năng lượng",
            "Tổng bao nhiêu cơ sở tổng sản lượng thế nào",
        ],
        ChatTopic.FACILITY_INFO: [
            "Trạm năng lượng này nằm ở tọa độ nào",
            "Vị trí địa lý quốc gia nào thành phố nào",
            "Thông tin địa chỉ của cơ sở là gì",
            "Múi giờ hiện tại của các trạm là gì",
        ],
    }

    def __init__(self, embedding_client=None) -> None:
        self._embedding_client = embedding_client
        self._topic_embeddings: dict[ChatTopic, list[list[float]]] = {}
        
    def initialize_semantic_router(self) -> None:
        """Pre-compute embeddings for canonical phrases."""
        if not self._embedding_client:
            return
        logger = __import__("logging").getLogger(__name__)
        try:
            flat_phrases = []
            topic_map = []
            for topic, phrases in self._TOPIC_CANONICAL_PHRASES.items():
                for phrase in phrases:
                    flat_phrases.append(phrase)
                    topic_map.append(topic)
                    
            embeddings = self._embedding_client.embed_batch(flat_phrases)
            
            for i, topic in enumerate(topic_map):
                if topic not in self._topic_embeddings:
                    self._topic_embeddings[topic] = []
                self._topic_embeddings[topic].append(embeddings[i])
            logger.info("Semantic router loaded %d canonical vectors.", len(flat_phrases))
        except Exception as e:
            logger.warning("Semantic router disabled: %s", e)
            self._topic_embeddings = {}
            self._embedding_client = None

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def detect_intent(self, message: str) -> IntentDetectionResult:
        normalized_message = normalize_vietnamese_text(message)
        if not normalized_message:
            raise ValueError("The question cannot be empty.")

        if self._embedding_client and self._topic_embeddings:
            try:
                user_vec = self._embedding_client.embed_text(message)
                max_sim = 0.0
                best_topic = None
                
                for topic, vecs in self._topic_embeddings.items():
                    for v in vecs:
                        sim = self._cosine_similarity(user_vec, v)
                        if sim > max_sim:
                            max_sim = sim
                            best_topic = topic
                            
                if max_sim > 0.65 and best_topic:
                    return IntentDetectionResult(
                        topic=best_topic,
                        confidence=round(max_sim, 2)
                    )
            except Exception as e:
                __import__("logging").getLogger(__name__).warning("Semantic routing failed, fallback to keyword: %s", e)
                self._embedding_client = None
                self._topic_embeddings = {}

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
