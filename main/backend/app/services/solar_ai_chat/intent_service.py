import logging
from dataclasses import dataclass
from unicodedata import normalize

from app.schemas.solar_ai_chat import ChatTopic


logger = logging.getLogger(__name__)
_INTENT_CACHE_LIMIT = 256


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
            "low-score",
            "low score facility",
            "low score facilities",
            "data quality",
            "quality alert",
            "quality alerts",
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
            "72 hours",
            "72 hour",
            "next 72",
            "next 72 hours",
            "next three days",
            "three day forecast",
            "three-day forecast",
            "3-day forecast",
        ),
        ChatTopic.PIPELINE_STATUS: (
            "pipeline",
            "tien do",
            "eta",
            "trang thai",
            "pipeline alert",
            "pipeline alerts",
            "canh bao",
        ),
        ChatTopic.ML_MODEL: (
            "mo hinh",
            "model",
            "fallback",
            "du phong",
            "gbt",
            "v4.2",
            "v4.1",
            "version",
            "model version",
            "current version",
            "forecast model",
            "tham so",
            "model r-squared",
            "r-squared model",
        ),
        ChatTopic.ENERGY_PERFORMANCE: (
            "hieu suat",
            "performance",
            "performance ratio",
            "capacity factor",
            "he so cong suat",
            "ti le hieu suat",
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
            "so sanh",
            "compare",
            "comparison",
            "top 2",
            "top 3",
            "top 5",
            "2 facilities",
            "2 facility",
            "top facilities",
            "2 tram",
            "hai tram",
            "3 tram",
            "ba tram",
        ),
        ChatTopic.SYSTEM_OVERVIEW: (
            "tong quan",
            "he thong",
            "san luong",
            "r2",
            "r-squared",
            "so co so",
            "facility count",
            "system overview",
            "current system overview",
            "overall system",
            "overall production",
            "production output",
            "quality score",
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
            "cong suat lap dat",
            "installed capacity",
            "largest capacity",
            "biggest station",
            "lon nhat",
            "nho nhat",
            "tram lon nhat",
            "nha may lon nhat",
            "co so lon nhat",
            "largest facilities",
            "highest facilities",
            "facility lon nhat",
            "facilities lon nhat",
            "thong tin tram",
            "thong tin co so",
            "thong tin nha may",
            "dia ly",
            "dia chi",
            "nam o",
            "dat o",
            "tram nao o",
            "co so o",
            "so tram",
            "bao nhieu tram",
            "tong so tram",
            "how many stations",
            "station count",
            "active stations",
            "list all stations",
            "liet ke tram",
            "liet ke tat ca",
        ),
    }

    _TOPIC_CANONICAL_PHRASES: dict[ChatTopic, list[str]] = {
        ChatTopic.DATA_QUALITY_ISSUES: [
            "Chất lượng dữ liệu hôm nay có tốt không",
            "Có trạm nào bị cảnh báo lỗi dữ liệu hay AQI thấp không",
            "Nguyên nhân trạm bị rớt dữ liệu là gì",
            "Any data quality alerts today",
            "Which facilities have low-score data quality",
        ],
        ChatTopic.FORECAST_72H: [
            "Dự báo sản lượng năng lượng trong 72 giờ tới",
            "Năng lượng khoảng tin cậy 3 ngày tiếp theo",
            "Xu hướng thời tiết 3 ngày tới ra sao",
            "Show expected energy production for the next 72 hours",
            "What is the three-day forecast confidence interval",
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
            "Give me the current system overview",
            "Show facility count, production output, quality score, and R-squared",
        ],
        ChatTopic.FACILITY_INFO: [
            "Trạm năng lượng này nằm ở tọa độ nào",
            "Vị trí địa lý quốc gia nào thành phố nào",
            "Thông tin địa chỉ của cơ sở là gì",
            "Múi giờ hiện tại của các trạm là gì",
        ],
    }

    def __init__(
        self,
        embedding_client=None,
        semantic_enabled: bool = True,
        semantic_min_confidence: float = 0.65,
        semantic_keyword_score_threshold: int = 1,
    ) -> None:
        self._embedding_client = embedding_client
        self._semantic_enabled = bool(semantic_enabled)
        self._semantic_min_confidence = max(0.0, min(1.0, float(semantic_min_confidence)))
        self._semantic_keyword_score_threshold = max(0, int(semantic_keyword_score_threshold))
        self._topic_embeddings: dict[ChatTopic, list[list[float]]] = {}
        self._intent_cache: dict[str, IntentDetectionResult] = {}

    def initialize_semantic_router(self) -> None:
        """Pre-compute embeddings for canonical phrases."""
        if not self._semantic_enabled or not self._embedding_client:
            return

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

        cached = self._intent_cache.get(normalized_message)
        if cached is not None:
            return cached

        matched_topic, matched_score = self._keyword_match(normalized_message)
        semantic_result: IntentDetectionResult | None = None

        if self._semantic_enabled and self._embedding_client and self._topic_embeddings:
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

                if max_sim >= self._semantic_min_confidence and best_topic:
                    semantic_result = IntentDetectionResult(
                        topic=best_topic,
                        confidence=round(max_sim, 2),
                    )
            except Exception as e:
                logger.warning("Semantic routing failed, fallback to keyword: %s", e)
                self._embedding_client = None
                self._topic_embeddings = {}

        # Fast path: strong keyword match, but semantic can still override when signals disagree.
        if matched_topic is not None and matched_score >= self._semantic_keyword_score_threshold:
            keyword_result = IntentDetectionResult(
                topic=matched_topic,
                confidence=self._keyword_confidence(matched_score),
            )
            # Keep strong keyword hits stable; only let semantic override when
            # keyword evidence is weak.
            if (
                semantic_result is not None
                and semantic_result.topic != keyword_result.topic
                and keyword_result.confidence < 0.8
            ):
                result = semantic_result
            elif (
                semantic_result is not None
                and semantic_result.confidence >= (keyword_result.confidence + 0.08)
                and keyword_result.confidence < 0.8
            ):
                result = semantic_result
            else:
                result = keyword_result
            self._cache_intent(normalized_message, result)
            return result

        if semantic_result is not None:
            self._cache_intent(normalized_message, semantic_result)
            return semantic_result

        if matched_topic is None:
            result = IntentDetectionResult(topic=ChatTopic.GENERAL, confidence=0.3)
            self._cache_intent(normalized_message, result)
            return result

        result = IntentDetectionResult(
            topic=matched_topic,
            confidence=self._keyword_confidence(matched_score),
        )
        self._cache_intent(normalized_message, result)
        return result

    def _keyword_match(self, normalized_message: str) -> tuple[ChatTopic | None, int]:
        matched_topic: ChatTopic | None = None
        matched_score = 0

        facility_priority_markers = (
            "installed capacity",
            "cong suat lap dat",
            "largest installed",
            "largest capacity",
            "station count",
            "how many stations",
            "bao nhieu tram",
            "tong so tram",
            "list all stations",
            "liet ke",
            "timezone of that station",
            "mui gio cua tram do",
            "tram do",
            "we just discussed",
            "luc dau",
        )
        facility_bias = 2 if any(m in normalized_message for m in facility_priority_markers) else 0

        ml_priority_markers = (
            "fallback",
            "du phong",
            "model version",
            "current version",
            "forecast model",
            "r-squared",
            "skill score",
            "nrmse",
        )
        ml_bias = 2 if any(m in normalized_message for m in ml_priority_markers) else 0

        for topic, keywords in self._TOPIC_KEYWORDS.items():
            topic_score = sum(1 for keyword in keywords if keyword in normalized_message)
            if topic is ChatTopic.ENERGY_PERFORMANCE and self._is_energy_comparison_query(normalized_message):
                # Bias toward ENERGY_PERFORMANCE for facility comparison requests.
                topic_score += 2
            if topic is ChatTopic.FACILITY_INFO and facility_bias:
                topic_score += facility_bias
            if topic is ChatTopic.ML_MODEL and ml_bias:
                topic_score += ml_bias
            if topic_score > matched_score:
                matched_topic = topic
                matched_score = topic_score

        return matched_topic, matched_score

    @staticmethod
    def _is_energy_comparison_query(normalized_message: str) -> bool:
        compare_markers = ("so sanh", "compare", "comparison", "versus", " vs ")
        facility_markers = ("facility", "facilities", "tram", "co so", "nha may")
        ranking_markers = (
            "top",
            "top 2",
            "largest",
            "highest",
            "lon nhat",
            "nho nhat",
            "2 facilities",
            "2 facility",
            "hai tram",
            "hai co so",
            "2 tram",
        )
        # Don't fire on AQI/weather metric queries — those belong to data_quality
        if "aqi" in normalized_message or "chi so aqi" in normalized_message:
            return False
        # Don't fire on installed capacity / facility info queries
        capacity_markers = (
            "cong suat lap dat", "installed capacity", "capacity mw",
            "cong suat", "bao nhieu tram", "so tram", "tong so tram",
            "how many stations", "station count", "liet ke",
        )
        if any(m in normalized_message for m in capacity_markers):
            return False
        has_compare = any(marker in normalized_message for marker in compare_markers)
        has_facility = any(marker in normalized_message for marker in facility_markers)
        has_ranking = any(marker in normalized_message for marker in ranking_markers)
        # Fire the bias if: (explicit comparison) OR (facility + ranking together)
        return has_facility and has_ranking or (has_compare and has_facility)

    @staticmethod
    def _keyword_confidence(matched_score: int) -> float:
        return round(min(0.99, 0.5 + (matched_score * 0.15)), 2)

    def _cache_intent(self, normalized_message: str, result: IntentDetectionResult) -> None:
        if normalized_message in self._intent_cache:
            self._intent_cache.pop(normalized_message, None)
        elif len(self._intent_cache) >= _INTENT_CACHE_LIMIT:
            oldest_key = next(iter(self._intent_cache))
            self._intent_cache.pop(oldest_key, None)
        self._intent_cache[normalized_message] = result
