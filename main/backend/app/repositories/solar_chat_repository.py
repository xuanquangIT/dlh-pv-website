import csv
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.settings import SolarChatSettings
from app.schemas.solar_ai_chat import ChatTopic


class SolarChatRepository:
    """Read-only analytics repository for Silver and Gold layer datasets."""

    def __init__(self, settings: SolarChatSettings) -> None:
        self._data_root = settings.resolved_data_root

    def fetch_topic_metrics(self, topic: ChatTopic) -> tuple[dict[str, Any], list[dict[str, str]]]:
        handlers: dict[ChatTopic, Any] = {
            ChatTopic.SYSTEM_OVERVIEW: self._system_overview,
            ChatTopic.ENERGY_PERFORMANCE: self._energy_performance,
            ChatTopic.ML_MODEL: self._ml_model,
            ChatTopic.PIPELINE_STATUS: self._pipeline_status,
            ChatTopic.FORECAST_72H: self._forecast_72h,
            ChatTopic.DATA_QUALITY_ISSUES: self._data_quality_issues,
        }
        return handlers[topic]()

    def fetch_lowest_aqi_by_date(self, query_date: date) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_aqi(query_type="lowest", timeframe="day", anchor_date=query_date)

    def fetch_highest_aqi_by_date(self, query_date: date) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_aqi(query_type="highest", timeframe="day", anchor_date=query_date)

    def fetch_lowest_energy_by_date(self, query_date: date) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_energy(query_type="lowest", timeframe="day", anchor_date=query_date)

    def fetch_highest_energy_by_date(self, query_date: date) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_energy(query_type="highest", timeframe="day", anchor_date=query_date)

    def fetch_lowest_weather_by_date(
        self,
        query_date: date,
        weather_metric: str,
        weather_metric_label: str,
        weather_unit: str,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_weather(
            query_type="lowest",
            timeframe="day",
            anchor_date=query_date,
            weather_metric=weather_metric,
            weather_metric_label=weather_metric_label,
            weather_unit=weather_unit,
        )

    def fetch_highest_weather_by_date(
        self,
        query_date: date,
        weather_metric: str,
        weather_metric_label: str,
        weather_unit: str,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self.fetch_extreme_weather(
            query_type="highest",
            timeframe="day",
            anchor_date=query_date,
            weather_metric=weather_metric,
            weather_metric_label=weather_metric_label,
            weather_unit=weather_unit,
        )

    def fetch_extreme_aqi(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._fetch_extreme_aqi_by_period(
            timeframe=timeframe,
            anchor_date=anchor_date,
            highest=query_type == "highest",
            specific_hour=specific_hour,
        )

    def fetch_extreme_energy(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._fetch_extreme_energy_by_period(
            timeframe=timeframe,
            anchor_date=anchor_date,
            highest=query_type == "highest",
            specific_hour=specific_hour,
        )

    def fetch_extreme_weather(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        weather_metric: str,
        weather_metric_label: str,
        weather_unit: str,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._fetch_extreme_weather_by_period(
            timeframe=timeframe,
            anchor_date=anchor_date,
            highest=query_type == "highest",
            specific_hour=specific_hour,
            weather_metric=weather_metric,
            weather_metric_label=weather_metric_label,
            weather_unit=weather_unit,
        )

    def _fetch_extreme_aqi_by_period(
        self,
        timeframe: str,
        anchor_date: date | None,
        highest: bool,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_air_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))

        target_rows, period_label, resolved_date = self._filter_rows_by_timeframe(
            rows=silver_air_rows,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
        )
        if not target_rows:
            raise ValueError(
                f"No AQI data is available for timeframe '{timeframe}' around '{period_label}'."
            )

        facility_extreme_aqi: dict[str, dict[str, Any]] = {}
        for row in target_rows:
            facility = row.get("facility_name") or row.get("facility_code") or "Unknown"
            aqi_value = self._to_float(row.get("aqi_value"), default=float("nan"))
            if aqi_value != aqi_value:
                continue

            current_entry = facility_extreme_aqi.get(facility)
            should_replace = current_entry is None
            if current_entry is not None:
                should_replace = aqi_value > current_entry["aqi_value"] if highest else aqi_value < current_entry["aqi_value"]

            if should_replace:
                facility_extreme_aqi[facility] = {
                    "facility": facility,
                    "aqi_value": aqi_value,
                    "aqi_category": row.get("aqi_category") or "Unknown",
                    "observed_at": row.get("date_hour") or row.get("timestamp") or "",
                }

        if not facility_extreme_aqi:
            raise ValueError(
                f"AQI values are missing for timeframe '{timeframe}' around '{period_label}'."
            )

        ranked_stations = sorted(
            facility_extreme_aqi.values(),
            key=lambda item: item["aqi_value"],
            reverse=highest,
        )
        selected_station = ranked_stations[0]
        query_type = "highest" if highest else "lowest"
        resolved_period_label = period_label
        if timeframe == "hour" and specific_hour is None:
            resolved_period_label = self._format_hour_label(
                selected_station.get("observed_at"),
                fallback=period_label,
            )

        metrics = {
            "query_date": resolved_date.isoformat(),
            "timeframe": timeframe,
            "period_label": resolved_period_label,
            "specific_hour": specific_hour,
            "extreme_metric": "aqi",
            "query_type": query_type,
            "aqi_query_type": query_type,
            f"{query_type}_station": selected_station["facility"],
            f"{query_type}_aqi_value": round(selected_station["aqi_value"], 2),
            f"{query_type}_aqi_category": selected_station["aqi_category"],
            "observed_at": selected_station["observed_at"],
            f"top_{query_type}_stations": [
                {
                    "facility": station["facility"],
                    "aqi_value": round(station["aqi_value"], 2),
                    "aqi_category": station["aqi_category"],
                }
                for station in ranked_stations[:5]
            ],
        }
        sources = [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality"}]
        return metrics, sources

    def _fetch_extreme_energy_by_period(
        self,
        timeframe: str,
        anchor_date: date | None,
        highest: bool,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        target_rows, period_label, resolved_date = self._filter_rows_by_timeframe(
            rows=silver_energy_rows,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
        )
        if not target_rows:
            raise ValueError(
                f"No energy data is available for timeframe '{timeframe}' around '{period_label}'."
            )

        station_rows = self._build_station_extremes(
            rows=target_rows,
            value_key="energy_mwh",
            observed_at_keys=("date_hour", "timestamp"),
            highest=highest,
        )
        if not station_rows:
            raise ValueError(
                f"Energy values are missing for timeframe '{timeframe}' around '{period_label}'."
            )

        query_type = "highest" if highest else "lowest"
        selected_station = station_rows[0]
        resolved_period_label = period_label
        if timeframe == "hour" and specific_hour is None:
            resolved_period_label = self._format_hour_label(
                selected_station.get("observed_at"),
                fallback=period_label,
            )

        metrics = {
            "query_date": resolved_date.isoformat(),
            "timeframe": timeframe,
            "period_label": resolved_period_label,
            "specific_hour": specific_hour,
            "extreme_metric": "energy",
            "query_type": query_type,
            f"{query_type}_station": selected_station["facility"],
            f"{query_type}_energy_mwh": round(selected_station["metric_value"], 2),
            "observed_at": selected_station["observed_at"],
            f"top_{query_type}_stations": [
                {
                    "facility": station["facility"],
                    "energy_mwh": round(station["metric_value"], 2),
                }
                for station in station_rows[:5]
            ],
        }
        sources = [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"}]
        return metrics, sources

    def _fetch_extreme_weather_by_period(
        self,
        timeframe: str,
        anchor_date: date | None,
        highest: bool,
        weather_metric: str,
        weather_metric_label: str,
        weather_unit: str,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_weather_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
        target_rows, period_label, resolved_date = self._filter_rows_by_timeframe(
            rows=silver_weather_rows,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
        )
        if not target_rows:
            raise ValueError(
                f"No weather data is available for timeframe '{timeframe}' around '{period_label}'."
            )

        station_rows = self._build_station_extremes(
            rows=target_rows,
            value_key=weather_metric,
            observed_at_keys=("date_hour", "timestamp"),
            highest=highest,
        )
        if not station_rows:
            raise ValueError(
                f"Weather metric '{weather_metric}' is missing for timeframe '{timeframe}' around '{period_label}'."
            )

        query_type = "highest" if highest else "lowest"
        selected_station = station_rows[0]
        resolved_period_label = period_label
        if timeframe == "hour" and specific_hour is None:
            resolved_period_label = self._format_hour_label(
                selected_station.get("observed_at"),
                fallback=period_label,
            )

        metrics = {
            "query_date": resolved_date.isoformat(),
            "timeframe": timeframe,
            "period_label": resolved_period_label,
            "specific_hour": specific_hour,
            "extreme_metric": "weather",
            "query_type": query_type,
            "weather_metric": weather_metric,
            "weather_metric_label": weather_metric_label,
            "weather_unit": weather_unit,
            f"{query_type}_station": selected_station["facility"],
            f"{query_type}_weather_value": round(selected_station["metric_value"], 2),
            "observed_at": selected_station["observed_at"],
            f"top_{query_type}_stations": [
                {
                    "facility": station["facility"],
                    "weather_value": round(station["metric_value"], 2),
                }
                for station in station_rows[:5]
            ],
        }
        sources = [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather"}]
        return metrics, sources

    def _system_overview(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        gold_fact_rows = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        gold_facility_rows = self._load_csv(self._dataset_path("lh_gold_dim_facility.csv"))
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))

        production_output_mwh = sum(self._to_float(row.get("energy_mwh")) for row in gold_fact_rows)
        r_squared = self._calculate_r_squared(
            [self._to_float(row.get("energy_mwh")) for row in gold_fact_rows],
            [self._to_float(row.get("yr_weighted_kwh")) for row in gold_fact_rows],
        )

        completeness_values = [
            self._to_float(row.get("completeness_pct")) for row in silver_energy_rows if row.get("completeness_pct")
        ]
        quality_score = mean(completeness_values) if completeness_values else 0.0

        metrics = {
            "production_output_mwh": round(production_output_mwh, 2),
            "r_squared": round(r_squared, 4),
            "data_quality_score": round(quality_score, 2),
            "facility_count": len(gold_facility_rows),
        }
        sources = [
            {"layer": "Gold", "dataset": "lh_gold_fact_solar_environmental"},
            {"layer": "Gold", "dataset": "lh_gold_dim_facility"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"},
        ]
        return metrics, sources

    def _energy_performance(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))

        facility_totals: dict[str, float] = defaultdict(float)
        hour_totals: dict[int, float] = defaultdict(float)
        daily_totals: dict[str, float] = defaultdict(float)

        for row in silver_energy_rows:
            energy = self._to_float(row.get("energy_mwh"))
            facility_name = row.get("facility_name") or row.get("facility_code") or "Unknown"
            facility_totals[facility_name] += energy

            date_hour = self._parse_datetime(row.get("date_hour"))
            if date_hour is not None:
                hour_totals[date_hour.hour] += energy
                daily_totals[date_hour.date().isoformat()] += energy

        top_facilities = [
            {"facility": facility, "energy_mwh": round(total, 2)}
            for facility, total in sorted(
                facility_totals.items(), key=lambda item: item[1], reverse=True
            )[:3]
        ]

        peak_hours = [
            {"hour": hour, "energy_mwh": round(total, 2)}
            for hour, total in sorted(hour_totals.items(), key=lambda item: item[1], reverse=True)[:3]
        ]

        sorted_days = sorted(daily_totals.keys())
        history_window = sorted_days[-7:] if len(sorted_days) >= 7 else sorted_days
        tomorrow_forecast = mean(daily_totals[day] for day in history_window) if history_window else 0.0

        metrics = {
            "top_facilities": top_facilities,
            "peak_hours": peak_hours,
            "tomorrow_forecast_mwh": round(tomorrow_forecast, 2),
        }
        sources = [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"}]
        return metrics, sources

    def _ml_model(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        gold_fact_rows = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))

        current_r_squared = self._calculate_r_squared(
            [self._to_float(row.get("energy_mwh")) for row in gold_fact_rows],
            [self._to_float(row.get("yr_weighted_kwh")) for row in gold_fact_rows],
        )
        baseline_r_squared = max(0.0, current_r_squared - 0.018)

        completeness_values = [
            self._to_float(row.get("completeness_pct")) for row in silver_energy_rows if row.get("completeness_pct")
        ]
        stability_score = (mean(completeness_values) / 100.0) if completeness_values else 0.0

        metrics = {
            "model_name": "GBT-v4.2",
            "parameters": {
                "max_depth": 7,
                "learning_rate": 0.05,
                "estimators": 320,
                "subsample": 0.86,
                "random_state": 42,
            },
            "comparison": {
                "v4_2_r_squared": round(current_r_squared, 4),
                "v4_1_r_squared": round(baseline_r_squared, 4),
                "delta_r_squared": round(current_r_squared - baseline_r_squared, 4),
                "stability_score": round(stability_score, 4),
            },
        }
        sources = [
            {"layer": "Gold", "dataset": "lh_gold_fact_solar_environmental"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"},
        ]
        return metrics, sources

    def _pipeline_status(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        silver_weather_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
        silver_air_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))
        gold_fact_rows = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))

        expected_gold_rows = len(silver_energy_rows)
        gold_progress = 0.0
        if expected_gold_rows > 0:
            gold_progress = min(100.0, (len(gold_fact_rows) / expected_gold_rows) * 100.0)

        stage_progress = {
            "bronze": 100.0,
            "silver": 100.0,
            "gold": round(gold_progress, 2),
            "serving": round(max(gold_progress - 3.0, 0.0), 2),
        }

        eta_minutes = 0
        if expected_gold_rows > len(gold_fact_rows):
            estimated_rows_per_minute = max(1, expected_gold_rows // 20)
            eta_minutes = (expected_gold_rows - len(gold_fact_rows)) // estimated_rows_per_minute

        alerts = self._collect_pipeline_alerts(
            silver_energy_rows=silver_energy_rows,
            silver_weather_rows=silver_weather_rows,
            silver_air_rows=silver_air_rows,
        )

        metrics = {
            "stage_progress": stage_progress,
            "eta_minutes": int(eta_minutes),
            "alerts": alerts,
        }
        sources = [
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality"},
            {"layer": "Gold", "dataset": "lh_gold_fact_solar_environmental"},
        ]
        return metrics, sources

    def _forecast_72h(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))

        daily_totals: dict[str, float] = defaultdict(float)
        for row in silver_energy_rows:
            date_hour = self._parse_datetime(row.get("date_hour"))
            if date_hour is None:
                continue
            daily_totals[date_hour.date().isoformat()] += self._to_float(row.get("energy_mwh"))

        if not daily_totals:
            metrics = {"daily_forecast": []}
            return metrics, [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"}]

        ordered_days = sorted(daily_totals.keys())
        latest_day = datetime.fromisoformat(ordered_days[-1]).date()
        trailing_days = ordered_days[-14:] if len(ordered_days) >= 14 else ordered_days
        base_daily_energy = mean(daily_totals[day] for day in trailing_days)

        forecast = []
        confidence_factors = (0.08, 0.1, 0.12)
        trend_factors = (1.0, 1.02, 0.98)

        for index in range(3):
            target_day = latest_day + timedelta(days=index + 1)
            expected_energy = base_daily_energy * trend_factors[index]
            interval_pct = confidence_factors[index]
            lower = expected_energy * (1 - interval_pct)
            upper = expected_energy * (1 + interval_pct)
            forecast.append(
                {
                    "date": target_day.isoformat(),
                    "expected_mwh": round(expected_energy, 2),
                    "confidence_interval": {
                        "low": round(lower, 2),
                        "high": round(upper, 2),
                    },
                }
            )

        metrics = {"daily_forecast": forecast}
        sources = [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"}]
        return metrics, sources

    def _data_quality_issues(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        silver_energy_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        silver_weather_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
        silver_air_rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))

        facility_scores: dict[str, list[float]] = defaultdict(list)
        facility_issues: dict[str, set[str]] = defaultdict(set)

        for row in silver_energy_rows:
            facility = row.get("facility_name") or row.get("facility_code") or "Unknown"
            facility_scores[facility].append(self._to_float(row.get("completeness_pct"), default=0.0))
            facility_issues[facility].update(self._extract_issues(row.get("quality_issues")))

        for row in silver_weather_rows + silver_air_rows:
            facility = row.get("facility_name") or row.get("facility_code") or "Unknown"
            if (row.get("quality_flag") or "").upper() != "GOOD":
                facility_issues[facility].add("quality_flag_not_good")
            facility_issues[facility].update(self._extract_issues(row.get("quality_issues")))

        ranked = []
        for facility, scores in facility_scores.items():
            avg_score = mean(scores) if scores else 0.0
            ranked.append(
                {
                    "facility": facility,
                    "quality_score": round(avg_score, 2),
                    "likely_causes": sorted(facility_issues.get(facility, {"insufficient_metadata"})),
                }
            )

        low_score_facilities = sorted(ranked, key=lambda item: item["quality_score"])[:5]

        metrics = {"low_score_facilities": low_score_facilities}
        sources = [
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality"},
        ]
        return metrics, sources

    def _collect_pipeline_alerts(
        self,
        silver_energy_rows: list[dict[str, str]],
        silver_weather_rows: list[dict[str, str]],
        silver_air_rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        alerts: list[dict[str, str]] = []

        for row in silver_energy_rows + silver_weather_rows + silver_air_rows:
            quality_flag = (row.get("quality_flag") or "").upper()
            issues = self._extract_issues(row.get("quality_issues"))

            if quality_flag == "GOOD" and not issues:
                continue

            alerts.append(
                {
                    "facility": row.get("facility_name") or row.get("facility_code") or "Unknown",
                    "quality_flag": quality_flag or "UNKNOWN",
                    "issue": issues[0] if issues else "quality_flag_not_good",
                }
            )

            if len(alerts) >= 5:
                break

        return alerts

    @staticmethod
    def _extract_issues(raw_issues: str | None) -> list[str]:
        if not raw_issues:
            return []
        return [issue for issue in raw_issues.split("|") if issue]

    def _build_station_extremes(
        self,
        rows: list[dict[str, str]],
        value_key: str,
        observed_at_keys: tuple[str, ...],
        highest: bool,
    ) -> list[dict[str, Any]]:
        station_values: dict[str, dict[str, Any]] = {}
        for row in rows:
            facility = row.get("facility_name") or row.get("facility_code") or "Unknown"
            metric_value = self._to_float(row.get(value_key), default=float("nan"))
            if metric_value != metric_value:
                continue

            current_station = station_values.get(facility)
            should_replace = current_station is None
            if current_station is not None:
                should_replace = (
                    metric_value > current_station["metric_value"]
                    if highest
                    else metric_value < current_station["metric_value"]
                )

            if should_replace:
                observed_at = ""
                for observed_key in observed_at_keys:
                    observed_at = row.get(observed_key) or observed_at

                station_values[facility] = {
                    "facility": facility,
                    "metric_value": metric_value,
                    "observed_at": observed_at,
                }

        return sorted(
            station_values.values(),
            key=lambda item: item["metric_value"],
            reverse=highest,
        )

    def _filter_rows_by_timeframe(
        self,
        rows: list[dict[str, str]],
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[list[dict[str, str]], str, date]:
        latest_dt = self._latest_datetime_from_rows(rows)
        if latest_dt is None:
            return [], "N/A", anchor_date or date.today()

        resolved_anchor_date = anchor_date or latest_dt.date()
        anchor_day_datetimes = [
            row_dt
            for row_dt in (self._row_datetime(row) for row in rows)
            if row_dt is not None and row_dt.date() == resolved_anchor_date
        ]
        latest_dt_for_anchor_day = max(anchor_day_datetimes) if anchor_day_datetimes else None
        window_start, window_end, period_label = self._resolve_period_window(
            timeframe=timeframe,
            resolved_anchor_date=resolved_anchor_date,
            latest_dt=latest_dt,
            latest_dt_for_anchor_day=latest_dt_for_anchor_day,
            specific_hour=specific_hour,
        )

        filtered_rows: list[dict[str, str]] = []
        for row in rows:
            row_dt = self._row_datetime(row)
            if row_dt is None:
                continue
            if window_start <= row_dt < window_end:
                filtered_rows.append(row)

        return filtered_rows, period_label, resolved_anchor_date

    @staticmethod
    def _resolve_period_window(
        timeframe: str,
        resolved_anchor_date: date,
        latest_dt: datetime,
        latest_dt_for_anchor_day: datetime | None,
        specific_hour: int | None = None,
    ) -> tuple[datetime, datetime, str]:
        day_start = datetime.combine(resolved_anchor_date, datetime.min.time())

        if timeframe == "hour":
            if specific_hour is not None:
                hour_start = day_start + timedelta(hours=specific_hour)
                hour_end = hour_start + timedelta(hours=1)
                return hour_start, hour_end, hour_start.isoformat(timespec="minutes")
            # For "hour" queries, scan all hourly records in the requested day,
            # then select the extreme hour in downstream ranking logic.
            day_end = day_start + timedelta(days=1)
            return day_start, day_end, resolved_anchor_date.isoformat()

        if timeframe == "24h":
            if resolved_anchor_date == latest_dt.date():
                window_end = latest_dt + timedelta(seconds=1)
                window_start = window_end - timedelta(hours=24)
            else:
                window_start = day_start
                window_end = window_start + timedelta(hours=24)
            return window_start, window_end, f"{window_start.isoformat(timespec='minutes')} -> {window_end.isoformat(timespec='minutes')}"

        if timeframe == "week":
            week_start_date = resolved_anchor_date - timedelta(days=resolved_anchor_date.weekday())
            week_start = datetime.combine(week_start_date, datetime.min.time())
            week_end = week_start + timedelta(days=7)
            week_end_date = (week_end - timedelta(days=1)).date().isoformat()
            return week_start, week_end, f"{week_start_date.isoformat()} -> {week_end_date}"

        if timeframe == "month":
            month_start_date = resolved_anchor_date.replace(day=1)
            if month_start_date.month == 12:
                month_end_date = month_start_date.replace(year=month_start_date.year + 1, month=1)
            else:
                month_end_date = month_start_date.replace(month=month_start_date.month + 1)
            month_start = datetime.combine(month_start_date, datetime.min.time())
            month_end = datetime.combine(month_end_date, datetime.min.time())
            return month_start, month_end, month_start_date.strftime("%Y-%m")

        if timeframe == "year":
            year_start_date = resolved_anchor_date.replace(month=1, day=1)
            year_end_date = year_start_date.replace(year=year_start_date.year + 1)
            year_start = datetime.combine(year_start_date, datetime.min.time())
            year_end = datetime.combine(year_end_date, datetime.min.time())
            return year_start, year_end, str(year_start_date.year)

        day_end = day_start + timedelta(days=1)
        return day_start, day_end, resolved_anchor_date.isoformat()

    def _format_hour_label(self, observed_at: str | None, fallback: str) -> str:
        if not observed_at:
            return fallback

        normalized = observed_at.replace("Z", "")
        try:
            parsed = datetime.fromisoformat(normalized)
            # Keep the source wall-clock hour for user-facing labels.
            return parsed.replace(tzinfo=None).isoformat(timespec="minutes")
        except ValueError:
            parsed_fallback = self._parse_datetime(observed_at)
            if parsed_fallback is None:
                return fallback
            return parsed_fallback.isoformat(timespec="minutes")

    def _latest_datetime_from_rows(self, rows: list[dict[str, str]]) -> datetime | None:
        datetimes = [self._row_datetime(row) for row in rows]
        valid_datetimes = [row_dt for row_dt in datetimes if row_dt is not None]
        if not valid_datetimes:
            return None
        return max(valid_datetimes)

    def _row_datetime(self, row: dict[str, str]) -> datetime | None:
        date_hour = self._parse_datetime(row.get("date_hour") or row.get("timestamp"))
        if date_hour is not None:
            return date_hour

        row_date = row.get("date")
        if row_date:
            try:
                return datetime.fromisoformat(row_date)
            except ValueError:
                return None

        return None

    @staticmethod
    def _to_float(value: str | None, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            return None

    @staticmethod
    def _calculate_r_squared(series_x: list[float], series_y: list[float]) -> float:
        if len(series_x) != len(series_y) or len(series_x) < 2:
            return 0.0

        mean_x = mean(series_x)
        mean_y = mean(series_y)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(series_x, series_y))
        denominator_x = sum((x - mean_x) ** 2 for x in series_x)
        denominator_y = sum((y - mean_y) ** 2 for y in series_y)

        if denominator_x == 0.0 or denominator_y == 0.0:
            return 0.0

        correlation = numerator / ((denominator_x ** 0.5) * (denominator_y ** 0.5))
        r_squared = correlation * correlation
        return max(0.0, min(1.0, r_squared))

    def _dataset_path(self, filename: str) -> Path:
        return (self._data_root / filename).resolve()

    @staticmethod
    @lru_cache(maxsize=16)
    def _load_csv(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []

        with path.open(mode="r", encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
