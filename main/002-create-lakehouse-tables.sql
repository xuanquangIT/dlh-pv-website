-- Lakehouse tables for Silver and Gold layers
-- Data is loaded from CSV files via the companion shell script.

-- Silver layer: hourly energy production
CREATE TABLE IF NOT EXISTS lh_silver_clean_hourly_energy (
    facility_code       VARCHAR(50)     NOT NULL,
    facility_name       VARCHAR(200),
    network_code        VARCHAR(20),
    network_region      VARCHAR(20),
    date_hour           TIMESTAMPTZ,
    energy_mwh          DOUBLE PRECISION DEFAULT 0,
    intervals_count     INTEGER         DEFAULT 0,
    quality_flag        VARCHAR(20),
    quality_issues      TEXT,
    completeness_pct    DOUBLE PRECISION DEFAULT 0,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);

-- Silver layer: hourly weather observations
CREATE TABLE IF NOT EXISTS lh_silver_clean_hourly_weather (
    facility_code       VARCHAR(50)     NOT NULL,
    facility_name       VARCHAR(200),
    timestamp           TIMESTAMPTZ,
    date_hour           TIMESTAMPTZ,
    date                DATE,
    shortwave_radiation DOUBLE PRECISION,
    direct_radiation    DOUBLE PRECISION,
    diffuse_radiation   DOUBLE PRECISION,
    direct_normal_irradiance DOUBLE PRECISION,
    temperature_2m      DOUBLE PRECISION,
    dew_point_2m        DOUBLE PRECISION,
    wet_bulb_temperature_2m DOUBLE PRECISION,
    cloud_cover         DOUBLE PRECISION,
    cloud_cover_low     DOUBLE PRECISION,
    cloud_cover_mid     DOUBLE PRECISION,
    cloud_cover_high    DOUBLE PRECISION,
    precipitation       DOUBLE PRECISION,
    sunshine_duration   DOUBLE PRECISION,
    total_column_integrated_water_vapour DOUBLE PRECISION,
    wind_speed_10m      DOUBLE PRECISION,
    wind_direction_10m  DOUBLE PRECISION,
    wind_gusts_10m      DOUBLE PRECISION,
    pressure_msl        DOUBLE PRECISION,
    is_valid            BOOLEAN,
    quality_flag        VARCHAR(20),
    quality_issues      TEXT,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);

-- Silver layer: hourly air quality
CREATE TABLE IF NOT EXISTS lh_silver_clean_hourly_air_quality (
    facility_code       VARCHAR(50)     NOT NULL,
    facility_name       VARCHAR(200),
    timestamp           TIMESTAMPTZ,
    date_hour           TIMESTAMPTZ,
    date                DATE,
    pm2_5               DOUBLE PRECISION,
    pm10                DOUBLE PRECISION,
    dust                DOUBLE PRECISION,
    nitrogen_dioxide    DOUBLE PRECISION,
    ozone               DOUBLE PRECISION,
    sulphur_dioxide     DOUBLE PRECISION,
    carbon_monoxide     DOUBLE PRECISION,
    uv_index            DOUBLE PRECISION,
    uv_index_clear_sky  DOUBLE PRECISION,
    aqi_category        VARCHAR(50),
    aqi_value           DOUBLE PRECISION,
    is_valid            BOOLEAN,
    quality_flag        VARCHAR(20),
    quality_issues      TEXT,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);

-- Gold layer: fact table for solar environmental metrics
CREATE TABLE IF NOT EXISTS lh_gold_fact_solar_environmental (
    facility_key        INTEGER,
    date_key            INTEGER,
    time_key            INTEGER,
    aqi_category_key    INTEGER,
    energy_mwh          DOUBLE PRECISION,
    intervals_count     INTEGER,
    shortwave_radiation DOUBLE PRECISION,
    direct_radiation    DOUBLE PRECISION,
    diffuse_radiation   DOUBLE PRECISION,
    direct_normal_irradiance DOUBLE PRECISION,
    irr_kwh_m2_hour     DOUBLE PRECISION,
    sunshine_hours      DOUBLE PRECISION,
    temperature_2m      DOUBLE PRECISION,
    dew_point_2m        DOUBLE PRECISION,
    humidity_2m         DOUBLE PRECISION,
    cloud_cover         DOUBLE PRECISION,
    cloud_cover_low     DOUBLE PRECISION,
    cloud_cover_mid     DOUBLE PRECISION,
    cloud_cover_high    DOUBLE PRECISION,
    precipitation       DOUBLE PRECISION,
    wind_speed_10m      DOUBLE PRECISION,
    wind_direction_10m  DOUBLE PRECISION,
    wind_gusts_10m      DOUBLE PRECISION,
    pressure_msl        DOUBLE PRECISION,
    pm2_5               DOUBLE PRECISION,
    pm10                DOUBLE PRECISION,
    dust                DOUBLE PRECISION,
    nitrogen_dioxide    DOUBLE PRECISION,
    ozone               DOUBLE PRECISION,
    sulphur_dioxide     DOUBLE PRECISION,
    carbon_monoxide     DOUBLE PRECISION,
    uv_index            DOUBLE PRECISION,
    uv_index_clear_sky  DOUBLE PRECISION,
    aqi_value           DOUBLE PRECISION,
    is_valid            BOOLEAN,
    quality_flag        VARCHAR(20),
    completeness_pct    DOUBLE PRECISION,
    yr_weighted_kwh     DOUBLE PRECISION,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);

-- Gold layer: facility dimension
CREATE TABLE IF NOT EXISTS lh_gold_dim_facility (
    facility_key                INTEGER PRIMARY KEY,
    facility_code               VARCHAR(50)     NOT NULL,
    facility_name               VARCHAR(200),
    location_lat                DOUBLE PRECISION,
    location_lng                DOUBLE PRECISION,
    total_capacity_mw           DOUBLE PRECISION,
    total_capacity_registered_mw DOUBLE PRECISION,
    total_capacity_maximum_mw   DOUBLE PRECISION
);

-- RAG: pgvector extension and document chunks
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type     VARCHAR(50)  NOT NULL,
    source_file  VARCHAR(500) NOT NULL,
    chunk_index  INTEGER      NOT NULL,
    content      TEXT         NOT NULL,
    embedding    vector(1536),
    created_at   TIMESTAMPTZ  DEFAULT now(),
    UNIQUE(source_file, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_rag_embedding
    ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
