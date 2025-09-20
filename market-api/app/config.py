from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "NAXS Market Realtime API"
    API_PREFIX: str = "/api"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: str = "*"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./naxs.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 10
    REDIS_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True

    # Cache TTLs (seconds)
    TTL_SPOT_ALL: float = 4.0
    TTL_MINUTE: float = 2.5
    TTL_MARKET_DATA: float = 30.0
    TTL_FACTOR_DATA: float = 300.0
    CACHE_PREFIX: str = "naxs:"

    # Performance Configuration
    MAX_WORKERS: int = 4
    WORKER_CONNECTIONS: int = 1000
    KEEPALIVE_TIMEOUT: int = 5
    REQUEST_TIMEOUT: int = 60  # 增加到60秒以避免超时
    MAX_REQUEST_SIZE: int = 16 * 1024 * 1024  # 16MB

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 20

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: List[str] = ["TimeoutError", "ConnectionError"]

    # Pagination defaults
    DEFAULT_PAGE_SIZE: int = 100
    MAX_PAGE_SIZE: int = 1000

    # Parquet root for /api/bars and admin jobs
    PARQUET_ROOT: str = "data/parquet"

    # Simple admin auth token (send via X-Admin-Token header)
    ADMIN_TOKEN: str = "changeme"  # 默认值，生产环境请通过环境变量设置

    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_TRACING: bool = True

    # Health Check
    HEALTH_CHECK_INTERVAL: int = 30
    HEALTH_CHECK_TIMEOUT: int = 5

    # Memory Management
    MAX_MEMORY_USAGE: float = 0.8  # 80% of available memory
    GC_THRESHOLD: int = 1000
    OBJECT_POOL_SIZE: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()





