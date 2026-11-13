"""Configuration comes from the environment (INDEX_PATH, LOG_LEVEL, ...)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    index_path: Path = Path("index.bin")
    log_level: str = "INFO"
    max_page_size: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
