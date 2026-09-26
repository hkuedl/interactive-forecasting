"""Application configuration. Secret values are read from the environment only."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IFORECAST_", env_file=".env", extra="ignore")

    database_url: str | None = None
    runtime_data_root: Path = PROJECT_ROOT / "runtime_data"
    artifact_root: Path = PROJECT_ROOT / "artifacts"
    experiment_config_root: Path = PROJECT_ROOT / "configs" / "experiments"
    log_level: str = "INFO"
    openai_model: str | None = None
    openai_api_key: SecretStr | None = None
    worker_concurrency: int = Field(default=1, ge=1)

    @field_validator("database_url", mode="before")
    @classmethod
    def empty_database_url(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("openai_model", "openai_api_key", mode="before")
    @classmethod
    def empty_optional_secret(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{PROJECT_ROOT / 'data' / 'interactive_forecasting.sqlite3'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
