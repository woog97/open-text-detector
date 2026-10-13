from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, DotEnvSettingsSource, EnvSettingsSource, SettingsConfigDict

ENV_PREFIX = "OTD_"
# Settings were named PANGRAM_* before the project became Open Text Detector. They are still read, below the new names.
LEGACY_ENV_PREFIX = "PANGRAM_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, env_file=".env", extra="ignore")
    data_dir: Path = Path(".data")
    result_storage: Literal["local", "s3"] = "local"
    result_s3_endpoint: str | None = None
    result_s3_bucket: str | None = None
    result_s3_region: str = "auto"
    result_s3_access_key: SecretStr | None = None
    result_s3_secret_key: SecretStr | None = None
    result_s3_prefix: str = "findings/v1"
    result_cache_bytes: int = Field(default=32 * 1024 * 1024, ge=0)
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    worker_enabled: bool = True
    allow_local_model_endpoints: bool = False
    model_device: str = "auto"
    meld_precision: Literal["float32", "float16", "bfloat16"] = "float32"
    laya_runtime: Literal["auto", "torch", "mlx"] = "auto"
    laya_precision: Literal["float32", "float16", "bfloat16"] = "float16"
    laya_batch_size: int = Field(default=4, ge=1, le=64)
    meld_batch_size: int = Field(default=1, ge=1, le=8)
    model_dir: Path = Path(__file__).resolve().parents[2] / "models"
    dataset_dir: Path = Path(__file__).resolve().parents[2] / "research" / "data"
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    max_upload_bytes: int = 20 * 1024 * 1024
    max_request_bytes: int = 110 * 1024 * 1024
    max_text_chars: int = 500_000
    provider_timeout: float = 120
    lease_seconds: int = 90
    worker_poll_seconds: float = .4
    requests_per_minute: int = 120

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        legacy_env = EnvSettingsSource(settings_cls, env_prefix=LEGACY_ENV_PREFIX)
        legacy_dotenv = DotEnvSettingsSource(settings_cls, env_file=dotenv_settings.env_file, env_prefix=LEGACY_ENV_PREFIX)
        return init_settings, env_settings, dotenv_settings, legacy_env, legacy_dotenv, file_secret_settings
