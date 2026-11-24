from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="allow")

    PROJECT_NAME: str = "VIGIL XAI-NIDS API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Storage, Static Frontend & Model Artifacts
    DATABASE_URL: str = f"sqlite+aiosqlite:///{REPO_ROOT.as_posix()}/backend/vigil.db"
    STATIC_DIR: Path = REPO_ROOT / "frontend"
    ARTIFACTS_DIR: Path = REPO_ROOT / "ml" / "artifacts"
    SAMPLE_DATA_PATH: Path = REPO_ROOT / "data" / "sample" / "sample_flows.csv"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Streaming Defaults
    DEFAULT_STREAM_RATE: float = 2.0  # flows/second
    MAX_STREAM_RATE: float = 10.0
    MIN_STREAM_RATE: float = 0.5
    
    # Default Inference Model
    DEFAULT_MODEL: str = "xgb"  # "xgb" | "rf" | "lstm"

settings = Settings()
