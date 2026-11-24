from typing import Dict, List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    engine: str
    loaded_models: List[str]
    active_default_model: str
    zero_day_detector: str
    feature_dimensions: int
    uptime_seconds: float
    memory_mb: float
    database_status: str
