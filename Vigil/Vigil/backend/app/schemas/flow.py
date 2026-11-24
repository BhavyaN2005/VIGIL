from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class SingleFlowRequest(BaseModel):
    # Either a dictionary mapping feature name -> float, OR an ordered list of 78 numeric floats
    features: Optional[Dict[str, float]] = None
    feature_vector: Optional[List[float]] = None
    
    # Optional network context
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = "TCP"

    @field_validator("feature_vector")
    @classmethod
    def check_vector_dim(cls, v):
        if v is not None and len(v) != 78:
            raise ValueError(f"Feature vector must contain exactly 78 features, received {len(v)}")
        return v


class SHAPContribution(BaseModel):
    feature_name: str
    raw_value: float
    scaled_value: float
    shap_score: float
    contribution_direction: str  # "positive" | "negative"


class ZeroDayDetection(BaseModel):
    reconstruction_error: float
    threshold_tau: float
    is_zero_day_anomaly: bool


class SingleFlowResponse(BaseModel):
    predicted_class: str
    confidence: float
    severity: str  # "Low" | "Medium" | "High" | "Critical"
    model_used: str
    class_probabilities: Dict[str, float]
    zero_day_detection: ZeroDayDetection
    base_value: float
    top_contributions: List[SHAPContribution]
    alert_logged: bool = False
    alert_id: Optional[int] = None


class BatchFlowItem(BaseModel):
    row_index: int
    predicted_class: str
    confidence: float
    severity: str
    is_zero_day_anomaly: bool
    reconstruction_error: float
    dst_port: int


class BatchFlowResponse(BaseModel):
    total_flows_processed: int
    model_used: str
    inference_latency_ms: float
    attack_summary: Dict[str, int]
    zero_day_anomalies_detected: int
    critical_alerts_count: int
    sample_flows: List[BatchFlowItem]
