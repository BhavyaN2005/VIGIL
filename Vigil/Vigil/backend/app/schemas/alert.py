from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, field_serializer


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: Optional[Union[datetime, str]] = None
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    attack_class: str
    severity: str
    confidence: float
    is_zero_day: bool
    reconstruction_error: float
    top_contributing_feature: str
    model_used: str
    status: str
    analyst_notes: Optional[str] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: Optional[Union[datetime, str]]) -> Optional[str]:
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return timestamp


class AlertUpdate(BaseModel):
    status: Optional[str] = None  # "Open" | "Investigating" | "Resolved"
    analyst_notes: Optional[str] = None


class PaginatedAlerts(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[AlertRead]
