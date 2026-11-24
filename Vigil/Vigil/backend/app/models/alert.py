import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from app.core.database import Base


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utc_now, index=True)
    src_ip = Column(String(64), default="192.168.10.50", index=True)
    dst_ip = Column(String(64), default="172.16.0.1", index=True)
    dst_port = Column(Integer, default=80)
    protocol = Column(String(16), default="TCP")
    attack_class = Column(String(32), index=True, nullable=False)
    severity = Column(String(16), index=True, nullable=False)  # "Low", "Medium", "High", "Critical"
    confidence = Column(Float, nullable=False)
    is_zero_day = Column(Boolean, default=False, index=True)
    reconstruction_error = Column(Float, default=0.0)
    top_contributing_feature = Column(String(128), default="")
    model_used = Column(String(32), default="XGBoost")
    status = Column(String(16), default="Open", index=True)  # "Open", "Investigating", "Resolved"
    analyst_notes = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "attack_class": self.attack_class,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "is_zero_day": self.is_zero_day,
            "reconstruction_error": round(self.reconstruction_error, 5),
            "top_contributing_feature": self.top_contributing_feature,
            "model_used": self.model_used,
            "status": self.status,
            "analyst_notes": self.analyst_notes,
        }
