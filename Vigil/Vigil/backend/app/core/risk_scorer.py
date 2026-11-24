"""
Composite Threat Risk Scoring Engine for VIGIL.
Calculates dynamic risk values combining ML model confidence, Autoencoder zero-day
anomaly deviation, asset business criticality, CTI reputation, and temporal posture.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Criticality weights for network asset classes
ASSET_INVENTORY: Dict[str, Dict[str, Any]] = {
    "10.0.4.15": {"name": "db-core-cluster-01", "role": "Primary Database Cluster", "criticality": 0.95, "zone": "Database Tier"},
    "10.0.4.20": {"name": "ad-controller-primary", "role": "Domain Controller / IAM", "criticality": 1.00, "zone": "Identity & Access"},
    "10.0.5.50": {"name": "pci-payment-gateway", "role": "PCI-DSS Payment Processor", "criticality": 0.98, "zone": "PCI Secured Zone"},
    "192.168.1.10": {"name": "dmz-nginx-ingress-01", "role": "DMZ Edge Reverse Proxy", "criticality": 0.75, "zone": "DMZ"},
    "192.168.1.15": {"name": "dmz-api-gateway", "role": "API Ingress Server", "criticality": 0.70, "zone": "DMZ"},
    "192.168.1.50": {"name": "vpn-concentrator", "role": "Remote Access Gateway", "criticality": 0.85, "zone": "Perimeter"},
    "172.16.2.100": {"name": "workstation-fin-04", "role": "Finance Dept Workstation", "criticality": 0.45, "zone": "User Workstations"},
    "172.16.2.101": {"name": "workstation-eng-09", "role": "Engineering Build Node", "criticality": 0.50, "zone": "User Workstations"},
    "172.16.2.102": {"name": "workstation-hr-02", "role": "HR Generalist Workstation", "criticality": 0.40, "zone": "User Workstations"}
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "NONE": 0.0,
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.80,
    "CRITICAL": 1.00
}


class RiskScorer:
    """
    Computes unified threat risk scores across network telemetry.
    """

    def __init__(
        self,
        w_ml: float = 0.35,
        w_zeroday: float = 0.20,
        w_asset: float = 0.25,
        w_cti: float = 0.15,
        w_time: float = 0.05
    ):
        self.w_ml = w_ml
        self.w_zeroday = w_zeroday
        self.w_asset = w_asset
        self.w_cti = w_cti
        self.w_time = w_time

    def get_asset_criticality(self, ip: Optional[str]) -> float:
        """
        Lookup or infer the business criticality of a host IP.
        """
        if not ip:
            return 0.30
        if ip in ASSET_INVENTORY:
            return ASSET_INVENTORY[ip]["criticality"]
        # Inferred by subnet
        if ip.startswith("10.0."):
            return 0.80  # Core internal servers
        if ip.startswith("192.168.1."):
            return 0.65  # DMZ perimeter
        if ip.startswith("172.16."):
            return 0.40  # Workstation endpoints
        return 0.30      # Generic endpoint

    def compute_risk(
        self,
        predicted_class: str,
        confidence: float,
        severity: str,
        is_zero_day: bool = False,
        reconstruction_error: float = 0.0,
        dst_ip: Optional[str] = None,
        cti_score: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate the dynamic composite risk score.
        """
        # 1. Base ML Component
        sev_factor = SEVERITY_WEIGHTS.get(severity.upper(), 0.5)
        ml_comp = confidence * sev_factor if predicted_class != "Benign" else 0.0

        # 2. Zero-Day Anomaly Component
        # Normalize error around baseline threshold 1.627
        zd_comp = 0.0
        if is_zero_day or reconstruction_error > 1.627:
            # Scale non-linearly
            excess = max(0.0, reconstruction_error - 1.627)
            zd_comp = min(1.0, 0.6 + excess * 0.4)

        # 3. Asset Criticality Component
        asset_crit = self.get_asset_criticality(dst_ip)

        # 4. CTI Threat Score Component
        cti_comp = min(1.0, max(0.0, cti_score))

        # 5. Temporal Posture (off-hours penalty)
        ts = timestamp or datetime.now(timezone.utc)
        hour = ts.hour
        is_off_hours = hour < 7 or hour > 19 or ts.weekday() >= 5
        time_comp = 0.85 if is_off_hours else 0.20

        # Weighted composite sum
        raw_score = (
            (self.w_ml * ml_comp) +
            (self.w_zeroday * zd_comp) +
            (self.w_asset * asset_crit) +
            (self.w_cti * cti_comp) +
            (self.w_time * time_comp)
        )

        # Dampen if traffic is strictly Benign and no zero-day indicator
        if predicted_class == "Benign" and not is_zero_day and reconstruction_error < 1.0:
            raw_score = min(raw_score, 0.15)

        composite_score = round(min(1.0, max(0.0, raw_score)), 3)

        # Categorize Risk Level
        if composite_score >= 0.80:
            level = "CRITICAL"
        elif composite_score >= 0.60:
            level = "HIGH"
        elif composite_score >= 0.35:
            level = "ELEVATED"
        else:
            level = "LOW"

        return {
            "risk_score": composite_score,
            "risk_level": level,
            "components": {
                "ml_threat_impact": round(ml_comp, 3),
                "zero_day_anomaly_impact": round(zd_comp, 3),
                "asset_criticality": round(asset_crit, 3),
                "cti_reputation_impact": round(cti_comp, 3),
                "temporal_factor": round(time_comp, 3)
            },
            "target_asset": ASSET_INVENTORY.get(dst_ip or "", {"name": dst_ip or "external", "role": "Unregistered Node", "zone": "General"})
        }


risk_engine = RiskScorer()
