import random
from typing import Tuple

def determine_severity(
    attack_class: str,
    confidence: float,
    is_zero_day: bool = False,
    reconstruction_error: float = 0.0,
    threshold_tau: float = 1.627,
) -> str:
    """
    Computes standard SOC severity rating based on threat topology and confidence.
    """
    if attack_class == "Benign" and not is_zero_day:
        return "Low"

    # Extreme volumetric or remote command control
    if attack_class in ["DDoS", "Bot"] and confidence >= 0.65:
        return "Critical"
        
    # High-impact zero-day anomaly
    if is_zero_day and reconstruction_error > (1.8 * threshold_tau):
        return "Critical"

    # Severe targeted attacks or confirmed zero-day
    if attack_class in ["DoS", "Brute Force", "Web Attack"] and confidence >= 0.50:
        return "High"
    if is_zero_day:
        return "High"

    # Reconnaissance / Scans
    if attack_class == "PortScan":
        return "Medium"

    if confidence >= 0.40:
        return "Medium"

    return "Low"


def generate_flow_metadata(dst_port: int = 80) -> Tuple[str, str, str]:
    """Generates realistic internal/external IP pairs and protocol for display."""
    src_subnets = ["192.168.1.", "10.0.0.", "172.16.1."]
    dst_subnets = ["172.16.0.", "10.200.1."]
    
    src_ip = f"{random.choice(src_subnets)}{random.randint(10, 250)}"
    dst_ip = f"{random.choice(dst_subnets)}{random.randint(2, 50)}"
    
    if dst_port in [53, 67, 68, 123, 161]:
        protocol = "UDP"
    else:
        protocol = "TCP"
        
    return src_ip, dst_ip, protocol
