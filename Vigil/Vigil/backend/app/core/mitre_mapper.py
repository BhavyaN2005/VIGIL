"""
MITRE ATT&CK Mapping & Cyber Kill Chain Progression Engine for VIGIL.
Maps detected network traffic patterns and predicted threats to adversary Tactics,
Techniques, and Procedures (TTPs), MITRE IDs, and next-stage progression probabilities.
"""

from typing import Dict, List, Any, Optional

# MITRE ATT&CK Knowledge Base for Network Attacks
MITRE_TACTIC_CHAIN = [
    {"stage": 1, "tactic": "Reconnaissance", "id": "TA0043", "description": "Gathering victim network topology and active services"},
    {"stage": 2, "tactic": "Initial Access", "id": "TA0001", "description": "Attempting ingress via public exploits or credentials"},
    {"stage": 3, "tactic": "Execution", "id": "TA0002", "description": "Executing malicious payloads or scripts on target"},
    {"stage": 4, "tactic": "Persistence / Privilege Escalation", "id": "TA0003", "description": "Establishing backdoor and elevating privileges"},
    {"stage": 5, "tactic": "Credential Access", "id": "TA0006", "description": "Harvesting credentials or brute-forcing authentication"},
    {"stage": 6, "tactic": "Lateral Movement", "id": "TA0008", "description": "Pivoting across internal subnets and adjacent systems"},
    {"stage": 7, "tactic": "Command & Control / Impact / Exfiltration", "id": "TA0011", "description": "C2 beaconing, data exfiltration, or denial of service"}
]

ATTACK_TTP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "PortScan": {
        "tactic": "Reconnaissance",
        "tactic_id": "TA0043",
        "technique": "Network Service Discovery",
        "technique_id": "T1046",
        "sub_techniques": ["T1046.001 - Port Scanning", "T1018 - Remote System Discovery"],
        "kill_chain_stage": 1,
        "severity": "MEDIUM",
        "description": "Adversary probes active IP addresses, open ports, and running network services to identify target vulnerabilities.",
        "detection_indicators": ["High SYN-to-ACK ratio", "Rapid sequential port hits", "Low byte count per flow"],
        "mitigation": "Configure rate limiting on perimeter firewalls, deploy port-knocking or deny-by-default ACLs, drop unrouted packets.",
        "likely_next_tactics": ["Initial Access (T1190)", "Credential Access (T1110)"]
    },
    "Brute Force": {
        "tactic": "Credential Access",
        "tactic_id": "TA0006",
        "technique": "Brute Force",
        "technique_id": "T1110",
        "sub_techniques": ["T1110.001 - Password Guessing", "T1110.003 - Password Spraying"],
        "kill_chain_stage": 5,
        "severity": "HIGH",
        "description": "Systematic attempt to gain authentication tokens or passwords by submitting high-frequency permutations against SSH, RDP, or HTTP login endpoints.",
        "detection_indicators": ["Consecutive TCP resets on port 22/3389", "Elevated failed authentication count in 5m window", "Uniform packet length"],
        "mitigation": "Enforce Multi-Factor Authentication (MFA), dynamic IP account lockout after 5 failed attempts, migrate SSH to certificate-based keys.",
        "likely_next_tactics": ["Lateral Movement (T1021)", "Execution (T1059)"]
    },
    "DoS": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Endpoint Denial of Service",
        "technique_id": "T1499",
        "sub_techniques": ["T1499.002 - Service Exhaustion Flood", "T1499.003 - Application Exhaustion"],
        "kill_chain_stage": 7,
        "severity": "HIGH",
        "description": "Flooding service queues, buffer pools, or CPU pipelines of a single host to cause degradation or complete service unresponsiveness.",
        "detection_indicators": ["Anomalous forward packet count", "High IAT (inter-arrival time) variance", "Incomplete HTTP request streams"],
        "mitigation": "Implement upstream TCP SYN-cookies, connection limits per source IP, ingress traffic scrubbing.",
        "likely_next_tactics": ["Defense Evasion (T1562)", "Exfiltration Over C2 (T1041)"]
    },
    "DDoS": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "sub_techniques": ["T1498.001 - Direct Network Flood", "T1498.002 - Reflection Amplification"],
        "kill_chain_stage": 7,
        "severity": "CRITICAL",
        "description": "Distributed denial of service leveraging botnet agents or amplification vectors (DNS, NTP, SSDP) to overwhelm infrastructure bandwidth.",
        "detection_indicators": ["Symmetric multi-source gigabit flow surges", "BGP route saturation", "Massive packet per second (PPS) spike"],
        "mitigation": "Trigger BGP FlowSpec automated blackholing, engage cloud DDoS mitigation scrubbing centers, rate-limit UDP amplification ports.",
        "likely_next_tactics": ["Secondary diversion for Data Exfiltration (T1048)"]
    },
    "Web Attack": {
        "tactic": "Initial Access",
        "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application",
        "technique_id": "T1190",
        "sub_techniques": ["T1059.007 - JavaScript Exploitation", "T1505.003 - Web Shell Deployment"],
        "kill_chain_stage": 2,
        "severity": "CRITICAL",
        "description": "Exploitation of web application vulnerabilities (SQL Injection, Cross-Site Scripting, RCE, Deserialization) to compromise perimeter services.",
        "detection_indicators": ["Payload entropy anomalies", "SQL syntax patterns in URI/body parameters", "HTTP 500 error bursts"],
        "mitigation": "Deploy Web Application Firewall (WAF) OWASP Core Rule Set, input sanitization, parameterized database queries, containerized micro-segmentation.",
        "likely_next_tactics": ["Privilege Escalation (T1068)", "Lateral Movement (T1021)"]
    },
    "Bot": {
        "tactic": "Command and Control",
        "tactic_id": "TA0011",
        "technique": "Application Layer Protocol",
        "technique_id": "T1071",
        "sub_techniques": ["T1071.001 - Web Protocols", "T1071.004 - DNS Tunneling / Beaconing"],
        "kill_chain_stage": 7,
        "severity": "HIGH",
        "description": "Compromised internal host communicating with adversary Command and Control (C2) server via periodic beacons, DNS tunnels, or encrypted websockets.",
        "detection_indicators": ["Periodic heartbeat IAT intervals", "High entropy subdomain DNS queries", "Outbound connections to newly registered domains"],
        "mitigation": "Sinkhole C2 domain names via DNS RPZ, block known malicious ASN routes, isolate compromised endpoint via EDR agent.",
        "likely_next_tactics": ["Lateral Movement (T1021)", "Data Exfiltration (T1048)"]
    },
    "Lateral Movement": {
        "tactic": "Lateral Movement",
        "tactic_id": "TA0008",
        "technique": "Remote Services",
        "technique_id": "T1021",
        "sub_techniques": ["T1021.001 - Remote Desktop Protocol", "T1021.002 - SMB/Windows Admin Shares"],
        "kill_chain_stage": 6,
        "severity": "CRITICAL",
        "description": "Adversary moves through the internal environment by pivoting from an initial compromised system to higher-value database or domain controller assets.",
        "detection_indicators": ["Internal East-West traffic spikes on 445/3389", "Pass-the-hash authentication anomalies", "Unusual service ticket requests"],
        "mitigation": "Enforce zero-trust network segmentation (microsegmentation), restrict SMB traffic between workstations, deploy LAPS for local admin password isolation.",
        "likely_next_tactics": ["Exfiltration (T1048)", "Impact - Ransomware (T1486)"]
    },
    "Data Exfiltration": {
        "tactic": "Exfiltration",
        "tactic_id": "TA0010",
        "technique": "Exfiltration Over Alternative Protocol",
        "technique_id": "T1048",
        "sub_techniques": ["T1048.003 - Exfiltration Over Unencrypted Non-C2 Protocol", "T1567 - Exfiltration Over Web Service"],
        "kill_chain_stage": 7,
        "severity": "CRITICAL",
        "description": "Unauthorized transfer of sensitive internal database tables, source code, or credentials to external adversary staging buckets.",
        "detection_indicators": ["Massive outbound forward byte counts", "Protracted flow duration to unfamiliar external IPs", "Encrypted archives in egress stream"],
        "mitigation": "Impose strict egress firewall filtering, DLP content inspection, quota alerts on outbound multi-megabyte transfers to non-approved destinations.",
        "likely_next_tactics": ["Impact (T1486 - Data Encrypted for Impact)"]
    },
    "Benign": {
        "tactic": "None",
        "tactic_id": "TA0000",
        "technique": "Normal Traffic",
        "technique_id": "T0000",
        "sub_techniques": [],
        "kill_chain_stage": 0,
        "severity": "NONE",
        "description": "Standard authorized enterprise network communications conforming to baseline behavioral profiles.",
        "detection_indicators": ["Standard packet sizes", "Regular IAT distributions", "Whitelisted destination IPs"],
        "mitigation": "No action required. Continuously update baseline telemetry.",
        "likely_next_tactics": []
    }
}


def get_mitre_mapping(attack_class: str) -> Dict[str, Any]:
    """
    Retrieve MITRE ATT&CK TTP mapping for a given attack classification.
    """
    # Normalize string
    normalized = attack_class.strip()
    if normalized in ATTACK_TTP_REGISTRY:
        return ATTACK_TTP_REGISTRY[normalized]

    for key, data in ATTACK_TTP_REGISTRY.items():
        if key.lower() == normalized.lower():
            return data

    # Default fallback for unknown or zero-day anomalies
    return {
        "tactic": "Defense Evasion / Unknown",
        "tactic_id": "TA0005",
        "technique": "Exploitation of Novel Zero-Day Anomaly",
        "technique_id": "T1203",
        "sub_techniques": ["T1203 - Exploitation for Client Execution"],
        "kill_chain_stage": 3,
        "severity": "HIGH",
        "description": "Novel behavioral deviation exceeding autoencoder reconstruction error threshold. Potential zero-day adversary activity.",
        "detection_indicators": ["High reconstruction loss (>1.627)", "Out-of-distribution feature combinations"],
        "mitigation": "Quarantine source host, capture full PCAP dump for forensic reverse-engineering, inspect process trees.",
        "likely_next_tactics": ["Command and Control (T1071)", "Lateral Movement (T1021)"]
    }


def get_kill_chain_progression(current_attack: str) -> Dict[str, Any]:
    """
    Calculate attack stage progression along the 7-stage Cyber Kill Chain,
    estimating next adversary tactics and transition probabilities.
    """
    mapping = get_mitre_mapping(current_attack)
    stage = mapping.get("kill_chain_stage", 1)

    # Next likely stage and tactics
    next_stage_info: Dict[str, Any] = {}
    if stage == 0:
        next_stage_info = {
            "next_stage": 1,
            "next_tactic": "Reconnaissance (T1046)",
            "transition_probability": 0.05,
            "time_to_impact_minutes": 1440
        }
    elif stage == 1:
        next_stage_info = {
            "next_stage": 2,
            "next_tactic": "Initial Access / Public Exploit (T1190)",
            "transition_probability": 0.78,
            "time_to_impact_minutes": 180
        }
    elif stage == 2:
        next_stage_info = {
            "next_stage": 5,
            "next_tactic": "Credential Harvesting / Brute Force (T1110)",
            "transition_probability": 0.84,
            "time_to_impact_minutes": 65
        }
    elif stage == 5:
        next_stage_info = {
            "next_stage": 6,
            "next_tactic": "Lateral Movement across Subnets (T1021)",
            "transition_probability": 0.91,
            "time_to_impact_minutes": 30
        }
    elif stage == 6:
        next_stage_info = {
            "next_stage": 7,
            "next_tactic": "Data Exfiltration & Ransomware Impact (T1048 / T1486)",
            "transition_probability": 0.95,
            "time_to_impact_minutes": 15
        }
    else:  # stage 7
        next_stage_info = {
            "next_stage": 7,
            "next_tactic": "Persistent C2 Beaconing & Secondary Impact",
            "transition_probability": 0.65,
            "time_to_impact_minutes": 5
        }

    return {
        "current_stage": stage,
        "current_tactic": mapping.get("tactic"),
        "technique_id": mapping.get("technique_id"),
        "technique_name": mapping.get("technique"),
        "chain_stages": MITRE_TACTIC_CHAIN,
        "next_prediction": next_stage_info
    }

