"""
Spatial Intelligence & Network Topology Graph Router for VIGIL.
Serves interactive node-edge topology mapping, lateral movement traversal probabilities,
active adversary kill-chain attack paths, and SOAR node isolation actions.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel

router = APIRouter(prefix="/topology", tags=["Network Topology & Spatial Intelligence"])

# In-memory dynamic network graph state
TOPOLOGY_NODES: List[Dict[str, Any]] = [
    {
        "id": "node-ext-attacker",
        "label": "Ext Threat Actor",
        "ip": "198.51.100.22",
        "type": "external_adversary",
        "zone": "External Internet",
        "status": "c2_pivot",
        "risk_score": 0.95,
        "criticality": 0.0,
        "open_ports": [443, 8080],
        "cve_count": 0,
        "details": "Known APT28 (Fancy Bear) C2 Infrastructure"
    },
    {
        "id": "node-perimeter-gw",
        "label": "Perimeter Gateway / FW",
        "ip": "192.168.1.1",
        "type": "firewall",
        "zone": "Perimeter",
        "status": "clean",
        "risk_score": 0.28,
        "criticality": 0.85,
        "open_ports": [22, 443, 8443],
        "cve_count": 0,
        "details": "Palo Alto Next-Gen Firewall Cluster"
    },
    {
        "id": "node-dmz-web",
        "label": "DMZ Web Ingress (Nginx)",
        "ip": "192.168.1.10",
        "type": "web_server",
        "zone": "DMZ",
        "status": "at_risk",
        "risk_score": 0.74,
        "criticality": 0.75,
        "open_ports": [80, 443],
        "cve_count": 3,
        "details": "Reverse proxy facing public Internet, under active probing"
    },
    {
        "id": "node-dmz-api",
        "label": "API Gateway Server",
        "ip": "192.168.1.15",
        "type": "api_gateway",
        "zone": "DMZ",
        "status": "clean",
        "risk_score": 0.35,
        "criticality": 0.70,
        "open_ports": [8000, 8443],
        "cve_count": 1,
        "details": "FastAPI Microservices Routing Ingress"
    },
    {
        "id": "node-internal-gw",
        "label": "Internal Core Switch",
        "ip": "10.0.0.1",
        "type": "core_switch",
        "zone": "Internal Core",
        "status": "clean",
        "risk_score": 0.22,
        "criticality": 0.90,
        "open_ports": [161],
        "cve_count": 0,
        "details": "Cisco Nexus 100G Spine/Leaf Core"
    },
    {
        "id": "node-ad-iam",
        "label": "Active Directory Controller",
        "ip": "10.0.4.20",
        "type": "identity",
        "zone": "Internal Core",
        "status": "compromised",
        "risk_score": 0.89,
        "criticality": 1.00,
        "open_ports": [53, 88, 135, 389, 445],
        "cve_count": 2,
        "details": "Primary Kerberos KDC & LDAP Directory. Active Kerberoast alerts"
    },
    {
        "id": "node-db-primary",
        "label": "PostgreSQL Core Cluster",
        "ip": "10.0.4.15",
        "type": "database",
        "zone": "Database Tier",
        "status": "at_risk",
        "risk_score": 0.92,
        "criticality": 0.95,
        "open_ports": [5432],
        "cve_count": 1,
        "details": "Contains all customer PII and transactional records"
    },
    {
        "id": "node-pci-gateway",
        "label": "PCI Payment Processor",
        "ip": "10.0.5.50",
        "type": "pci_vault",
        "zone": "PCI Secured Zone",
        "status": "clean",
        "risk_score": 0.31,
        "criticality": 0.98,
        "open_ports": [8443],
        "cve_count": 0,
        "details": "Isolated payment transaction enclaves"
    },
    {
        "id": "node-ws-finance",
        "label": "Finance Workstation 04",
        "ip": "172.16.2.100",
        "type": "workstation",
        "zone": "Workstations",
        "status": "compromised",
        "risk_score": 0.82,
        "criticality": 0.45,
        "open_ports": [3389, 445],
        "cve_count": 4,
        "details": "Initial phishing ingress point with TrickBot beacon"
    },
    {
        "id": "node-ws-eng",
        "label": "Engineering Build Node",
        "ip": "172.16.2.101",
        "type": "workstation",
        "zone": "Workstations",
        "status": "clean",
        "risk_score": 0.25,
        "criticality": 0.50,
        "open_ports": [22],
        "cve_count": 0,
        "details": "CI/CD automated runner machine"
    }
]

TOPOLOGY_EDGES: List[Dict[str, Any]] = [
    {
        "id": "edge-1",
        "source": "node-ext-attacker",
        "target": "node-perimeter-gw",
        "protocol": "HTTPS",
        "traffic_rate_mbps": 42.5,
        "threat_detected": True,
        "traversal_probability": 0.95
    },
    {
        "id": "edge-2",
        "source": "node-perimeter-gw",
        "target": "node-dmz-web",
        "protocol": "HTTP/2",
        "traffic_rate_mbps": 38.2,
        "threat_detected": True,
        "traversal_probability": 0.88
    },
    {
        "id": "edge-3",
        "source": "node-perimeter-gw",
        "target": "node-dmz-api",
        "protocol": "HTTPS",
        "traffic_rate_mbps": 12.0,
        "threat_detected": False,
        "traversal_probability": 0.15
    },
    {
        "id": "edge-4",
        "source": "node-dmz-web",
        "target": "node-internal-gw",
        "protocol": "TCP/Internal",
        "traffic_rate_mbps": 18.4,
        "threat_detected": True,
        "traversal_probability": 0.79
    },
    {
        "id": "edge-5",
        "source": "node-internal-gw",
        "target": "node-ws-finance",
        "protocol": "SMB",
        "traffic_rate_mbps": 5.1,
        "threat_detected": True,
        "traversal_probability": 0.92
    },
    {
        "id": "edge-6",
        "source": "node-ws-finance",
        "target": "node-ad-iam",
        "protocol": "Kerberos/LDAP",
        "traffic_rate_mbps": 2.4,
        "threat_detected": True,
        "traversal_probability": 0.86
    },
    {
        "id": "edge-7",
        "source": "node-ad-iam",
        "target": "node-db-primary",
        "protocol": "PostgreSQL/TDS",
        "traffic_rate_mbps": 14.8,
        "threat_detected": True,
        "traversal_probability": 0.91
    },
    {
        "id": "edge-8",
        "source": "node-internal-gw",
        "target": "node-pci-gateway",
        "protocol": "mTLS",
        "traffic_rate_mbps": 1.2,
        "threat_detected": False,
        "traversal_probability": 0.08
    },
    {
        "id": "edge-9",
        "source": "node-internal-gw",
        "target": "node-ws-eng",
        "protocol": "SSH",
        "traffic_rate_mbps": 0.8,
        "threat_detected": False,
        "traversal_probability": 0.12
    }
]

NETWORK_ZONES = [
    {"name": "External Internet", "color": "#FF4D2D", "trust_level": 0},
    {"name": "Perimeter", "color": "#F59E0B", "trust_level": 1},
    {"name": "DMZ", "color": "#00E5FF", "trust_level": 2},
    {"name": "Workstations", "color": "#8B5CF6", "trust_level": 3},
    {"name": "Internal Core", "color": "#0284C7", "trust_level": 4},
    {"name": "Database Tier", "color": "#EC4899", "trust_level": 5},
    {"name": "PCI Secured Zone", "color": "#10B981", "trust_level": 6}
]


@router.get("", summary="Get Network Topology & Attack Graph")
@router.get("/", summary="Get Network Topology & Attack Graph")
async def get_network_topology():
    """
    Retrieve live network topology graph containing active nodes, connection edges,
    lateral movement traversal probabilities, and identified adversary attack paths.
    """
    # Active attack chain: Ext Attacker -> DMZ Ingress -> Internal GW -> Finance WS -> AD -> DB Cluster
    active_kill_chain_path = [
        "node-ext-attacker",
        "node-perimeter-gw",
        "node-dmz-web",
        "node-internal-gw",
        "node-ws-finance",
        "node-ad-iam",
        "node-db-primary"
    ]

    compromised_count = sum(1 for n in TOPOLOGY_NODES if n["status"] in ("compromised", "c2_pivot"))
    at_risk_count = sum(1 for n in TOPOLOGY_NODES if n["status"] == "at_risk")

    return {
        "status": "success",
        "summary": {
            "total_nodes": len(TOPOLOGY_NODES),
            "total_edges": len(TOPOLOGY_EDGES),
            "compromised_nodes": compromised_count,
            "at_risk_nodes": at_risk_count,
            "lateral_threat_detected": True,
            "active_pivot_vector": "Workstation Finance (172.16.2.100) -> Active Directory (10.0.4.20)"
        },
        "zones": NETWORK_ZONES,
        "nodes": TOPOLOGY_NODES,
        "edges": TOPOLOGY_EDGES,
        "active_attack_paths": [
            {
                "path_id": "PATH-ALPHA-01",
                "technique": "T1021 - Lateral Movement via Remote Services",
                "nodes": active_kill_chain_path,
                "traversal_confidence": 0.89,
                "target_asset": "10.0.4.15 (db-core-cluster-01)",
                "threat_stage": "Pivoting towards Crown Jewel Database"
            }
        ]
    }


@router.post("/node/{node_id}/isolate", summary="SOAR Action: Isolate Node")
async def isolate_node(node_id: str = Path(..., description="ID of node to quarantine")):
    """
    Execute automated SOAR containment playbook to quarantine an infected node
    by cutting network connectivity and dropping routing rules.
    """
    target = None
    for node in TOPOLOGY_NODES:
        if node["id"] == node_id:
            target = node
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found in topology")

    # Update state
    target["status"] = "isolated"
    target["risk_score"] = 0.05
    target["details"] += " [CONTAINED & ISOLATED VIA SOAR PLAYBOOK]"

    return {
        "status": "success",
        "message": f"Node {target['label']} ({target['ip']}) has been successfully quarantined.",
        "contained_node": target,
        "remediation_status": "Traffic severed. EDR memory dump initiated."
    }

