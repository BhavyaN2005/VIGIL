import pytest
from app.core.mitre_mapper import get_mitre_mapping, get_kill_chain_progression, ATTACK_TTP_REGISTRY
from app.core.cti_enricher import cti_engine
from app.core.risk_scorer import risk_engine


def test_mitre_mapper_all_attack_classes():
    for attack in ["DoS", "DDoS", "PortScan", "Brute Force", "Web Attack", "Bot", "Lateral Movement", "Data Exfiltration"]:
        info = get_mitre_mapping(attack)
        assert info is not None
        assert "tactic" in info
        assert "technique_id" in info
        assert info["technique_id"].startswith("T")
        assert "mitigation" in info
        assert info["kill_chain_stage"] >= 1


def test_mitre_mapper_case_insensitivity_and_unknown():
    # Case insensitivity
    info = get_mitre_mapping("portscan")
    assert info["technique_id"] == "T1046"

    # Unknown attack fallback
    unknown_info = get_mitre_mapping("SuperZeroDayQuantumExploit")
    assert unknown_info["technique_id"] == "T1203"
    assert "reconstruction loss" in unknown_info["detection_indicators"][0]


def test_kill_chain_progression():
    chain = get_kill_chain_progression("PortScan")
    assert chain["current_stage"] == 1
    assert chain["next_prediction"]["next_stage"] == 2
    assert chain["next_prediction"]["transition_probability"] > 0.5

    # Lateral movement progression
    chain_lat = get_kill_chain_progression("Lateral Movement")
    assert chain_lat["current_stage"] == 6
    assert chain_lat["next_prediction"]["next_stage"] == 7
    assert "Exfiltration" in chain_lat["next_prediction"]["next_tactic"]


def test_cti_enricher():
    # Known malicious IP
    known = cti_engine.enrich_ip("192.168.1.105")
    assert known["reputation_score"] >= 0.8
    assert known["actor"] == "Wizard Spider (UNC1878)"
    assert known["is_bulletproof"] is True

    # Internal private IP
    internal = cti_engine.enrich_ip("10.0.0.5")
    assert internal["is_malicious"] is False

    # External IP
    external = cti_engine.enrich_ip("8.8.8.8")
    assert "reputation_score" in external

    # Flow enrichment
    flow = {"src_ip": "192.168.1.105", "dst_ip": "10.0.4.15", "predicted_class": "Bot"}
    enriched = cti_engine.enrich_flow(flow)
    assert "cti" in enriched
    assert enriched["cti"]["is_flagged_by_feeds"] is True


def test_risk_scorer():
    # Benign flow
    benign_risk = risk_engine.compute_risk(
        predicted_class="Benign",
        confidence=0.99,
        severity="NONE",
        is_zero_day=False,
        reconstruction_error=0.2
    )
    assert benign_risk["risk_score"] <= 0.20
    assert benign_risk["risk_level"] == "LOW"

    # Critical threat targeted at high-value asset
    crit_risk = risk_engine.compute_risk(
        predicted_class="Lateral Movement",
        confidence=0.95,
        severity="CRITICAL",
        is_zero_day=True,
        reconstruction_error=2.45,
        dst_ip="10.0.4.20",  # Active Directory Controller
        cti_score=0.90
    )
    assert crit_risk["risk_score"] >= 0.80
    assert crit_risk["risk_level"] == "CRITICAL"
    assert crit_risk["components"]["asset_criticality"] == 1.0

