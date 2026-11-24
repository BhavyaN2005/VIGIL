import pytest
from app.core.database import AsyncSessionLocal
from app.services.alert_service import AlertService


@pytest.mark.asyncio
async def test_alerts_lifecycle_and_filtering(client):
    # 1. Seed two test alerts via AlertService
    async with AsyncSessionLocal() as db:
        alert1 = await AlertService.create_alert(
            db=db,
            src_ip="192.168.1.55",
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="TCP",
            attack_class="DoS",
            severity="High",
            confidence=0.92,
            is_zero_day=False,
            reconstruction_error=0.45,
            top_contributing_feature="Flow Bytes/s",
            model_used="XGBoost",
        )
        alert2 = await AlertService.create_alert(
            db=db,
            src_ip="192.168.1.66",
            dst_ip="10.0.0.2",
            dst_port=4444,
            protocol="TCP",
            attack_class="Bot",
            severity="Critical",
            confidence=0.88,
            is_zero_day=True,
            reconstruction_error=2.45,
            top_contributing_feature="Destination Port",
            model_used="XGBoost",
        )
        alert1_id = alert1.id
        alert2_id = alert2.id

    # 2. Query all alerts
    resp = await client.get("/alerts?page=1&page_size=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    assert len(data["items"]) >= 2

    # 3. Filter by severity Critical
    resp_crit = await client.get("/alerts?severity=Critical")
    assert resp_crit.status_code == 200
    crit_items = resp_crit.json()["items"]
    assert all(item["severity"] == "Critical" for item in crit_items)

    # 4. Filter by is_zero_day=True
    resp_zd = await client.get("/alerts?is_zero_day=true")
    assert resp_zd.status_code == 200
    zd_items = resp_zd.json()["items"]
    assert all(item["is_zero_day"] is True for item in zd_items)

    # 5. Patch alert1 status to 'Investigating'
    patch_resp = await client.patch(
        f"/alerts/{alert1_id}",
        json={"status": "Investigating", "analyst_notes": "Triage initiated by Tier 2 SOC."},
    )
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["status"] == "Investigating"
    assert patched["analyst_notes"] == "Triage initiated by Tier 2 SOC."

    # 6. Patch alert1 status to 'Resolved'
    patch_resp2 = await client.patch(
        f"/alerts/{alert1_id}",
        json={"status": "Resolved", "analyst_notes": "Firewall IP block rule deployed."},
    )
    assert patch_resp2.status_code == 200
    assert patch_resp2.json()["status"] == "Resolved"

    # 7. Non-existent alert patch
    not_found = await client.patch("/alerts/99999", json={"status": "Resolved"})
    assert not_found.status_code == 404
