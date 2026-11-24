import pytest


@pytest.mark.asyncio
async def test_get_forecast_default(client):
    response = await client.get("/forecast")
    assert response.status_code == 200
    data = response.json()
    assert "forecast_horizons" in data
    assert "composite_risk_score" in data
    assert "overall_threat_level" in data
    assert "predicted_attacks" in data
    assert len(data["predicted_attacks"]) > 0
    assert "hourly_projections" in data
    assert len(data["hourly_projections"]) == 24
    assert "kill_chain" in data
    assert "top_risk_assets" in data
    assert "recommended_actions" in data
    assert "shap_explanation" in data

    # Verify confidence intervals format [lower, upper]
    first_pred = data["predicted_attacks"][0]
    assert "probability" in first_pred
    assert "ci" in first_pred
    assert len(first_pred["ci"]) == 2
    assert first_pred["ci"][0] <= first_pred["ci"][1]


@pytest.mark.asyncio
async def test_get_forecast_segment(client):
    response = await client.get("/forecast/dmz")
    assert response.status_code == 200
    data = response.json()
    assert data["segment"] == "dmz"


@pytest.mark.asyncio
async def test_get_horizon_summary(client):
    response = await client.get("/forecast/horizons/summary")
    assert response.status_code == 200
    data = response.json()
    assert "horizons" in data
    assert "T_plus_1h" in data["horizons"]
    assert "T_plus_6h" in data["horizons"]
    assert "T_plus_24h" in data["horizons"]
    assert "top_3_threats" in data
    assert len(data["top_3_threats"]) <= 3


@pytest.mark.asyncio
async def test_simulate_threat_escalation(client):
    payload = {
        "attack_type": "DDoS",
        "intensity_multiplier": 2.5,
        "target_segment": "dmz"
    }
    response = await client.post("/forecast/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "simulation_executed"
    assert "updated_forecast" in data
    ddos_attack = next((a for a in data["updated_forecast"]["predicted_attacks"] if a["type"] == "DDoS"), None)
    assert ddos_attack is not None
    assert ddos_attack["probability"] > 0.5

