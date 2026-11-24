import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "XGBoost" in data["loaded_models"]
    assert "Random Forest" in data["loaded_models"]
    assert "FlowLSTM" in data["loaded_models"]
    assert "FlowAutoencoder" in data["loaded_models"]
    assert data["feature_dimensions"] == 78
    assert data["memory_mb"] > 0
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/health" in data["endpoints"]
    assert "/predict/single" in data["endpoints"]
