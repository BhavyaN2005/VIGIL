import pytest
from app.services.stream_service import stream_service


@pytest.mark.asyncio
async def test_stream_inject_endpoint(client):
    response = await client.post("/stream/inject?attack=DDoS&count=12")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["attack_class"] == "DDoS"
    assert data["count_injected"] == 12
    assert data["total_queued"] >= 12


@pytest.mark.asyncio
async def test_metrics_models_endpoint(client):
    response = await client.get("/metrics/models")
    assert response.status_code == 200
    data = response.json()
    assert "model_comparison" in data
    assert len(data["model_comparison"]) == 4
    assert "top_15_features" in data
