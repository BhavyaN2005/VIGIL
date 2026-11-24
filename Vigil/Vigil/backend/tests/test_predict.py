import io
import pytest
import numpy as np
import pandas as pd
from app.core.model_registry import registry
from ml.config import TARGET_CLASSES


@pytest.mark.asyncio
async def test_predict_single_default_model(client):
    # Create benign flow request
    features = {f: 1.0 for f in registry.feature_names}
    features["Destination Port"] = 443.0
    features["Flow Duration"] = 5000.0

    payload = {
        "features": features,
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.5",
        "dst_port": 443,
        "protocol": "TCP",
    }

    response = await client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in TARGET_CLASSES
    assert data["model_used"] == "XGBoost"
    assert "class_probabilities" in data
    assert "zero_day_detection" in data
    assert "reconstruction_error" in data["zero_day_detection"]
    assert "top_contributions" in data
    assert len(data["top_contributions"]) == 8


@pytest.mark.asyncio
async def test_predict_single_model_switching(client):
    features = {f: 5.0 for f in registry.feature_names}
    payload = {"features": features}

    # 1. Random Forest
    rf_resp = await client.post("/predict/single?model=rf", json=payload)
    assert rf_resp.status_code == 200
    assert rf_resp.json()["model_used"] == "Random Forest"

    # 2. FlowLSTM
    lstm_resp = await client.post("/predict/single?model=lstm", json=payload)
    assert lstm_resp.status_code == 200
    assert lstm_resp.json()["model_used"] == "FlowLSTM"


@pytest.mark.asyncio
async def test_predict_single_threat_logging(client):
    # Create DoS attack flow
    features = {f: 50.0 for f in registry.feature_names}
    features["Destination Port"] = 80.0
    features["Flow Duration"] = 5000000.0
    features["Total Fwd Packets"] = 1500.0
    features["Total Backward Packets"] = 1.0
    features["Flow Bytes/s"] = 150000.0

    payload = {
        "features": features,
        "src_ip": "10.0.0.99",
        "dst_ip": "172.16.0.1",
        "dst_port": 80,
    }

    response = await client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # If classified as attack or zero-day, alert must be logged
    if data["predicted_class"] != "Benign" or data["zero_day_detection"]["is_zero_day_anomaly"]:
        assert data["alert_logged"] is True
        assert data["alert_id"] is not None


@pytest.mark.asyncio
async def test_predict_batch_csv_upload(client):
    # Create small valid in-memory CSV
    rows = []
    for _ in range(15):
        row = {f: np.random.uniform(1.0, 50.0) for f in registry.feature_names}
        row["Destination Port"] = 80
        rows.append(row)
    df = pd.DataFrame(rows)

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {"file": ("test_capture.csv", csv_buffer, "text/csv")}
    response = await client.post("/predict/batch?model=xgb", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_flows_processed"] == 15
    assert "attack_summary" in data
    assert "inference_latency_ms" in data
    assert len(data["sample_flows"]) == 15


@pytest.mark.asyncio
async def test_predict_batch_invalid_file(client):
    # Non-CSV upload
    files = {"file": ("payload.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    response = await client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_batch_empty_csv(client):
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = await client.post("/predict/batch", files=files)
    assert response.status_code == 400
