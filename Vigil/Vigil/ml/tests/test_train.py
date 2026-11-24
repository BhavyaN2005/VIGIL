"""
Unit Tests for VIGIL Multi-Model Training & Benchmark (ml/train.py)
Validates model artifact serialization, metrics schema, and evaluation outputs.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pytest
import torch

from ml.config import ARTIFACTS_DIR, TARGET_CLASSES


def test_exported_artifacts_exist():
    required_files = [
        "best_model.joblib",
        "rf_model.joblib",
        "xgb_model.joblib",
        "lstm_model.pt",
        "autoencoder.pt",
        "autoencoder_threshold.json",
        "scaler.joblib",
        "feature_list.json",
        "label_map.json",
        "metrics.json",
    ]
    for fname in required_files:
        p = ARTIFACTS_DIR / fname
        assert p.exists(), f"Missing required artifact: {fname}"


def test_models_deserialization():
    rf = joblib.load(ARTIFACTS_DIR / "rf_model.joblib")
    assert hasattr(rf, "predict"), "Random Forest failed to deserialize"

    xgb = joblib.load(ARTIFACTS_DIR / "xgb_model.joblib")
    assert hasattr(xgb, "predict"), "XGBoost failed to deserialize"

    best = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    assert hasattr(best, "predict"), "Best model failed to deserialize"

    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    assert hasattr(scaler, "transform"), "Scaler failed to deserialize"

    lstm_weights = torch.load(ARTIFACTS_DIR / "lstm_model.pt", map_location="cpu")
    assert "lstm.weight_ih_l0" in lstm_weights, "LSTM state dict invalid"

    ae_weights = torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location="cpu")
    assert "encoder.0.weight" in ae_weights, "Autoencoder state dict invalid"


def test_metrics_json_schema():
    with open(ARTIFACTS_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    assert "model_comparison" in metrics
    models = metrics["model_comparison"]
    assert len(models) == 4, f"Expected 4 models evaluated, got {len(models)}"

    model_names = [m["model_name"] for m in models]
    assert "Random Forest" in model_names
    assert "XGBoost" in model_names
    assert "FlowLSTM" in model_names
    assert "FlowAutoencoder" in model_names

    # Check supervised metric bounds
    for m in models[:3]:
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["latency_ms_per_1k"] > 0
        assert len(m["confusion_matrix"]) == len(TARGET_CLASSES)

    # Check Autoencoder metrics
    ae = next(m for m in models if m["model_name"] == "FlowAutoencoder")
    assert "threshold_tau" in ae
    assert ae["threshold_tau"] > 0
    assert 0.0 <= ae["anomaly_auroc"] <= 1.0

    # Check Top 15 Features
    assert "top_15_features" in metrics
    assert len(metrics["top_15_features"]) <= 15
    for feat in metrics["top_15_features"]:
        assert "feature" in feat
        assert "combined_importance" in feat
