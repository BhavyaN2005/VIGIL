"""
Unit Tests for VIGIL SHAP Explainability Engine (ml/explain.py)
Validates local feature attributions, ranking, probabilities, and zero-day scoring.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from ml.config import TARGET_CLASSES
from ml.explain import VigilExplainer


@pytest.fixture(scope="module")
def explainer():
    return VigilExplainer()


def test_explainer_initialization(explainer):
    assert explainer.model is not None
    assert explainer.scaler is not None
    assert explainer.explainer is not None
    assert len(explainer.feature_names) == 78


def test_explain_flow_with_vector(explainer):
    # Create mock raw flow
    raw_vector = np.random.uniform(0.0, 100.0, size=len(explainer.feature_names))
    raw_vector[0] = 80  # Destination Port
    raw_vector[1] = 50000  # Flow Duration

    result = explainer.explain_flow(raw_vector, top_k=8)

    # 1. Structure assertions
    assert "predicted_class" in result
    assert result["predicted_class"] in TARGET_CLASSES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    # 2. Probability distribution
    assert "class_probabilities" in result
    probs = result["class_probabilities"]
    assert len(probs) == len(TARGET_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-2

    # 3. Zero-day anomaly detection
    assert "zero_day_detection" in result
    zd = result["zero_day_detection"]
    assert "reconstruction_error" in zd
    assert "threshold_tau" in zd
    assert "is_zero_day_anomaly" in zd
    assert isinstance(zd["is_zero_day_anomaly"], (bool, np.bool_))

    # 4. Top contributions (exactly 8)
    assert "top_contributions" in result
    top_8 = result["top_contributions"]
    assert len(top_8) == 8

    # Check sorting order: absolute SHAP scores must be descending
    scores = [abs(item["shap_score"]) for item in top_8]
    assert scores == sorted(scores, reverse=True), "Top contributions are not sorted by absolute SHAP score"

    # Check item schema
    for item in top_8:
        assert "feature_name" in item
        assert item["feature_name"] in explainer.feature_names
        assert "raw_value" in item
        assert "scaled_value" in item
        assert "shap_score" in item
        assert item["contribution_direction"] in ["positive", "negative"]


def test_explain_flow_with_dictionary(explainer):
    feature_dict = {feat: np.random.uniform(1.0, 50.0) for feat in explainer.feature_names}
    result = explainer.explain_flow(feature_dict, top_k=5)

    assert result["predicted_class"] in TARGET_CLASSES
    assert len(result["top_contributions"]) == 5
