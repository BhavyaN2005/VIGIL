"""
VIGIL: Explainable AI (XAI) & SHAP Attribution Engine
Wraps shap.TreeExplainer and FlowAutoencoder Zero-Day Anomaly Detection
to provide real-time, defense-grade local feature attributions for SOC analysts.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import shap
import torch

from ml.config import ARTIFACTS_DIR, TARGET_CLASSES
from ml.models.autoencoder_net import FlowAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VIGIL-Explain")


class VigilExplainer:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self._load_artifacts()
        self._init_shap_explainer()

    def _load_artifacts(self):
        """Loads serialized models, scaler, threshold, and feature metadata."""
        logger.info(f"Loading VIGIL artifacts from: {self.artifacts_dir}")
        
        # Load best model or fall back to RF/XGB
        if (self.artifacts_dir / "best_model.joblib").exists():
            self.model = joblib.load(self.artifacts_dir / "best_model.joblib")
        elif (self.artifacts_dir / "xgb_model.joblib").exists():
            self.model = joblib.load(self.artifacts_dir / "xgb_model.joblib")
        else:
            self.model = joblib.load(self.artifacts_dir / "rf_model.joblib")

        self.scaler = joblib.load(self.artifacts_dir / "scaler.joblib")
        
        with open(self.artifacts_dir / "feature_list.json", "r") as f:
            self.feature_names = json.load(f)

        with open(self.artifacts_dir / "label_map.json", "r") as f:
            raw_map = json.load(f)
            self.label_map = {int(k): v for k, v in raw_map.items()}

        # Load Autoencoder for Zero-Day anomaly scoring
        ae_path = self.artifacts_dir / "autoencoder.pt"
        self.autoencoder = None
        self.ae_threshold = 0.05
        
        if ae_path.exists():
            self.autoencoder = FlowAutoencoder(input_dim=len(self.feature_names))
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=torch.device("cpu")))
            self.autoencoder.eval()
            
            thresh_path = self.artifacts_dir / "autoencoder_threshold.json"
            if thresh_path.exists():
                with open(thresh_path, "r") as f:
                    self.ae_threshold = json.load(f).get("threshold_tau", 0.05)

    def _init_shap_explainer(self):
        """Initializes TreeExplainer once to enable sub-millisecond local explanations."""
        logger.info("Initializing TreeExplainer for instantaneous SHAP attribution...")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_flow(
        self,
        features: Union[np.ndarray, List[float], Dict[str, float]],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Explains a single network flow vector or feature dictionary.
        
        Args:
            features: 1D array of length 78 (raw or scaled), or dict of {feature_name: raw_value}
            top_k: Number of top contributing features to return (default: 8)
            
        Returns:
            Structured explanation dictionary with predicted class, class probabilities,
            zero-day anomaly status, and top contributing features.
        """
        # Convert dict or list to numpy array
        if isinstance(features, dict):
            raw_vector = np.array([features.get(f, 0.0) for f in self.feature_names], dtype=np.float64)
            scaled_vector = self.scaler.transform(raw_vector.reshape(1, -1))[0]
        elif isinstance(features, list):
            raw_vector = np.array(features, dtype=np.float64)
            scaled_vector = self.scaler.transform(raw_vector.reshape(1, -1))[0]
        else:
            # Assume raw numpy vector
            raw_vector = features.astype(np.float64)
            scaled_vector = self.scaler.transform(raw_vector.reshape(1, -1))[0]

        X_scaled_2d = scaled_vector.reshape(1, -1)

        # 1. Model Prediction & Probabilities
        probs = self.model.predict_proba(X_scaled_2d)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.label_map.get(pred_idx, TARGET_CLASSES[pred_idx])

        # 2. Zero-Day Anomaly Detection via Autoencoder
        zero_day_info = {
            "reconstruction_error": 0.0,
            "threshold_tau": round(self.ae_threshold, 5),
            "is_zero_day_anomaly": False,
        }
        if self.autoencoder is not None:
            with torch.no_grad():
                tensor_in = torch.tensor(X_scaled_2d, dtype=torch.float32)
                recon_err = float(self.autoencoder.compute_reconstruction_error(tensor_in).item())
                zero_day_info["reconstruction_error"] = round(recon_err, 5)
                zero_day_info["is_zero_day_anomaly"] = recon_err > self.ae_threshold

        # 3. SHAP TreeExplainer Local Attribution
        shap_values = self.explainer.shap_values(X_scaled_2d)
        
        # Handle SHAP multi-class output formats:
        # Format A: List of arrays [class_0_array, class_1_array, ...] each of shape (1, features)
        # Format B: 3D array of shape (1, features, n_classes)
        # Format C: 2D array of shape (1, features) for binary classification
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_idx][0]
        elif len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, pred_idx]
        else:
            class_shap = shap_values[0]

        # Base value handling
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[pred_idx])
        else:
            base_value = float(expected_val)

        # 4. Rank Top K Contributing Features by absolute SHAP score
        abs_shap = np.abs(class_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]

        contributions = []
        for idx in top_indices:
            score = float(class_shap[idx])
            contributions.append({
                "feature_name": self.feature_names[idx],
                "raw_value": round(float(raw_vector[idx]), 4),
                "scaled_value": round(float(scaled_vector[idx]), 4),
                "shap_score": round(score, 5),
                "contribution_direction": "positive" if score >= 0 else "negative",
            })

        return {
            "predicted_class": pred_label,
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": {
                self.label_map.get(i, TARGET_CLASSES[i]): round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "zero_day_detection": zero_day_info,
            "base_value": round(base_value, 4),
            "top_contributions": contributions,
        }


def explain_single_flow(raw_features: Union[np.ndarray, List[float], Dict[str, float]], top_k: int = 8) -> Dict[str, Any]:
    """Convenience helper function for single flow explanation."""
    explainer = VigilExplainer()
    return explainer.explain_flow(raw_features, top_k=top_k)


if __name__ == "__main__":
    from ml.preprocess import run_pipeline
    X_train, X_test, y_train, y_test, scaler, features, stats = run_pipeline()
    explainer = VigilExplainer()
    sample_raw = scaler.inverse_transform(X_test[0:1])[0]
    explanation = explainer.explain_flow(sample_raw)
    print("Sample Explanation Output:")
    print(json.dumps(explanation, indent=2))
