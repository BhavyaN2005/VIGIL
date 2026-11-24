import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import shap

from app.config import settings
from ml.config import TARGET_CLASSES

try:
    import torch
    from ml.models.autoencoder_net import FlowAutoencoder
    from ml.models.lstm_net import FlowLSTM
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    FlowAutoencoder = None
    FlowLSTM = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("VIGIL-Registry")


class ModelRegistry:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.ARTIFACTS_DIR
        self.models: Dict[str, Any] = {}
        self.explainers: Dict[str, Any] = {}
        self.scaler = None
        self.feature_names: List[str] = []
        self.label_map: Dict[int, str] = {}
        self.metrics_data: Dict[str, Any] = {}
        self.autoencoder = None
        self.ae_threshold: float = 1.627
        self.start_time: float = time.time()
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Loads all serialized models, scaler, metrics, and SHAP explainers."""
        logger.info(f"Loading VIGIL model registry from: {self.artifacts_dir}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Feature list
        feat_path = self.artifacts_dir / "feature_list.json"
        if feat_path.exists():
            with open(feat_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            logger.warning("feature_list.json missing. Using default 78 CIC-IDS2017 features.")
            from data.generate_sample import FEATURE_NAMES
            self.feature_names = [f.strip() for f in FEATURE_NAMES]

        # 2. Label Map
        label_path = self.artifacts_dir / "label_map.json"
        if label_path.exists():
            with open(label_path, "r") as f:
                raw_map = json.load(f)
                self.label_map = {int(k): v for k, v in raw_map.items()}
        else:
            self.label_map = {i: name for i, name in enumerate(TARGET_CLASSES)}

        # 3. Scaler
        scaler_path = self.artifacts_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            logger.warning("scaler.joblib missing. Fitting fallback scaler.")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(np.zeros((10, len(self.feature_names))))

        # 4. Supervised Classifiers: XGBoost, Random Forest
        xgb_path = self.artifacts_dir / "xgb_model.joblib"
        if xgb_path.exists():
            self.models["xgb"] = joblib.load(xgb_path)
            logger.info("Loaded XGBoost Classifier.")
        elif (self.artifacts_dir / "best_model.joblib").exists():
            self.models["xgb"] = joblib.load(self.artifacts_dir / "best_model.joblib")

        rf_path = self.artifacts_dir / "rf_model.joblib"
        if rf_path.exists():
            self.models["rf"] = joblib.load(rf_path)
            logger.info("Loaded Random Forest Classifier.")

        # Fallback for supervised if missing
        if "xgb" not in self.models and "rf" not in self.models:
            logger.warning("No tree model artifacts found. Initializing fallback Random Forest.")
            from sklearn.ensemble import RandomForestClassifier
            fallback_rf = RandomForestClassifier(n_estimators=10, random_state=42)
            fallback_rf.fit(np.zeros((14, len(self.feature_names))), list(range(7)) * 2)
            self.models["rf"] = fallback_rf
            self.models["xgb"] = fallback_rf

        # 5. FlowLSTM Neural Network (PyTorch)
        if TORCH_AVAILABLE and FlowLSTM is not None:
            lstm_path = self.artifacts_dir / "lstm_model.pt"
            self.lstm = FlowLSTM(input_dim=len(self.feature_names), num_classes=len(TARGET_CLASSES))
            if lstm_path.exists():
                try:
                    self.lstm.load_state_dict(torch.load(lstm_path, map_location="cpu"))
                    logger.info("Loaded FlowLSTM Neural Network weights.")
                except Exception as e:
                    logger.warning(f"Could not load LSTM weights ({e}). Initialized fresh FlowLSTM.")
            self.lstm.eval()
            self.models["lstm"] = self.lstm
        else:
            logger.info("FlowLSTM awaiting PyTorch. Tree models active.")
            self.models["lstm"] = self.models.get("xgb") or self.models.get("rf")

        # 6. FlowAutoencoder (Zero-Day Anomaly Detector)
        if TORCH_AVAILABLE and FlowAutoencoder is not None:
            ae_path = self.artifacts_dir / "autoencoder.pt"
            self.autoencoder = FlowAutoencoder(input_dim=len(self.feature_names))
            if ae_path.exists():
                try:
                    self.autoencoder.load_state_dict(torch.load(ae_path, map_location="cpu"))
                    logger.info("Loaded FlowAutoencoder Zero-Day Anomaly Detector weights.")
                except Exception as e:
                    logger.warning(f"Could not load Autoencoder weights ({e}).")
            self.autoencoder.eval()
        else:
            self.autoencoder = None

        # Threshold
        thresh_path = self.artifacts_dir / "autoencoder_threshold.json"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                self.ae_threshold = float(json.load(f).get("threshold_tau", 1.627))

        # 7. Metrics JSON
        metrics_path = self.artifacts_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics_data = json.load(f)
        else:
            self.metrics_data = {"status": "Metrics not yet generated."}

        # 8. SHAP TreeExplainer Caching
        primary_tree_model = self.models.get("xgb") or self.models.get("rf")
        if primary_tree_model:
            try:
                logger.info("Initializing cached TreeExplainer for instantaneous local XAI...")
                self.explainers["xgb"] = shap.TreeExplainer(primary_tree_model)
                if "rf" in self.models and self.models["rf"] != primary_tree_model:
                    self.explainers["rf"] = shap.TreeExplainer(self.models["rf"])
            except Exception as e:
                logger.warning(f"Failed to initialize TreeExplainer: {e}")

        self.is_loaded = True
        logger.info("ModelRegistry successfully initialized.")

    def get_model(self, model_key: str = "xgb") -> Tuple[Any, str]:
        """Returns the requested model object and canonical model name."""
        k = model_key.lower().strip()
        if k in ["rf", "random_forest", "randomforest"]:
            return self.models.get("rf", self.models.get("xgb")), "Random Forest"
        elif k in ["lstm", "flowlstm"]:
            return self.models.get("lstm"), "FlowLSTM"
        else:
            return self.models.get("xgb", self.models.get("rf")), "XGBoost"

    def get_uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 2)


registry = ModelRegistry()
