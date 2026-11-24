"""
VIGIL: Configuration and Hyperparameters
Supports seamless execution across local development (8GB RAM sample mode)
and Kaggle / Colab full-scale cluster execution.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "sample_flows.csv"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

# Global Random State
RANDOM_STATE = 42

# 7-Class Canonical Taxonomy Mapping
LABEL_MAPPING = {
    # Benign
    "BENIGN": "Benign",
    "Benign": "Benign",
    
    # DoS
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS": "DoS",
    
    # DDoS
    "DDoS": "DDoS",
    
    # PortScan
    "PortScan": "PortScan",
    
    # Brute Force
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Brute Force": "Brute Force",
    
    # Web Attack (handles hyphens, en-dashes, em-dashes, non-breaking spaces)
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack": "Web Attack",
    
    # Bot
    "Bot": "Bot",
}

# Rare classes explicitly dropped with thesis documentation
RARE_CLASSES_TO_DROP = ["Heartbleed", "Infiltration"]

# Canonical Ordered Target Classes
TARGET_CLASSES = [
    "Benign",
    "DoS",
    "DDoS",
    "PortScan",
    "Brute Force",
    "Web Attack",
    "Bot",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(TARGET_CLASSES)}

# Model Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 16,
    "min_samples_split": 5,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": len(TARGET_CLASSES),
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "epochs": 10,
    "batch_size": 128,
}

AUTOENCODER_PARAMS = {
    "hidden_dims": [48, 24, 12],
    "learning_rate": 1e-3,
    "epochs": 12,
    "batch_size": 128,
    "threshold_percentile": 99.0,
}
