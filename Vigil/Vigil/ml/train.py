"""
VIGIL: Production Multi-Model Training & Benchmark Suite
Trains Random Forest, XGBoost, FlowLSTM, and FlowAutoencoder on exact same split.
Computes comprehensive metrics (Macro/Weighted F1, ROC/PR AUC, Latency, Confusion Matrices)
and exports all production artifacts to ml/artifacts/.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from ml.config import (
    ARTIFACTS_DIR,
    AUTOENCODER_PARAMS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    LSTM_PARAMS,
    RANDOM_STATE,
    RF_PARAMS,
    SAMPLE_DATA_PATH,
    TARGET_CLASSES,
    XGB_PARAMS,
)
from ml.models.autoencoder_net import FlowAutoencoder
from ml.models.lstm_net import FlowLSTM
from ml.preprocess import clean_dataframe, prepare_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VIGIL-Train")


def measure_inference_latency(predict_fn, X_sample: np.ndarray, n_iterations: int = 10) -> float:
    """
    Accurately measures inference latency in milliseconds per 1,000 flows.
    """
    # Warmup
    predict_fn(X_sample[:50])
    
    # Timing
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        predict_fn(X_sample)
    elapsed = time.perf_counter() - t0
    
    total_samples = len(X_sample) * n_iterations
    ms_per_1k = (elapsed / total_samples) * 1000 * 1000
    return round(ms_per_1k, 3)


def compute_multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str
) -> Dict[str, Any]:
    """
    Computes comprehensive macro/weighted metrics, per-class breakdown,
    ROC-AUC (OVR), PR-AUC (OVR), and confusion matrices.
    """
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    accuracy = float((y_true == y_pred).mean())

    # One-vs-Rest ROC-AUC and PR-AUC
    n_classes = len(TARGET_CLASSES)
    y_true_oh = np.eye(n_classes)[y_true]
    
    try:
        roc_auc = float(roc_auc_score(y_true_oh, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true_oh, y_proba, average="macro"))
    except Exception:
        pr_auc = 0.0

    # Per-class metrics
    clf_report = classification_report(
        y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True, zero_division=0
    )
    per_class = {}
    for c_name in TARGET_CLASSES:
        if c_name in clf_report:
            per_class[c_name] = {
                "precision": round(float(clf_report[c_name]["precision"]), 4),
                "recall": round(float(clf_report[c_name]["recall"]), 4),
                "f1_score": round(float(clf_report[c_name]["f1-score"]), 4),
                "support": int(clf_report[c_name]["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_macro": round(roc_auc, 4),
        "pr_auc_macro": round(pr_auc, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    logger.info("Training Random Forest Classifier (100 trees, balanced subsample)...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    metrics = compute_multiclass_metrics(y_test, y_pred, y_proba, "Random Forest")
    metrics["latency_ms_per_1k"] = measure_inference_latency(rf.predict, X_test)
    logger.info(f"Random Forest -> Macro F1: {metrics['macro_f1']}, Latency: {metrics['latency_ms_per_1k']} ms/1k")
    return rf, metrics


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[XGBClassifier, Dict[str, Any]]:
    logger.info("Training XGBoost Classifier (multi:softprob, balanced sample weights)...")
    sample_weights = compute_sample_weight("balanced", y_train)
    
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)

    metrics = compute_multiclass_metrics(y_test, y_pred, y_proba, "XGBoost")
    metrics["latency_ms_per_1k"] = measure_inference_latency(xgb.predict, X_test)
    logger.info(f"XGBoost -> Macro F1: {metrics['macro_f1']}, Latency: {metrics['latency_ms_per_1k']} ms/1k")
    return xgb, metrics


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[FlowLSTM, Dict[str, Any]]:
    logger.info("Training FlowLSTM Neural Network...")
    input_dim = X_train.shape[1]
    num_classes = len(TARGET_CLASSES)
    
    model = FlowLSTM(
        input_dim=input_dim,
        hidden_dim=LSTM_PARAMS.get("hidden_dim", LSTM_PARAMS.get("hidden_size", 64)),
        num_layers=LSTM_PARAMS["num_layers"],
        num_classes=num_classes,
        dropout=LSTM_PARAMS["dropout"],
    )

    # Compute class weights for loss function
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_PARAMS["learning_rate"])

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=LSTM_PARAMS["batch_size"], shuffle=True)

    model.train()
    for epoch in range(LSTM_PARAMS["epochs"]):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(test_tensor)
        y_proba = torch.softmax(logits, dim=-1).cpu().numpy()
        y_pred = np.argmax(y_proba, axis=-1)

    metrics = compute_multiclass_metrics(y_test, y_pred, y_proba, "FlowLSTM")
    
    def lstm_predict(x):
        model.eval()
        with torch.no_grad():
            return torch.argmax(model(torch.tensor(x, dtype=torch.float32)), dim=-1).numpy()
            
    metrics["latency_ms_per_1k"] = measure_inference_latency(lstm_predict, X_test)
    logger.info(f"FlowLSTM -> Macro F1: {metrics['macro_f1']}, Latency: {metrics['latency_ms_per_1k']} ms/1k")
    return model, metrics


def train_autoencoder(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[FlowAutoencoder, float, Dict[str, Any]]:
    logger.info("Training FlowAutoencoder for Unsupervised Zero-Day Anomaly Detection...")
    input_dim = X_train.shape[1]
    
    # Train STRICTLY on Benign flows (class 0)
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]
    logger.info(f"Autoencoder training strictly on {len(X_train_benign)} Benign samples.")

    model = FlowAutoencoder(input_dim=input_dim, hidden_dims=AUTOENCODER_PARAMS["hidden_dims"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=AUTOENCODER_PARAMS["learning_rate"])

    dataset = TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=AUTOENCODER_PARAMS["batch_size"], shuffle=True)

    model.train()
    for epoch in range(AUTOENCODER_PARAMS["epochs"]):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

    # Calculate Adaptive Threshold tau on training benign flows
    model.eval()
    with torch.no_grad():
        train_benign_t = torch.tensor(X_train_benign, dtype=torch.float32)
        train_recon_errors = model.compute_reconstruction_error(train_benign_t).cpu().numpy()
        tau = float(np.percentile(train_recon_errors, AUTOENCODER_PARAMS["threshold_percentile"]))
        logger.info(f"Adaptive Zero-Day Anomaly Threshold (tau at {AUTOENCODER_PARAMS['threshold_percentile']}th percentile): {tau:.5f}")

    # Evaluate on test set (Binary: 0 = Benign, 1 = Attack)
    with torch.no_grad():
        test_t = torch.tensor(X_test, dtype=torch.float32)
        test_recon_errors = model.compute_reconstruction_error(test_t).cpu().numpy()
        
    y_test_binary = (y_test != 0).astype(int)  # 1 for any attack
    predicted_anomalies = (test_recon_errors > tau).astype(int)

    # Anomaly AUROC
    try:
        anomaly_auroc = float(roc_auc_score(y_test_binary, test_recon_errors))
    except Exception:
        anomaly_auroc = 0.0

    # Recall on actual attacks
    attack_mask = y_test_binary == 1
    attack_recall = float((predicted_anomalies[attack_mask] == 1).mean()) if attack_mask.sum() > 0 else 0.0

    # False positive rate on benign
    benign_test_mask = y_test_binary == 0
    benign_fpr = float((predicted_anomalies[benign_test_mask] == 1).mean()) if benign_test_mask.sum() > 0 else 0.0

    metrics = {
        "model_name": "FlowAutoencoder",
        "role": "Unsupervised Zero-Day Anomaly Detection",
        "threshold_tau": round(tau, 5),
        "anomaly_auroc": round(anomaly_auroc, 4),
        "attack_detection_recall": round(attack_recall, 4),
        "benign_false_alarm_rate": round(benign_fpr, 4),
        "mean_reconstruction_error_benign": round(float(test_recon_errors[benign_test_mask].mean()), 5),
        "mean_reconstruction_error_attack": round(float(test_recon_errors[attack_mask].mean()), 5),
    }

    def ae_predict(x):
        model.eval()
        with torch.no_grad():
            return (model.compute_reconstruction_error(torch.tensor(x, dtype=torch.float32)).numpy() > tau).astype(int)

    metrics["latency_ms_per_1k"] = measure_inference_latency(ae_predict, X_test)
    logger.info(f"Autoencoder -> Anomaly AUROC: {metrics['anomaly_auroc']}, Attack Recall: {metrics['attack_detection_recall']}, Latency: {metrics['latency_ms_per_1k']} ms/1k")
    return model, tau, metrics


def extract_top_feature_importances(rf: RandomForestClassifier, xgb: XGBClassifier, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Extracts top 15 consensus features across Random Forest and XGBoost."""
    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_

    combined = (rf_imp + xgb_imp) / 2.0
    top_indices = np.argsort(combined)[::-1][:15]

    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature": feature_names[idx],
            "combined_importance": round(float(combined[idx]), 4),
            "rf_importance": round(float(rf_imp[idx]), 4),
            "xgb_importance": round(float(xgb_imp[idx]), 4),
        })
    return top_features


def main():
    parser = argparse.ArgumentParser(description="VIGIL Production Multi-Model Pipeline")
    parser.add_argument("--data-dir", type=str, default=str(SAMPLE_DATA_PATH), help="Path to CSV dataset")
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR), help="Output artifacts directory")
    parser.add_argument("--sample-mode", type=bool, default=True, help="Running on sample flows for rapid execution")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting VIGIL Training Pipeline (Sample Mode: {args.sample_mode}) ===")
    
    # 1. Preprocess & Split
    import pandas as pd
    raw_df = pd.read_csv(args.data_dir, low_memory=False)
    cleaned_df, cleaning_stats = clean_dataframe(raw_df)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_split(cleaned_df)

    # 2. Train 4 Models
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_test, y_test)
    ae_model, ae_tau, ae_metrics = train_autoencoder(X_train, y_train, X_test, y_test)

    # 3. Determine Best Supervised Model by Macro F1
    supervised_candidates = [
        ("Random Forest", rf_model, rf_metrics),
        ("XGBoost", xgb_model, xgb_metrics),
    ]
    best_name, best_model_obj, best_metrics = max(supervised_candidates, key=lambda x: x[2]["macro_f1"])
    logger.info(f"Best Supervised Model: {best_name} with Macro F1: {best_metrics['macro_f1']}")

    # 4. Top Feature Importances
    top_features = extract_top_feature_importances(rf_model, xgb_model, feature_names)

    # 5. Export Artifacts
    logger.info(f"Exporting serialized model artifacts to: {output_dir}")
    
    # Best Model & Tree Models
    joblib.dump(best_model_obj, output_dir / "best_model.joblib")
    joblib.dump(rf_model, output_dir / "rf_model.joblib")
    joblib.dump(xgb_model, output_dir / "xgb_model.joblib")
    
    # Neural Networks (PyTorch)
    torch.save(lstm_model.state_dict(), output_dir / "lstm_model.pt")
    torch.save(ae_model.state_dict(), output_dir / "autoencoder.pt")
    
    # Scaler
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    # Autoencoder Threshold
    with open(output_dir / "autoencoder_threshold.json", "w") as f:
        json.dump({"threshold_tau": ae_tau, "percentile": AUTOENCODER_PARAMS["threshold_percentile"]}, f, indent=2)

    # Feature List
    with open(output_dir / "feature_list.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Label Map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in IDX_TO_CLASS.items()}, f, indent=2)

    # Comprehensive Metrics JSON
    consolidated_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "sample_mode": args.sample_mode,
        "dataset_statistics": cleaning_stats,
        "best_model": {
            "name": best_name,
            "macro_f1": best_metrics["macro_f1"],
            "weighted_f1": best_metrics["weighted_f1"],
            "accuracy": best_metrics["accuracy"],
        },
        "model_comparison": [
            rf_metrics,
            xgb_metrics,
            lstm_metrics,
            ae_metrics,
        ],
        "top_15_features": top_features,
        "classes": TARGET_CLASSES,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(consolidated_metrics, f, indent=2)

    logger.info("=== VIGIL ML Pipeline Complete. All Artifacts Successfully Exported! ===")


if __name__ == "__main__":
    main()
