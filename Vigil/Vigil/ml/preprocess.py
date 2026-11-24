"""
VIGIL: Data Preprocessing & Sanitization Pipeline
Handles dirty CSV headers, infinite/NaN handling, deduplication,
7-class taxonomy consolidation, rare class logging, and zero-leakage scaling.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import (
    CLASS_TO_IDX,
    LABEL_MAPPING,
    RANDOM_STATE,
    RARE_CLASSES_TO_DROP,
    SAMPLE_DATA_PATH,
    TARGET_CLASSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VIGIL-Preprocess")


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans raw network flow dataframe:
    1. Strips leading/trailing whitespace from all column names.
    2. Identifies and counts rare classes before dropping them.
    3. Replaces inf and -inf with NaN, then drops NaN and duplicate rows.
    4. Normalizes and consolidates labels into 7 canonical attack categories.
    
    Returns cleaned dataframe and a statistics dictionary for project report/thesis.
    """
    initial_rows = len(df)
    stats: Dict[str, int] = {"initial_rows": initial_rows}

    # 1. Strip column header whitespaces
    df.columns = df.columns.str.strip()
    logger.info(f"Stripped whitespace from {len(df.columns)} column headers.")

    if "Label" not in df.columns:
        raise ValueError("Missing 'Label' column in dataframe after stripping whitespace.")

    # 2. Identify & Drop ultra-rare classes (Heartbleed, Infiltration)
    rare_counts: Dict[str, int] = {}
    for rare_class in RARE_CLASSES_TO_DROP:
        mask = df["Label"].astype(str).str.strip().str.contains(rare_class, case=False, na=False)
        count = int(mask.sum())
        rare_counts[rare_class] = count
        if count > 0:
            df = df[~mask]
            logger.info(f"[Thesis Documentation] Dropped rare class '{rare_class}' (count: {count}) to prevent degenerate splits.")

    stats["dropped_rare_classes"] = rare_counts

    # 3. Handle inf, -inf and NaNs
    feature_cols = [c for c in df.columns if c != "Label"]
    
    # Replace inf and -inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop NaNs
    nan_rows_before = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    stats["dropped_nan_rows"] = nan_rows_before
    logger.info(f"Dropped {nan_rows_before} rows containing NaN or infinite values.")

    # Drop exact duplicates
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    stats["dropped_duplicate_rows"] = duplicate_rows
    logger.info(f"Dropped {duplicate_rows} exact duplicate flow records.")

    # 4. Consolidate labels into 7 canonical classes
    def map_label(raw_label: str) -> str:
        cleaned = str(raw_label).strip()
        # Direct lookup
        if cleaned in LABEL_MAPPING:
            return LABEL_MAPPING[cleaned]
        # Regex / Substring heuristics for unicode hyphens or encoding variants
        if re.search(r"Web\s*Attack", cleaned, re.IGNORECASE):
            return "Web Attack"
        if re.search(r"DoS", cleaned, re.IGNORECASE):
            return "DoS"
        if re.search(r"DDoS", cleaned, re.IGNORECASE):
            return "DDoS"
        if re.search(r"PortScan", cleaned, re.IGNORECASE):
            return "PortScan"
        if re.search(r"Patator", cleaned, re.IGNORECASE):
            return "Brute Force"
        if re.search(r"Bot", cleaned, re.IGNORECASE):
            return "Bot"
        if re.search(r"BENIGN", cleaned, re.IGNORECASE):
            return "Benign"
        return cleaned

    df["Label"] = df["Label"].apply(map_label)
    
    # Filter only target classes
    df = df[df["Label"].isin(TARGET_CLASSES)]

    stats["final_rows"] = len(df)
    stats["class_distribution"] = df["Label"].value_counts().to_dict()
    logger.info(f"Final clean dataset: {len(df)} rows across {len(stats['class_distribution'])} classes.")

    return df, stats


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Performs stratified 80/20 train/test split.
    Fits StandardScaler STRICTLY on the training split to prevent data leakage.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    feature_cols = [c for c in df.columns if c != "Label"]
    
    # Ensure all feature columns are numeric
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = df["Label"].map(CLASS_TO_IDX).values

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Stratified split: {len(y_train)} train samples (80%), {len(y_test)} test samples (20%).")

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Fitted StandardScaler strictly on training split (Zero Data Leakage enforced).")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def run_pipeline(
    csv_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str], Dict]:
    target_path = csv_path or SAMPLE_DATA_PATH
    logger.info(f"Loading raw network capture from: {target_path}")
    
    df = pd.read_csv(target_path, low_memory=False)
    cleaned_df, stats = clean_dataframe(df)
    
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_split(cleaned_df)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, stats


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features, stats = run_pipeline()
    print("Preprocessing completed successfully.")
    print("Class distribution:", stats["class_distribution"])
