"""
Unit Tests for VIGIL Preprocessing Pipeline (ml/preprocess.py)
Validates whitespace cleaning, inf/NaN handling, deduplication, 7-class mapping,
rare class exclusion, and zero data leakage.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from ml.config import SAMPLE_DATA_PATH, TARGET_CLASSES
from ml.preprocess import clean_dataframe, prepare_train_test_split


@pytest.fixture
def dirty_mock_df():
    """Generates a small test dataframe containing common CIC-IDS2017 edge cases."""
    data = {
        " Destination Port": [80, 443, 22, 80, 8080, 53, 4444, 443, 80, 80],
        " Flow Duration ": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 1000, 1000],
        "Flow Bytes/s": [10.5, np.inf, 30.0, 40.0, -np.inf, 60.0, 70.0, 80.0, 10.5, 10.5],
        " Total Fwd Packets": [2, 4, 6, np.nan, 10, 12, 14, 16, 2, 2],
        " Label ": [
            "BENIGN",
            "DoS Hulk",
            "SSH-Patator",
            "DDoS",
            "Web Attack – Brute Force",
            "PortScan",
            "Heartbleed",    # Should be dropped
            "Infiltration",  # Should be dropped
            "BENIGN",        # Duplicate
            "BENIGN",        # Duplicate
        ],
    }
    return pd.DataFrame(data)


def test_column_whitespace_stripping(dirty_mock_df):
    cleaned, stats = clean_dataframe(dirty_mock_df.copy())
    for col in cleaned.columns:
        assert col == col.strip(), f"Column header '{col}' still contains leading/trailing whitespace"
    assert "Label" in cleaned.columns


def test_inf_and_nan_removal(dirty_mock_df):
    cleaned, stats = clean_dataframe(dirty_mock_df.copy())
    assert not cleaned.isna().any().any(), "Cleaned dataframe contains remaining NaN values"
    assert not np.isinf(cleaned.select_dtypes(include=np.number).values).any(), "Cleaned dataframe contains inf"
    assert stats["dropped_nan_rows"] > 0


def test_duplicate_removal(dirty_mock_df):
    cleaned, stats = clean_dataframe(dirty_mock_df.copy())
    assert stats["dropped_duplicate_rows"] >= 1
    assert len(cleaned) < len(dirty_mock_df)


def test_rare_class_dropping(dirty_mock_df):
    cleaned, stats = clean_dataframe(dirty_mock_df.copy())
    assert "Heartbleed" not in cleaned["Label"].values
    assert "Infiltration" not in cleaned["Label"].values
    assert stats["dropped_rare_classes"]["Heartbleed"] == 1
    assert stats["dropped_rare_classes"]["Infiltration"] == 1


def test_seven_class_consolidation():
    test_labels = pd.DataFrame({
        "Flow Duration": [100] * 8,
        "Label": [
            "BENIGN",
            "DoS GoldenEye",
            "DDoS",
            "PortScan",
            "FTP-Patator",
            "Web Attack - XSS",
            "Web Attack – Sql Injection",
            "Bot",
        ],
    })
    cleaned, _ = clean_dataframe(test_labels)
    assert set(cleaned["Label"].unique()).issubset(set(TARGET_CLASSES))
    assert "Benign" in cleaned["Label"].values
    assert "DoS" in cleaned["Label"].values
    assert "Brute Force" in cleaned["Label"].values
    assert "Web Attack" in cleaned["Label"].values


def test_zero_data_leakage():
    # Use real sample flows
    df = pd.read_csv(SAMPLE_DATA_PATH)
    cleaned, _ = clean_dataframe(df)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = prepare_train_test_split(cleaned)
    
    # Verify 80/20 proportion
    total_samples = len(y_train) + len(y_test)
    test_ratio = len(y_test) / total_samples
    assert 0.19 <= test_ratio <= 0.21, f"Expected ~0.20 test split, got {test_ratio:.3f}"
    
    # Verify training set has zero mean and unit variance
    train_mean = np.mean(X_train_scaled, axis=0)
    train_std = np.std(X_train_scaled, axis=0)
    np.testing.assert_allclose(train_mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(train_std, 1.0, atol=1e-5)
    
    # Test set transformed using scaler should NOT have exactly zero mean
    # (since scaler parameters were learned strictly on train)
    test_mean = np.mean(X_test_scaled, axis=0)
    assert not np.allclose(test_mean, 0.0, atol=1e-6), "Test set had exact zero mean; data leakage suspected!"
