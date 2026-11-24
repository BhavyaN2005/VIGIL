import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from app.core.model_registry import registry
from app.core.security import determine_severity, generate_flow_metadata
from app.schemas.flow import (
    BatchFlowItem,
    BatchFlowResponse,
    SHAPContribution,
    SingleFlowRequest,
    SingleFlowResponse,
    ZeroDayDetection,
)
from ml.config import TARGET_CLASSES

logger = logging.getLogger("VIGIL-InferenceService")


class InferenceService:
    @staticmethod
    def _prepare_vector(request: SingleFlowRequest) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares raw and scaled numpy feature vectors from request, sanitizing non-finite values."""
        if request.features is not None:
            raw = np.array([request.features.get(f, 0.0) for f in registry.feature_names], dtype=np.float64)
        elif request.feature_vector is not None:
            raw = np.array(request.feature_vector, dtype=np.float64)
        else:
            raw = np.zeros(len(registry.feature_names), dtype=np.float64)

        # Sanitize any NaN, Inf, or out-of-range floats from dirty network captures
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)

        scaled = registry.scaler.transform(raw.reshape(1, -1))[0]
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=50.0, neginf=-50.0)
        return raw, scaled

    @classmethod
    def predict_single(
        cls, request: SingleFlowRequest, model_key: str = "xgb"
    ) -> SingleFlowResponse:
        t0 = time.perf_counter()
        raw_vec, scaled_vec = cls._prepare_vector(request)
        X_2d = scaled_vec.reshape(1, -1)

        # 1. Supervised Model Classification
        model_obj, model_name = registry.get_model(model_key)
        
        if TORCH_AVAILABLE and model_key.lower() == "lstm" and isinstance(model_obj, torch.nn.Module):
            model_obj.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X_2d, dtype=torch.float32)
                logits = model_obj(tensor_x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
        else:
            probs = model_obj.predict_proba(X_2d)[0]
            pred_idx = int(np.argmax(probs))

        pred_class = registry.label_map.get(pred_idx, TARGET_CLASSES[pred_idx])
        confidence = float(probs[pred_idx])

        # 2. FlowAutoencoder Zero-Day Anomaly Detection
        recon_error = 0.0
        is_zero_day = False
        if TORCH_AVAILABLE and registry.autoencoder is not None:
            with torch.no_grad():
                ae_input = torch.tensor(X_2d, dtype=torch.float32)
                recon_error = float(registry.autoencoder.compute_reconstruction_error(ae_input).item())
                is_zero_day = recon_error > registry.ae_threshold

        # 3. Severity Calculation
        severity = determine_severity(
            pred_class,
            confidence,
            is_zero_day=is_zero_day,
            reconstruction_error=recon_error,
            threshold_tau=registry.ae_threshold,
        )

        # 4. TreeSHAP Local Feature Attribution
        base_value = 0.0
        top_contributions: List[SHAPContribution] = []
        explainer = registry.explainers.get("xgb") or registry.explainers.get("rf")
        
        if explainer is not None:
            try:
                shap_vals = explainer.shap_values(X_2d)
                if isinstance(shap_vals, list):
                    class_shap = shap_vals[pred_idx][0]
                elif len(shap_vals.shape) == 3:
                    class_shap = shap_vals[0, :, pred_idx]
                else:
                    class_shap = shap_vals[0]

                exp_val = explainer.expected_value
                base_value = float(exp_val[pred_idx] if isinstance(exp_val, (list, np.ndarray)) else exp_val)

                # Rank top 8 features by absolute SHAP attribution
                abs_scores = np.abs(class_shap)
                top_indices = np.argsort(abs_scores)[::-1][:8]

                for idx in top_indices:
                    score = float(class_shap[idx])
                    top_contributions.append(
                        SHAPContribution(
                            feature_name=registry.feature_names[idx],
                            raw_value=round(float(raw_vec[idx]), 4),
                            scaled_value=round(float(scaled_vec[idx]), 4),
                            shap_score=round(score, 5),
                            contribution_direction="positive" if score >= 0 else "negative",
                        )
                    )
            except Exception as e:
                logger.warning(f"Error computing TreeSHAP for single flow: {e}")

        # Fallback if SHAP computation produced fewer than 8 features
        if not top_contributions:
            for i in range(min(8, len(registry.feature_names))):
                top_contributions.append(
                    SHAPContribution(
                        feature_name=registry.feature_names[i],
                        raw_value=round(float(raw_vec[i]), 4),
                        scaled_value=round(float(scaled_vec[i]), 4),
                        shap_score=0.0,
                        contribution_direction="positive",
                    )
                )

        class_probabilities = {
            registry.label_map.get(i, TARGET_CLASSES[i]): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return SingleFlowResponse(
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            severity=severity,
            model_used=model_name,
            class_probabilities=class_probabilities,
            zero_day_detection=ZeroDayDetection(
                reconstruction_error=round(recon_error, 5),
                threshold_tau=round(registry.ae_threshold, 5),
                is_zero_day_anomaly=is_zero_day,
            ),
            base_value=round(base_value, 4),
            top_contributions=top_contributions,
            alert_logged=False,
            alert_id=None,
        )

    @classmethod
    def predict_batch(
        cls, df: pd.DataFrame, model_key: str = "xgb"
    ) -> BatchFlowResponse:
        t0 = time.perf_counter()
        model_obj, model_name = registry.get_model(model_key)

        # 1. Align Features with Registry Schema
        df.columns = df.columns.str.strip()
        feature_matrix = []
        for feat in registry.feature_names:
            if feat in df.columns:
                feature_matrix.append(pd.to_numeric(df[feat], errors="coerce").fillna(0.0).values)
            else:
                feature_matrix.append(np.zeros(len(df), dtype=np.float64))

        X_raw = np.column_stack(feature_matrix)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = registry.scaler.transform(X_raw)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=50.0, neginf=-50.0)

        # 2. Batch Inference
        if TORCH_AVAILABLE and model_key.lower() == "lstm" and isinstance(model_obj, torch.nn.Module):
            model_obj.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
                logits = model_obj(tensor_x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_indices = np.argmax(probs, axis=-1)
        else:
            probs = model_obj.predict_proba(X_scaled)
            pred_indices = np.argmax(probs, axis=-1)

        confidences = np.max(probs, axis=-1)

        # 3. Batch Autoencoder Zero-Day Anomaly Detection
        reconstruction_errors = np.zeros(len(df))
        if TORCH_AVAILABLE and registry.autoencoder is not None:
            with torch.no_grad():
                ae_input = torch.tensor(X_scaled, dtype=torch.float32)
                reconstruction_errors = registry.autoencoder.compute_reconstruction_error(ae_input).cpu().numpy()

        is_zero_days = reconstruction_errors > registry.ae_threshold
        zero_day_count = int(is_zero_days.sum())

        # 4. Aggregate Attack Summary
        attack_summary: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
        sample_flows: List[BatchFlowItem] = []
        critical_count = 0

        # Sample preview (up to 50 rows)
        preview_limit = min(50, len(df))
        for i in range(len(df)):
            c_idx = int(pred_indices[i])
            c_name = registry.label_map.get(c_idx, TARGET_CLASSES[c_idx])
            attack_summary[c_name] += 1
            conf = float(confidences[i])
            zd = bool(is_zero_days[i])
            err = float(reconstruction_errors[i])
            
            sev = determine_severity(c_name, conf, is_zero_day=zd, reconstruction_error=err, threshold_tau=registry.ae_threshold)
            if sev == "Critical":
                critical_count += 1

            if i < preview_limit:
                dst_port = int(X_raw[i, 0]) if X_raw.shape[1] > 0 else 80
                sample_flows.append(
                    BatchFlowItem(
                        row_index=i,
                        predicted_class=c_name,
                        confidence=round(conf, 4),
                        severity=sev,
                        is_zero_day_anomaly=zd,
                        reconstruction_error=round(err, 5),
                        dst_port=dst_port,
                    )
                )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return BatchFlowResponse(
            total_flows_processed=len(df),
            model_used=model_name,
            inference_latency_ms=latency_ms,
            attack_summary=attack_summary,
            zero_day_anomalies_detected=zero_day_count,
            critical_alerts_count=critical_count,
            sample_flows=sample_flows,
        )
