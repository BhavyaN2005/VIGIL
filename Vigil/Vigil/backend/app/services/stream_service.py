import asyncio
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.core.model_registry import registry
from app.core.security import determine_severity, generate_flow_metadata
from app.services.inference_service import InferenceService
from app.schemas.flow import SingleFlowRequest

logger = logging.getLogger("VIGIL-StreamService")


class StreamService:
    def __init__(self, sample_csv_path: Optional[Path] = None):
        self.sample_csv_path = sample_csv_path or settings.SAMPLE_DATA_PATH
        self.injection_queue: deque = deque()
        self.df_cache: Optional[pd.DataFrame] = None
        self._load_cache()

    def _load_cache(self):
        """Loads sample CSV into memory for low-latency streaming."""
        if self.sample_csv_path.exists():
            try:
                df = pd.read_csv(self.sample_csv_path, low_memory=False)
                df.columns = df.columns.str.strip()
                self.df_cache = df
                logger.info(f"Loaded {len(df)} sample replay flows into StreamService cache.")
            except Exception as e:
                logger.warning(f"Failed to load sample flows for replay: {e}")
                self.df_cache = None
        else:
            self.df_cache = None

    def inject_attack_burst(self, attack_class: str = "DDoS", count: int = 10) -> int:
        """
        Injects a burst of simulated attack flows into the live stream queue.
        Enables interactive live SOC demonstration of immediate threat spikes.
        """
        logger.info(f"Injecting simulated attack burst: {count} x '{attack_class}' flows.")
        for _ in range(count):
            flow = self._generate_synthetic_attack_flow(attack_class)
            self.injection_queue.append(flow)
        return len(self.injection_queue)

    def _generate_synthetic_attack_flow(self, attack_class: str) -> Dict[str, Any]:
        """Generates realistic synthetic flow attributes for injected attack types."""
        flow = {f: float(np.random.exponential(15.0)) for f in registry.feature_names}
        
        if attack_class == "DDoS":
            flow["Flow Packets/s"] = 850000.0
            flow["Flow Bytes/s"] = 150000000.0
            flow["Flow Duration"] = 1200.0
            flow["Destination Port"] = 80.0
        elif attack_class == "DoS":
            flow["Total Fwd Packets"] = 1200.0
            flow["Total Backward Packets"] = 1.0
            flow["Flow Duration"] = 2500000.0
            flow["Destination Port"] = 80.0
        elif attack_class == "PortScan":
            flow["Destination Port"] = float(random.randint(1, 65535))
            flow["SYN Flag Count"] = 1.0
            flow["ACK Flag Count"] = 0.0
            flow["Flow Duration"] = 85.0
        elif attack_class == "Brute Force":
            flow["Destination Port"] = 22.0
            flow["PSH Flag Count"] = 35.0
            flow["Flow Duration"] = 8500.0
        elif attack_class == "Bot":
            flow["Destination Port"] = 6667.0
            flow["Idle Mean"] = 120000.0
        else:
            flow["Destination Port"] = 443.0

        return {"features": flow, "forced_label": attack_class}

    async def flow_generator(
        self, rate: float = 2.0, model_key: str = "xgb"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously yields classified flow packets at the specified rate (flows/sec).
        Pops injected attack bursts with highest priority.
        """
        delay = 1.0 / max(settings.MIN_STREAM_RATE, min(rate, settings.MAX_STREAM_RATE))
        row_idx = 0

        while True:
            t_start = time.perf_counter()

            # 1. Check Injection Queue First
            if len(self.injection_queue) > 0:
                injected = self.injection_queue.popleft()
                feat_dict = injected["features"]
                is_injected = True
            else:
                is_injected = False
                if self.df_cache is not None and len(self.df_cache) > 0:
                    row = self.df_cache.iloc[row_idx % len(self.df_cache)]
                    row_idx += 1
                    feat_dict = {}
                    for f in registry.feature_names:
                        val = pd.to_numeric(row.get(f, 0.0), errors="coerce")
                        if pd.isna(val) or np.isinf(val):
                            val = 0.0
                        feat_dict[f] = float(val)
                else:
                    # Fallback random normal flow
                    feat_dict = {f: float(np.random.exponential(5.0)) for f in registry.feature_names}
                    feat_dict["Destination Port"] = 443.0

            # 2. Run Single Inference
            req = SingleFlowRequest(features=feat_dict)
            prediction = InferenceService.predict_single(req, model_key=model_key)

            # If injected with forced label and prediction came out Benign, respect simulation intent
            if is_injected and "forced_label" in injected and prediction.predicted_class == "Benign":
                prediction.predicted_class = injected["forced_label"]
                prediction.severity = "High" if injected["forced_label"] != "DDoS" else "Critical"

            dst_port = int(feat_dict.get("Destination Port", 80))
            src_ip, dst_ip, protocol = generate_flow_metadata(dst_port)

            is_zd = bool(prediction.zero_day_detection.is_zero_day_anomaly)
            recon_err = float(prediction.zero_day_detection.reconstruction_error)

            top_contribs = [
                {
                    "feature": c.feature_name,
                    "feature_name": c.feature_name,
                    "shap_value": c.shap_score,
                    "shap_score": c.shap_score,
                    "raw_value": c.raw_value,
                    "scaled_value": c.scaled_value,
                    "direction": c.contribution_direction,
                }
                for c in prediction.top_contributions
            ]

            packet_event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "protocol": protocol,
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "severity": prediction.severity,
                "is_zero_day_anomaly": is_zd,
                "reconstruction_error": recon_err,
                "zero_day_detection": {
                    "is_zero_day": is_zd,
                    "is_zero_day_anomaly": is_zd,
                    "reconstruction_error": recon_err,
                    "threshold_tau": float(prediction.zero_day_detection.threshold_tau),
                },
                "is_simulated_injection": is_injected,
                "model_used": prediction.model_used,
                "flow_duration_ms": round(float(feat_dict.get("Flow Duration", 0.0)) / 1000.0, 2),
                "flow_bytes_per_sec": round(float(feat_dict.get("Flow Bytes/s", 0.0)), 2),
                "flow_packets_per_sec": round(float(feat_dict.get("Flow Packets/s", 0.0)), 2),
                "top_contributions": top_contribs,
            }

            # Asynchronously persist alerts to SQLite for threats
            if prediction.predicted_class != "Benign" or is_zd:
                try:
                    from app.core.database import AsyncSessionLocal
                    from app.services.alert_service import AlertService
                    async with AsyncSessionLocal() as session:
                        top_feat = prediction.top_contributions[0].feature_name if prediction.top_contributions else "Anomaly"
                        await AlertService.create_alert(
                            db=session,
                            src_ip=src_ip,
                            dst_ip=dst_ip,
                            dst_port=dst_port,
                            protocol=protocol,
                            attack_class=prediction.predicted_class,
                            severity=prediction.severity,
                            confidence=prediction.confidence,
                            is_zero_day=is_zd,
                            reconstruction_error=recon_err,
                            top_contributing_feature=top_feat,
                            model_used=prediction.model_used,
                        )
                except Exception as ex:
                    logger.debug(f"Failed to persist stream alert: {ex}")

            yield packet_event

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.01, delay - elapsed)
            await asyncio.sleep(sleep_time)


stream_service = StreamService()
