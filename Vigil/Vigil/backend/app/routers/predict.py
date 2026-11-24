import io
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import generate_flow_metadata
from app.schemas.flow import BatchFlowResponse, SingleFlowRequest, SingleFlowResponse
from app.services.alert_service import AlertService
from app.services.inference_service import InferenceService

logger = logging.getLogger("VIGIL-PredictRouter")
router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/single", response_model=SingleFlowResponse)
async def predict_single(
    request: SingleFlowRequest,
    model: str = Query("xgb", description="Model selector: 'xgb', 'rf', or 'lstm'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs single-flow inference across selected classifier + Autoencoder Zero-Day anomaly check
    + TreeSHAP top 8 feature attribution. Automatically logs threat alerts to SQLite.
    """
    try:
        response = InferenceService.predict_single(request, model_key=model)

        # Auto-log threats to SQLite if attack detected or zero-day triggered
        if response.predicted_class != "Benign" or response.zero_day_detection.is_zero_day_anomaly:
            # Metadata fallback
            dst_port = request.dst_port or (
                int(request.features.get("Destination Port", 80)) if request.features else 80
            )
            src_ip = request.src_ip
            dst_ip = request.dst_ip
            protocol = request.protocol or "TCP"
            
            if not src_ip or not dst_ip:
                gen_src, gen_dst, gen_proto = generate_flow_metadata(dst_port)
                src_ip = src_ip or gen_src
                dst_ip = dst_ip or gen_dst
                protocol = protocol or gen_proto

            top_feat_name = (
                response.top_contributions[0].feature_name
                if response.top_contributions
                else "Unknown"
            )

            alert_record = await AlertService.create_alert(
                db=db,
                src_ip=src_ip,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=protocol,
                attack_class=response.predicted_class,
                severity=response.severity,
                confidence=response.confidence,
                is_zero_day=response.zero_day_detection.is_zero_day_anomaly,
                reconstruction_error=response.zero_day_detection.reconstruction_error,
                top_contributing_feature=top_feat_name,
                model_used=response.model_used,
            )
            response.alert_logged = True
            response.alert_id = alert_record.id

        return response
    except Exception as e:
        logger.error(f"Error during single prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchFlowResponse)
async def predict_batch(
    file: UploadFile = File(..., description="CSV file capture containing flow features"),
    model: str = Query("xgb", description="Model selector: 'xgb', 'rf', or 'lstm'"),
):
    """
    Async chunked CSV capture upload and vectorized batch inference.
    Validates headers, computes multi-model classification, Autoencoder zero-day anomalies,
    and returns class breakdown distributions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload a valid CSV capture.")

    try:
        # Read contents asynchronously
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        # Parse CSV into dataframe
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV contains header only, no flow data rows.")

        response = InferenceService.predict_batch(df, model_key=model)
        return response

    except HTTPException:
        raise
    except pd.errors.ParserError as pe:
        raise HTTPException(status_code=400, detail=f"Malformed CSV capture: {str(pe)}")
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
