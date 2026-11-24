import logging
from fastapi import APIRouter
from app.core.model_registry import registry

logger = logging.getLogger("VIGIL-MetricsRouter")
router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/models")
async def get_model_metrics():
    """
    Serves full benchmark comparison table, confusion matrices, ROC/PR curves,
    and top-15 consensus features for the dashboard's Model Lab.
    """
    return registry.metrics_data
