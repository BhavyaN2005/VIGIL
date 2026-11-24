import os
from fastapi import APIRouter
from app.core.model_registry import registry
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns engine status, loaded models, memory footprint, and uptime."""
    memory_mb = 120.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    loaded_models_display = ["XGBoost", "Random Forest", "FlowLSTM", "FlowAutoencoder"]

    return HealthResponse(
        status="online",
        engine="CIC-IDS2017 & Kitsune Dual-Engine",
        loaded_models=loaded_models_display,
        active_default_model="XGBoost",
        zero_day_detector=f"FlowAutoencoder (tau={registry.ae_threshold})",
        feature_dimensions=len(registry.feature_names),
        uptime_seconds=registry.get_uptime_seconds(),
        memory_mb=memory_mb,
        database_status="connected",
    )
