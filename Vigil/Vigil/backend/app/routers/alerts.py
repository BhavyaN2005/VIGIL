import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.alert import AlertRead, AlertUpdate, PaginatedAlerts
from app.services.alert_service import AlertService

logger = logging.getLogger("VIGIL-AlertsRouter")
router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=PaginatedAlerts)
async def list_alerts(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    severity: Optional[str] = Query(None, description="Filter by severity: 'Low', 'Medium', 'High', 'Critical'"),
    attack_class: Optional[str] = Query(None, description="Filter by attack class"),
    is_zero_day: Optional[bool] = Query(None, description="Filter by zero-day anomaly flag"),
    status: Optional[str] = Query(None, description="Filter by status: 'Open', 'Investigating', 'Resolved'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves paginated SOC threat alerts from SQLite with multi-field filtering.
    """
    alerts = await AlertService.get_alerts(
        db=db,
        page=page,
        page_size=page_size,
        severity=severity,
        attack_class=attack_class,
        is_zero_day=is_zero_day,
        status=status,
    )
    return alerts


@router.patch("/{alert_id}", response_model=AlertRead)
async def update_alert_status(
    alert_id: int,
    payload: AlertUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates analyst investigation status ('Open' -> 'Investigating' -> 'Resolved') and notes.
    """
    updated = await AlertService.update_alert(db=db, alert_id=alert_id, update_data=payload)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found.")

    return AlertRead.model_validate(updated)
