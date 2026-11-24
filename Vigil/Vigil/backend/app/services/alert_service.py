import math
from typing import List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert
from app.schemas.alert import AlertRead, AlertUpdate, PaginatedAlerts


class AlertService:
    @staticmethod
    async def create_alert(
        db: AsyncSession,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        attack_class: str,
        severity: str,
        confidence: float,
        is_zero_day: bool,
        reconstruction_error: float,
        top_contributing_feature: str,
        model_used: str,
    ) -> Alert:
        """Persists a new alert record to SQLite."""
        alert = Alert(
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            attack_class=attack_class,
            severity=severity,
            confidence=confidence,
            is_zero_day=is_zero_day,
            reconstruction_error=reconstruction_error,
            top_contributing_feature=top_contributing_feature,
            model_used=model_used,
            status="Open",
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def get_alerts(
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        severity: Optional[str] = None,
        attack_class: Optional[str] = None,
        is_zero_day: Optional[bool] = None,
        status: Optional[str] = None,
    ) -> PaginatedAlerts:
        """Queries paginated alerts with multi-field filtering."""
        query = select(Alert)

        if severity:
            query = query.where(Alert.severity == severity)
        if attack_class:
            query = query.where(Alert.attack_class == attack_class)
        if is_zero_day is not None:
            query = query.where(Alert.is_zero_day == is_zero_day)
        if status:
            query = query.where(Alert.status == status)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        # Order and paginate
        query = query.order_by(desc(Alert.timestamp)).offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(query)
        rows = result.scalars().all()

        total_pages = math.ceil(total / page_size) if total > 0 else 1

        items = [
            AlertRead(
                id=r.id,
                timestamp=r.timestamp.isoformat() if r.timestamp else None,
                src_ip=r.src_ip,
                dst_ip=r.dst_ip,
                dst_port=r.dst_port,
                protocol=r.protocol,
                attack_class=r.attack_class,
                severity=r.severity,
                confidence=r.confidence,
                is_zero_day=r.is_zero_day,
                reconstruction_error=r.reconstruction_error,
                top_contributing_feature=r.top_contributing_feature,
                model_used=r.model_used,
                status=r.status,
                analyst_notes=r.analyst_notes,
            )
            for r in rows
        ]

        return PaginatedAlerts(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=items,
        )

    @staticmethod
    async def update_alert(
        db: AsyncSession, alert_id: int, update_data: AlertUpdate
    ) -> Optional[Alert]:
        """Updates analyst status and notes for a specific alert."""
        query = select(Alert).where(Alert.id == alert_id)
        result = await db.execute(query)
        alert = result.scalar_one_or_none()

        if alert is None:
            return None

        if update_data.status is not None:
            alert.status = update_data.status
        if update_data.analyst_notes is not None:
            alert.analyst_notes = update_data.analyst_notes

        await db.commit()
        await db.refresh(alert)
        return alert
