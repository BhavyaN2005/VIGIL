import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.services.stream_service import stream_service

logger = logging.getLogger("VIGIL-StreamRouter")
router = APIRouter(tags=["Streaming"])


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    rate: float = Query(2.0, ge=0.5, le=10.0, description="Flows emitted per second"),
    model: str = Query("xgb", description="Model selector: 'xgb', 'rf', or 'lstm'"),
):
    """
    Low-latency WebSocket live network flow telemetry stream.
    Emits continuous packet events from real replay traffic + live attack injections.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected. Rate: {rate} flows/sec, Model: {model}")

    try:
        async for packet in stream_service.flow_generator(rate=rate, model_key=model):
            await websocket.send_json(packet)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket stream terminated: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/stream/replay")
async def sse_stream(
    rate: float = Query(2.0, ge=0.5, le=10.0, description="Flows emitted per second"),
    model: str = Query("xgb", description="Model selector: 'xgb', 'rf', or 'lstm'"),
):
    """
    Server-Sent Events (SSE) alternative for live network flow streaming.
    """
    async def event_generator():
        async for packet in stream_service.flow_generator(rate=rate, model_key=model):
            yield f"data: {json.dumps(packet)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stream/inject")
async def inject_attack_burst(
    attack: str = Query("DDoS", description="Attack type: 'DDoS', 'DoS', 'PortScan', 'Brute Force', 'Bot'"),
    count: int = Query(15, ge=1, le=100, description="Number of anomalous flows to inject"),
):
    """
    Injects an immediate burst of simulated attack flows into the live replay queue.
    Enables interactive live SOC demonstration of threat detection and telemetry spikes.
    """
    queue_len = stream_service.inject_attack_burst(attack_class=attack, count=count)
    return {
        "status": "success",
        "message": f"Injected {count} x '{attack}' flows into live stream queue.",
        "attack_class": attack,
        "count_injected": count,
        "total_queued": queue_len,
    }
