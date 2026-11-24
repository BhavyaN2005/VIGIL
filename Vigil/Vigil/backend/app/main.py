import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.database import init_db
from app.core.model_registry import registry
from app.routers import alerts, health, metrics, predict, stream, forecast, topology
# Backward-compatibility alias
graph = topology

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VIGIL-Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan: Preloads model registry and initializes SQLite schema on startup.
    """
    logger.info("Initializing VIGIL Backend Core Services...")
    await init_db()
    registry.load_artifacts()
    logger.info("VIGIL Backend Engine Online and Ready.")
    yield
    logger.info("VIGIL Backend Engine Shutting Down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Explainable AI Network Intrusion Detection System (XAI-NIDS) REST & WebSocket Backend",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(stream.router)
    app.include_router(alerts.router)
    app.include_router(metrics.router)
    app.include_router(forecast.router)
    app.include_router(topology.router)

    # Mount Static Files & Frontend Assets
    frontend_dir = settings.STATIC_DIR
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static_assets")
        js_dir = frontend_dir / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js_assets")

    @app.get("/dashboard", tags=["Dashboard"])
    @app.get("/console", tags=["Dashboard"])
    async def dashboard_view():
        if frontend_dir.exists():
            console_file = frontend_dir / "console.html"
            if console_file.exists():
                return FileResponse(str(console_file))
            dashboard_file = frontend_dir / "dashboard.html"
            if dashboard_file.exists():
                return FileResponse(str(dashboard_file))
        return JSONResponse({"error": "Console template not found"}, status_code=404)

    @app.get("/", tags=["Root"])
    async def root(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" in accept and frontend_dir.exists():
            index_file = frontend_dir / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "online",
            "home_url": "/",
            "dashboard_url": "/dashboard",
            "docs_url": "/docs",
            "endpoints": [
                "/health",
                "/predict/single",
                "/predict/batch",
                "/ws/stream",
                "/stream/replay",
                "/stream/inject",
                "/alerts",
                "/metrics/models",
                "/forecast",
                "/topology",
            ],
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
