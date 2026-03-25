from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.backtests import router as backtests_router
from app.api.routes.strategies import router as strategies_router

app = FastAPI(title="Backtest Framework", version="0.1.0")
app.include_router(strategies_router)
app.include_router(backtests_router)

_STATIC_DIR = Path(__file__).parent / "static"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=_ARTIFACTS_DIR), name="artifacts")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
