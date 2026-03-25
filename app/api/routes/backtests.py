from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.backtest import BacktestRequest, BacktestResultResponse
from app.services.backtest_service import BacktestRequestData, backtest_service

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


@router.post("", response_model=BacktestResultResponse)
def create_backtest(request: BacktestRequest) -> BacktestResultResponse:
    try:
        run = backtest_service.create_backtest(BacktestRequestData(**request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BacktestResultResponse(**run.to_dict())


@router.get("/{backtest_id}", response_model=BacktestResultResponse)
def get_backtest(backtest_id: str) -> BacktestResultResponse:
    run = backtest_service.get_backtest(backtest_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return BacktestResultResponse(**run.to_dict())
