from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.strategy import StrategyListResponse, StrategySpecResponse
from app.services.strategy_registry import get_strategy_spec, list_strategy_specs

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("", response_model=StrategyListResponse)
def list_strategies() -> StrategyListResponse:
    return StrategyListResponse(strategies=list_strategy_specs())


@router.get("/{strategy_name}", response_model=StrategySpecResponse)
def get_strategy(strategy_name: str) -> StrategySpecResponse:
    spec = get_strategy_spec(strategy_name)
    if spec is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return spec
