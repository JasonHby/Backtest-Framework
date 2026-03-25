from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    strategy_name: str = Field(default="cta")
    portfolio_id: str = Field(default="main")
    data_path: str = Field(default="REST_api_data")
    start: Optional[str] = None
    end: Optional[str] = None
    starting_cash: float = Field(default=10_000.0, gt=0)
    use_calibration: bool = True
    strategy_params: Dict[str, object] = Field(default_factory=dict)


class BacktestSummary(BaseModel):
    final_cash: float
    total_pnl: float
    total_return_pct: float
    realized_return_pct: float
    max_drawdown_pct: float
    cagr: Optional[float] = None
    calmar: Optional[float] = None
    sharpe_ann: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate_pct: float
    avg_hold_hours: Optional[float] = None
    total_slip: Optional[float] = None
    avg_slip_pct: Optional[float] = None
    dd_multiple: Optional[float] = None
    n_trades: int
    n_partials: int
    calibrated_k: float


class EquityPoint(BaseModel):
    timestamp: float
    equity: float


class BacktestResultResponse(BaseModel):
    backtest_id: str
    status: str
    strategy_name: str
    portfolio_id: str
    summary: BacktestSummary
    trades: List[Dict[str, object]]
    partial_trades: List[Dict[str, object]]
    equity_curve: List[EquityPoint]
    signal_chart_url: Optional[str] = None
