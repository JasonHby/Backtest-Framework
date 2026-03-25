from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from Backtester.loader import load_events
from Backtester.recorder import Recorder
from Backtester.report import compute_metrics, export_artifacts
from core.calibration import CalibrationAgent
from core.executor import ExecutionAgent
from core.plumbing import run_simulation
from strategies.CTA import CTA


@dataclass
class BacktestRequestData:
    strategy_name: str = "cta"
    portfolio_id: str = "main"
    data_path: str = "REST_api_data"
    start: Optional[str] = None
    end: Optional[str] = None
    starting_cash: float = 10_000.0
    use_calibration: bool = True
    strategy_params: Dict[str, object] = field(default_factory=dict)


@dataclass
class BacktestSummaryData:
    final_cash: float
    total_pnl: float
    total_return_pct: float
    realized_return_pct: float
    max_drawdown_pct: float
    cagr: Optional[float]
    calmar: Optional[float]
    sharpe_ann: Optional[float]
    profit_factor: Optional[float]
    win_rate_pct: float
    avg_hold_hours: Optional[float]
    total_slip: Optional[float]
    avg_slip_pct: Optional[float]
    dd_multiple: Optional[float]
    n_trades: int
    n_partials: int
    calibrated_k: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "final_cash": self.final_cash,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "realized_return_pct": self.realized_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "cagr": self.cagr,
            "calmar": self.calmar,
            "sharpe_ann": self.sharpe_ann,
            "profit_factor": self.profit_factor,
            "win_rate_pct": self.win_rate_pct,
            "avg_hold_hours": self.avg_hold_hours,
            "total_slip": self.total_slip,
            "avg_slip_pct": self.avg_slip_pct,
            "dd_multiple": self.dd_multiple,
            "n_trades": self.n_trades,
            "n_partials": self.n_partials,
            "calibrated_k": self.calibrated_k,
        }


@dataclass
class BacktestRun:
    backtest_id: str
    status: str
    strategy_name: str
    portfolio_id: str
    summary: Dict[str, float]
    trades: List[Dict[str, object]]
    partial_trades: List[Dict[str, object]]
    equity_curve: List[Dict[str, float]]
    signal_chart_url: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "backtest_id": self.backtest_id,
            "status": self.status,
            "strategy_name": self.strategy_name,
            "portfolio_id": self.portfolio_id,
            "summary": self.summary,
            "trades": self.trades,
            "partial_trades": self.partial_trades,
            "equity_curve": self.equity_curve,
            "signal_chart_url": self.signal_chart_url,
        }


def _parse_iso_to_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()


@lru_cache(maxsize=8)
def _load_events_cached(data_path: str):
    return tuple(load_events(data_path))


def _filter_events(events, start_ts: Optional[float], end_ts: Optional[float]):
    out = []
    for evt in events:
        ts = float(getattr(evt, "timestamp", 0.0) or 0.0)
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(evt)
    return out


def _build_equity_curve(trades: List[Dict[str, object]], starting_cash: float) -> List[Dict[str, float]]:
    curve = [{"timestamp": 0.0, "equity": float(starting_cash)}]
    for trade in sorted(trades, key=lambda row: float(row.get("exit_ts", 0.0) or 0.0)):
        curve.append(
            {
                "timestamp": float(trade.get("exit_ts", 0.0) or 0.0),
                "equity": float(trade.get("cash_after", starting_cash) or starting_cash),
            }
        )
    return curve


def _sanitize_metric(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _compute_summary(
    trades: List[Dict[str, object]],
    partial_trades: List[Dict[str, object]],
    starting_cash: float,
    calibrated_k: float,
) -> BacktestSummaryData:
    final_cash = float(trades[-1]["cash_after"]) if trades else float(starting_cash)
    total_pnl = final_cash - float(starting_cash)
    metrics = compute_metrics(trades, starting_cash=starting_cash)
    return BacktestSummaryData(
        final_cash=final_cash,
        total_pnl=total_pnl,
        total_return_pct=float(metrics.get("realized_return_pct", 0.0) or 0.0),
        realized_return_pct=float(metrics.get("realized_return_pct", 0.0) or 0.0),
        max_drawdown_pct=float(metrics.get("max_dd_pct", 0.0) or 0.0),
        cagr=_sanitize_metric(metrics.get("cagr")),
        calmar=_sanitize_metric(metrics.get("calmar")),
        sharpe_ann=_sanitize_metric(metrics.get("sharpe_ann")),
        profit_factor=_sanitize_metric(metrics.get("profit_factor")),
        win_rate_pct=float(metrics.get("win_rate_pct", 0.0) or 0.0),
        avg_hold_hours=_sanitize_metric(metrics.get("avg_hold_hours")),
        total_slip=_sanitize_metric(metrics.get("total_slip")),
        avg_slip_pct=_sanitize_metric(metrics.get("avg_slip_pct")),
        dd_multiple=_sanitize_metric(metrics.get("dd_multiple")),
        n_trades=int(metrics.get("n_trades", 0) or 0),
        n_partials=len(partial_trades),
        calibrated_k=float(calibrated_k),
    )


class BacktestService:
    def __init__(self):
        self._runs: Dict[str, BacktestRun] = {}
        self.artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def create_backtest(self, request: BacktestRequestData) -> BacktestRun:
        if request.strategy_name != "cta":
            raise ValueError(f"Unsupported strategy: {request.strategy_name}")

        base_params = {
            "symbol": "BTCUSDT",
            "short": 50,
            "long": 336,
            "qty": None,
            "stop_atr": 2.5,
            "atr_len": 50,
            "allow_long": True,
            "allow_short": True,
            "take_profit_r1": 0.3,
            "take_profit_frac1": 0.5,
            "take_profit_r": 3.0,
            "breakeven_r": 0.5,
            "giveback_k": 4.0,
            "prefer_giveback": False,
            "portfolio_id": request.portfolio_id,
        }
        strategy_params = {**base_params, **request.strategy_params}

        events = list(_load_events_cached(request.data_path))
        start_ts = _parse_iso_to_ts(request.start)
        end_ts = _parse_iso_to_ts(request.end)
        filtered_events = _filter_events(events, start_ts, end_ts)
        if not filtered_events:
            raise ValueError("No events matched the selected time range.")

        calibrated_k = 0.0
        adv_map = {}
        if request.use_calibration:
            cta_cal = CTA(**strategy_params)
            exec_cal = ExecutionAgent(starting_cash=request.starting_cash, shortfall_coeff=0.0)
            calib = CalibrationAgent(bar_per_day=int(strategy_params["atr_len"]))
            run_simulation([cta_cal, exec_cal, calib], filtered_events)
            calibrated_k = calib.compute_k()
            adv_map = calib.get_adv_map()

        cta_real = CTA(**strategy_params)
        exec_real = ExecutionAgent(
            starting_cash=request.starting_cash,
            shortfall_coeff=calibrated_k,
            adv_map=adv_map,
        )
        rec = Recorder()
        run_simulation([cta_real, exec_real, rec], filtered_events)

        equity_curve = _build_equity_curve(rec.closed_trades, request.starting_cash)
        summary = _compute_summary(
            trades=rec.closed_trades,
            partial_trades=rec.partial_trades,
            starting_cash=request.starting_cash,
            calibrated_k=calibrated_k,
        )

        backtest_id = f"bt_{uuid4().hex[:10]}"
        artifact_dir = self.artifacts_root / backtest_id
        export_artifacts(
            outdir=artifact_dir,
            underlying=rec.underlying,
            trades=rec.closed_trades,
            partial_trades=rec.partial_trades,
            starting_cash=request.starting_cash,
            cta_params=strategy_params,
            k=calibrated_k,
            tag=request.portfolio_id,
            make_zip=False,
            save_signals=True,
        )
        signal_chart_path = artifact_dir / "signals.html"
        run = BacktestRun(
            backtest_id=backtest_id,
            status="completed",
            strategy_name=request.strategy_name,
            portfolio_id=request.portfolio_id,
            summary=summary.to_dict(),
            trades=rec.closed_trades,
            partial_trades=rec.partial_trades,
            equity_curve=equity_curve,
            signal_chart_url=(
                f"/artifacts/{backtest_id}/signals.html"
                if signal_chart_path.exists()
                else None
            ),
        )
        self._runs[backtest_id] = run
        return run

    def get_backtest(self, backtest_id: str) -> Optional[BacktestRun]:
        return self._runs.get(backtest_id)

    def list_backtests(self) -> List[BacktestRun]:
        return list(self._runs.values())


backtest_service = BacktestService()
