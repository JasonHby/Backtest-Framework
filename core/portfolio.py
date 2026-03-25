from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TradeState:
    entry_price: Optional[float] = None
    entry_ts: Optional[float] = None
    entry_qty: float = 0.0
    trade_vol: float = 0.0
    total_entry_qty: float = 0.0
    cash_at_entry: Optional[float] = None
    slip_accum: float = 0.0
    slip_pct_accum: float = 0.0
    slip_n: int = 0
    vwap_exit_num: float = 0.0
    vwap_exit_den: float = 0.0
    entry_base_price: Optional[float] = None
    lost_qty_open_signed: float = 0.0
    lost_qty_total_signed: float = 0.0
    qty_effect_cost_accum: float = 0.0

    def reset(self) -> None:
        self.entry_price = None
        self.entry_ts = None
        self.entry_qty = 0.0
        self.trade_vol = 0.0
        self.total_entry_qty = 0.0
        self.cash_at_entry = None
        self.slip_accum = 0.0
        self.slip_pct_accum = 0.0
        self.slip_n = 0
        self.vwap_exit_num = 0.0
        self.vwap_exit_den = 0.0
        self.entry_base_price = None
        self.lost_qty_open_signed = 0.0
        self.lost_qty_total_signed = 0.0
        self.qty_effect_cost_accum = 0.0


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    trades: Dict[str, TradeState] = field(default_factory=dict)

    def get_position(self, security: str) -> float:
        return float(self.positions.get(security, 0.0))

    def set_position(self, security: str, position: float) -> None:
        self.positions[security] = float(position)

    def get_trade_state(self, security: str) -> TradeState:
        if security not in self.trades:
            self.trades[security] = TradeState()
        return self.trades[security]

    def equity(self, security_prices: Dict[str, float]) -> float:
        inventory_value = sum(
            qty * float(security_prices.get(symbol, 0.0))
            for symbol, qty in self.positions.items()
        )
        return self.cash + inventory_value


class PortfolioManager:
    def __init__(self, starting_cash: float):
        self.starting_cash = float(starting_cash)
        self._portfolios: Dict[str, PortfolioState] = {}

    def get_portfolio(self, portfolio_id: str) -> PortfolioState:
        if portfolio_id not in self._portfolios:
            self._portfolios[portfolio_id] = PortfolioState(cash=self.starting_cash)
        return self._portfolios[portfolio_id]

    def all_portfolios(self) -> Dict[str, PortfolioState]:
        return self._portfolios
