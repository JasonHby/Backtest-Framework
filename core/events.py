from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["market", "next_open"]
QuantityType = Literal[
    "units",
    "percent_position",
    "all_position",
    "percent_cash",
    "all_cash",
    "target_position",
]


@dataclass
class Event:
    timestamp: float


@dataclass
class MarketUpdate(Event):
    security: str


@dataclass
class PRC(MarketUpdate):
    price: float
    volume: float


@dataclass
class BOOK(MarketUpdate):
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    mid: float = field(init=False)

    def __post_init__(self) -> None:
        self.mid = 0.5 * (self.bids[0][0] + self.asks[0][0])


@dataclass
class BAR(MarketUpdate):
    O: float
    H: float
    L: float
    C: float
    bodyfill: bool = field(init=False)
    volume: float = 0.0
    close_time: float = 0.0
    quote_vol: float = 0.0
    trades: int = 0
    taker_base_vol: float = 0.0
    taker_quote_vol: float = 0.0

    def __post_init__(self) -> None:
        self.bodyfill = self.C < self.O


@dataclass
class TickBar(BAR):
    numticks: int = 0


@dataclass
class TimeBar(BAR):
    seconds: int = 0


@dataclass
class COMM(Event):
    sender: str
    text: str

    def __init__(self, timestamp: float, sender: str, text: str):
        super().__init__(timestamp)
        self.sender = sender
        self.text = text


@dataclass
class OrderRequest(Event):
    sender: str
    portfolio_id: str
    security: str
    side: OrderSide
    quantity: Optional[float] = None
    quantity_type: QuantityType = "units"
    order_type: OrderType = "market"
    reduce_only: bool = False
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeFill(Event):
    price: float
    qty: float
    value: float
    portfolio_id: str = "main"
    security: str = ""
    side: str = ""
    reason: str = ""
    base_price: Optional[float] = None
    adv_at_fill: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvUpdate(Event):
    position: float
    inventory_value: float
    portfolio_id: str = "main"
    security: str = ""
    cash: float = 0.0
    equity: float = 0.0


@dataclass
class PartialExitReport(Event):
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str
    trade_type: str
    cash_after: float
    position_after: float
    portfolio_id: str = "main"
    security: str = ""
    slippage_cost: float = 0.0
    slippage_pct: float = 0.0


@dataclass
class TradeReport(Event):
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    inventory_after: float
    cash_after: float
    trade_volume: float
    slippage_cost: float
    slippage_pct: float
    portfolio_id: str = "main"
    security: str = ""
    holding_time: float = 0.0
    exit_reason: str = ""
    trade_type: str = ""
    exit_price_vwap: float = 0.0
    quantity_effect_cost: float = 0.0
    total_effect_cost: float = 0.0
