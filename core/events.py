# Run 1 pass without calibration run

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple,Optional

# ─── 1. Core event types ──────────────────────────────────────────────────────

@dataclass
class Event:
    """
    Base class for all events in the system.
    timestamp: float (seconds since epoch)
    """
    timestamp: float

@dataclass
class MarketUpdate(Event):
    """
    A market data update for a specific symbol.
    security: the ticker symbol, e.g. "BTCUSDT"
    """
    security: str

@dataclass
class PRC(MarketUpdate):
    """
    A trade tick: (price, volume).
    Example: PRC(timestamp, "BTCUSDT", price=10800.0, volume=0.005)
    """
    price:  float
    volume: float

@dataclass
class BOOK(MarketUpdate):
    """
    Full-depth order-book snapshot.
    bids / asks are lists of (price, size), already sorted best→worst.
    mid is computed as (best_bid + best_ask) / 2.
    """
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    mid:  float = field(init=False)

    def __post_init__(self):
        # compute top-of-book mid
        self.mid = 0.5 * (self.bids[0][0] + self.asks[0][0])


@dataclass
class BAR(MarketUpdate):
    """
    An OHLC bar: (O, H, L, C).
    bodyfill = True if candle is down (C < O), else False.

    We also declare the extra fields your CSV loader will
    attach so they exist on the dataclass by default.
    """
    O: float
    H: float
    L: float
    C: float

    # candle body-colour helper (computed in __post_init__)
    bodyfill: bool = field(init=False)

    # extra CSV columns (defaults allow loader to overwrite safely)
    volume:         float = 0.0
    close_time:     float = 0.0   # seconds
    quote_vol:      float = 0.0
    trades:         int   = 0
    taker_base_vol: float = 0.0
    taker_quote_vol:float = 0.0

    def __post_init__(self):
        # fill the boolean after O/H/L/C are set
        self.bodyfill = (self.C < self.O)


@dataclass
class TickBar(BAR):
    """Special BAR built from a fixed number of ticks."""
    numticks: int=0

@dataclass
class TimeBar(BAR):
    """Special BAR built over a fixed time interval (seconds)."""
    seconds: int=0


# ─── 2. Signalling events ────────────────────────────────────────────────────

@dataclass
class COMM(Event):
    """
    Text-based message between agents.
      timestamp: when it was generated
      sender:    who sent it (agent.name)
      text:      the carried message
    """
    sender: str
    text:   str

    def __init__(self, timestamp: float, sender: str, text: str):
        super().__init__(timestamp)
        self.sender = sender
        self.text   = text

#@dataclass
# class TradeFill(Event):
#     """
#     Signals that an (simulated) order got filled.
#       timestamp: when the fill occurred
#       price:     fill price
#       qty:       fill quantity (+ long, – short)
#     """
#     price: float
#     qty:   float
#     value: float
#
#     def __init__(self, timestamp: float, price: float, qty: float, value: float = 0.0):
#         super().__init__(timestamp)
#         self.price = price
#         self.qty   = qty
#         self.value = value

@dataclass
class TradeFill(Event):
    """
    Signals that an (simulated) order got filled.
    timestamp: when the fill occurred
    price:     fill price
    qty:       fill quantity (+ long, − short)
    """
    price: float
    qty: float
    value: float
    # NEW for calibration (optional):
    base_price: Optional[float] = None   # 执行时使用的基准价（open/close）
    adv_at_fill: Optional[float] = None  # 成交时刻的 ADV

    def __init__(
        self,
        timestamp: float,
        price: float,
        qty: float,
        value: float = 0.0,
        # NEW ↓↓↓
        base_price: Optional[float] = None, adv_at_fill: Optional[float] = None,
    ):
        super().__init__(timestamp)
        self.price = price
        self.qty   = qty
        self.value = value
        # NEW ↓↓↓
        self.base_price = base_price
        self.adv_at_fill = adv_at_fill


@dataclass
class InvUpdate(Event):
    """
    Broadcast of the current inventory level.
      timestamp: when the inventory was updated
      inventory: target or actual inventory level
    """
    position:        float
    inventory_value: float

    def __init__(self, timestamp: float, position: float, inventory_value: float):
        super().__init__(timestamp)
        self.position        = position
        self.inventory_value = inventory_value

# ─── 3. Reporting events ─────────────────────────────────────────────────────

@dataclass
class PartialExitReport(Event):
    """
    Non-terminal partial exit slice (e.g., partial take-profit/giveback).
    Used for detailed analytics; final performance still comes from TradeReport.
    """
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    qty: float                  # slice quantity closed (>0)
    pnl: float                  # realized PnL for this slice
    reason: str                 # take_profit1 / giveback / stop_loss (slice) ...
    trade_type: str             # "long" or "short"
    cash_after: float
    position_after: float
    slippage_cost: float = 0.0
    slippage_pct: float = 0.0

    def __init__(self, *,
                 entry_ts: float, exit_ts: float,
                 entry_price: float, exit_price: float,
                 qty: float, pnl: float,
                 reason: str, trade_type: str,
                 cash_after: float, position_after: float,
                 slippage_cost: float = 0.0, slippage_pct: float = 0.0):
        super().__init__(exit_ts)
        self.entry_ts = entry_ts
        self.exit_ts = exit_ts
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.qty = qty
        self.pnl = pnl
        self.reason = reason
        self.trade_type = trade_type
        self.cash_after = cash_after
        self.position_after = position_after
        self.slippage_cost = slippage_cost
        self.slippage_pct = slippage_cost

@dataclass
class TradeReport(Event):
    """
    Fired when a trade closes, carrying all the details.
      entry_ts, exit_ts, entry_price, exit_price, qty, pnl, return_pct,
      inventory_after (dollar), cash_after (dollar)
    """
    entry_ts:        float
    exit_ts:         float
    entry_price:     float
    exit_price:      float
    qty:             float
    pnl:             float
    return_pct:      float
    inventory_after: float
    cash_after:      float
    trade_volume:    float
    slippage_cost:   float
    slippage_pct:    float
    holding_time:    float = 0.0
    exit_reason:     str   = ""   # “stop_loss” / “giveback” / “take_profit” / “ema_flip” 等
    trade_type:      str   = ""   # "long" or "short"
    exit_price_vwap: float = 0.0  # 实际出口 VWAP（含滑点）审计
    # === 新增：三列成本（正式字段） ===
    #entry_slip_cost: float = 0.0  # 入场时的价格滑点现金（只统计开仓切片）
    quantity_effect_cost: float = 0.0  # 因“少拿到的数量”造成的损益流失（含多空对称）
    total_effect_cost: float = 0.0  # = entry_slip_cost + quantity_effect_cost

def __init__(
        self,
        timestamp: float,
        entry_ts: float,
        exit_ts: float,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        return_pct: float,
        inventory_after: float,
        cash_after: float,
        trade_volume: float,
        slippage_cost: float,
        slippage_pct: float,
        holding_time: float = 0.0,
        exit_reason: str = "",
        trade_type: str = "",
        exit_price_vwap: float = 0.0,
# 三列成本（可省略，默认 0）
        #entry_slip_cost: float = 0.0,
        quantity_effect_cost: float = 0.0,
        total_effect_cost: float = 0.0,

    ):
        super().__init__(timestamp)
        self.entry_ts = entry_ts
        self.exit_ts = exit_ts
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.qty = qty
        self.pnl = pnl
        self.return_pct = return_pct
        self.inventory_after = inventory_after
        self.cash_after = cash_after
        self.trade_volume = trade_volume
        self.slippage_cost = slippage_cost
        self.slippage_pct = slippage_pct
        self.holding_time = holding_time
        self.exit_reason = exit_reason
        self.trade_type = trade_type
        self.exit_price_vwap = exit_price_vwap
        # 三列成本
        #self.entry_slip_cost = entry_slip_cost
        self.quantity_effect_cost = quantity_effect_cost
        self.total_effect_cost = total_effect_cost


# @dataclass
# class TradeReport(Event):
#     """
#     Fired when a trade closes, carrying all the details.
#       entry_ts, exit_ts, entry_price, exit_price, qty, pnl, return_pct,
#       inventory_after (dollar), cash_after (dollar)
#     """
#     entry_ts:        float
#     exit_ts:         float
#     entry_price:     float
#     exit_price:      float
#     qty:             float
#     pnl:             float
#     return_pct:      float
#     inventory_after: float
#     cash_after:      float
#     trade_volume: float
#     slippage_cost: float
#     slippage_pct: float
#     holding_time: float = 0.0
#     exit_reason: str = ""  # “stop_loss” or “ema_flip”
#     trade_type: str = ""  # "long" or "short"
#     exit_price_vwap: float = 0.0  # VWAP of all exit slices (for audit)
#     # ==== NEW: no-slippage baseline & reconciliation fields ====
#     pnl_no_slip: float = 0.0
#     return_no_slip_pct: float = 0.0
#     delta_ret_pct: float = 0.0
#     entry_base_price: float = 0.0
#     exit_base_vwap: float = 0.0
#
#
#     def __init__(
#             self,
#             timestamp: float,
#             entry_ts: float,
#             exit_ts: float,
#             entry_price: float,
#             exit_price: float,
#             qty: float,
#             pnl: float,
#             return_pct: float,
#             inventory_after: float,
#             cash_after: float,
#             trade_volume: float,
#             slippage_cost: float,
#             slippage_pct: float,
#             holding_time: float = 0.0,
#             exit_reason: str = "", # “stop_loss” or “ema_flip”
#             trade_type: str = "",
#             exit_price_vwap: float = 0.0,
#             pnl_no_slip: float = 0.0,
#             return_no_slip_pct: float = 0.0,
#             delta_ret_pct: float = 0.0,
#             entry_base_price: float = 0.0,
#             exit_base_vwap: float = 0.0,
#
#     ):
#         super().__init__(timestamp)
#         self.entry_ts = entry_ts
#         self.exit_ts = exit_ts
#         self.entry_price = entry_price
#         self.exit_price = exit_price
#         self.qty = qty
#         self.pnl = pnl
#         self.return_pct = return_pct
#         self.inventory_after = inventory_after
#         self.cash_after = cash_after
#         self.trade_volume = trade_volume
#         self.slippage_cost = slippage_cost
#         self.slippage_pct = slippage_pct
#         self.holding_time = holding_time
#         self.exit_reason = exit_reason
#         self.trade_type = trade_type
#         self.exit_price_vwap = exit_price_vwap
#         self.pnl_no_slip = pnl_no_slip
#         self.return_no_slip_pct = return_no_slip_pct
#         self.delta_ret_pct = delta_ret_pct
#         self.entry_base_price = entry_base_price