# Run 1 pass without calibration run

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple

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

@dataclass
class TradeFill(Event):
    """
    Signals that an (simulated) order got filled.
      timestamp: when the fill occurred
      price:     fill price
      qty:       fill quantity (+ long, – short)
    """
    price: float
    qty:   float
    value: float

    def __init__(self, timestamp: float, price: float, qty: float, value: float = 0.0):
        super().__init__(timestamp)
        self.price = price
        self.qty   = qty
        self.value = value
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
    # # ← ADD THESE THREE
    # adv_at_fill:     float  # ADV_i at the time of the fill
    # part_rate:       float  # signed_qty / ADV_i
    # slippage_pct:    float  # (fill_price / mid_price) - 1


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
            inventory_held_in_trade: float,
    # # # ← NEW fields for calibration:
    #          adv_at_fill:    float,    # Average daily volume at time of fill
    #          part_rate:      float,   # = qty / adv_at_fill
    #          slippage_pct:   float    # = (fill_price − mid_price) / mid_price
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
        self.inventory_held_in_trade = inventory_held_in_trade
        # self.adv_at_fill = adv_at_fill
        # self.part_rate = part_rate
        # self.slippage_pct = slippage_pct