# Backtester/recorder.py

from typing import List, Tuple, Dict
from core.events   import Event, BAR, TradeReport, PartialExitReport
from core.fsm      import Agent

class Recorder(Agent):
    """
    1) BAR:              记录底层价格（画基准/净值曲线）
    2) TradeReport:      记录每笔已平仓交易（含 no-slip 字段）
    3) PartialExitReport:记录部分平仓切片（审计用）
    """
    def __init__(self):
        super().__init__(name="RECORDER")
        self.underlying: List[Tuple[float, float]] = []
        self.closed_trades: List[Dict] = []
        self.partial_trades: List[Dict] = []

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, TradeReport, PartialExitReport))

    def main(self, e: Event) -> None:
        if isinstance(e, BAR):
            # 统一使用收盘价作基准
            px = float(getattr(e, "C", 0.0) or 0.0)
            self.underlying.append((float(e.timestamp), px))
            return

        if isinstance(e, PartialExitReport):
            rec = {
                "entry_ts": float(getattr(e, "entry_ts", 0.0) or 0.0),
                "exit_ts": float(getattr(e, "exit_ts", 0.0) or 0.0),
                "entry_price": float(getattr(e, "entry_price", 0.0) or 0.0),
                "exit_price": float(getattr(e, "exit_price", 0.0) or 0.0),
                "qty": float(getattr(e, "qty", 0.0) or 0.0),
                "pnl": float(getattr(e, "pnl", 0.0) or 0.0),
                "reason": getattr(e, "reason", "") or "",
                "trade_type": getattr(e, "trade_type", "") or "",
                "cash_after": float(getattr(e, "cash_after", 0.0) or 0.0),
                "position_after": float(getattr(e, "position_after", 0.0) or 0.0),
                "slippage_cost": float(getattr(e, "slippage_cost", 0.0) or 0.0),
                "slippage_pct": float(getattr(e, "slippage_pct", 0.0) or 0.0),
            }
            print(f"[REC.partial] {rec}")
            self.partial_trades.append(rec)
            return

        if isinstance(e, TradeReport):
            # 持有时长（秒）
            holding_s = max(0.0, float(getattr(e, "exit_ts", 0.0) or 0.0) -
                                   float(getattr(e, "entry_ts", 0.0) or 0.0))

            rec = {
                "entry_ts": float(getattr(e, "entry_ts", 0.0) or 0.0),
                "exit_ts": float(getattr(e, "exit_ts", 0.0) or 0.0),
                "entry_price": float(getattr(e, "entry_price", 0.0) or 0.0),
                "exit_price": float(getattr(e, "exit_price", 0.0) or 0.0),
                "exit_price_vwap": float(getattr(e, "exit_price_vwap", 0.0) or 0.0),
                "qty": float(getattr(e, "qty", 0.0) or 0.0),
                "pnl": float(getattr(e, "pnl", 0.0) or 0.0),
                "return_pct": float(getattr(e, "return_pct", 0.0) or 0.0),
                "inventory_after": float(getattr(e, "inventory_after", 0.0) or 0.0),
                "cash_after": float(getattr(e, "cash_after", 0.0) or 0.0),
                "trade_volume": float(getattr(e, "trade_volume", 0.0) or 0.0),
                "slippage_cost": float(getattr(e, "slippage_cost", 0.0) or 0.0),
                "slippage_pct": float(getattr(e, "slippage_pct", 0.0) or 0.0),
                "holding_time_s": holding_s,
                "exit_reason": getattr(e, "exit_reason", "") or "",
                "trade_type": getattr(e, "trade_type", "") or "",
            }

            # ===== baseline & reconciliation 字段 =====
            for k in (
                    #"pnl_no_slip",
                    #"return_no_slip_pct",
                    #"delta_ret_pct",
                    "entry_base_price",
                    "exit_base_vwap",
            ):
                if hasattr(e, k):
                    rec[k] = float(getattr(e, k) or 0.0)

            # ===== 价格效应 / 数量效应 / 总影响（若 dataclass 有这些字段就记录）=====
            for k in (
                    #"entry_slip_cost",
                    "quantity_effect_cost",
                    "total_effect_cost",
            ):
                if hasattr(e, k):
                    rec[k] = float(getattr(e, k) or 0.0)

            print(f"[REC.trade] {rec}")
            self.closed_trades.append(rec)


# # Backtester/recorder.py
#
# from typing import List, Tuple, Dict
# from datetime import datetime
#
# from core.events   import Event, BAR, TradeReport, PartialExitReport
# from core.fsm      import Agent
#
# class Recorder(Agent):
#     """
#     1) On BAR:         snapshot underlying price for benchmark curve
#     2) On TradeReport: store each closed trade's metrics
#     3) On PartialExit: store each partial-exit slice for analytics
#     """
#     def __init__(self):
#         super().__init__(name="RECORDER")
#         self.underlying   : List[Tuple[float, float]] = []
#         self.closed_trades: List[Dict[str, float]]    = []
#         self.partial_exits: List[Dict[str, float]]    = []
#
#     def observe(self, e) -> bool:
#         return isinstance(e, (BAR, TradeReport, PartialExitReport))
#
#     def main(self, e):
#         # 1) benchmark price series
#         if isinstance(e, BAR):
#             self.underlying.append((e.timestamp, e.C))
#             return
#
#         # 2) partial exit slices
#         if isinstance(e, PartialExitReport):
#             rec = {
#                 "entry_ts": e.entry_ts,
#                 "exit_ts": e.exit_ts,
#                 "trade_type": e.trade_type,
#                 "entry_price": e.entry_price,
#                 "exit_price": e.exit_price,     # real slice price
#                 "qty": e.qty,
#                 "pnl": e.pnl,
#                 "reason": e.reason,
#                 "cash_after": e.cash_after,
#                 "position_after": e.position_after,
#                 "slippage_cost": e.slippage_cost,
#             }
#             self.partial_exits.append(rec)
#             return
#
#         # 3) final closed trades
#         if isinstance(e, TradeReport):
#             raw_diff = e.exit_ts - e.entry_ts
#             holding_secs = (raw_diff.total_seconds() if hasattr(raw_diff, "total_seconds") else raw_diff)
#             rec = {
#                 "entry_ts": e.entry_ts,
#                 "exit_ts": e.exit_ts,
#                 "trade_type": e.trade_type,
#                 "entry_price": e.entry_price,
#                 "exit_price": e.exit_price,             # last real exit price
#                 "exit_price_vwap": e.exit_price_vwap,   # VWAP over slices
#                 "qty": e.qty,
#                 "pnl": e.pnl,
#                 "return_pct": e.return_pct,
#                 "inventory_after": e.inventory_after,
#                 "cash_after": e.cash_after,
#                 "holding_time_s": holding_secs,
#                 "exit_reason": e.exit_reason,
#                 "inventory_delta": abs(e.qty),
#                 "trade_volume": e.trade_volume,
#                 "slippage_cost": e.slippage_cost,
#                 "slippage_pct": e.slippage_pct,
#             }
#             self.closed_trades.append(rec)


# # backtester/recorder.py
#
# import json
# from typing import List, Tuple, Dict
#
# from core.events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm      import Agent
# from core.plumbing import emit
#
#
# class Recorder(Agent):
#     """
#     1) On BAR:         snapshot underlying price for benchmark curve
#     2) On TradeReport: store each trade's metrics (entry/exit ts, PnL, return_pct, etc.)
#     """
#     def __init__(self):
#         super().__init__(name="RECORDER")
#
#         # time-series of the underlying price (for buy-&-hold benchmark)
#         self.underlying   : List[Tuple[float, float]] = []
#
#         # per-trade records, each dict must include:
#         #   entry_ts, exit_ts, entry_price, exit_price, qty, pnl, return_pct, inventory_after
#         self.closed_trades: List[Dict[str, float]]   = []
#         # NEW: counters for frequency & holding time
#         self.num_trades = 0
#         self.total_holding_secs = 0.0
#
#     # def notify(self, event: Event) -> None:
#     #     # only act on TradeReport events
#     #     if isinstance(event, TradeReport):
#     #         # compute holding time in seconds
#     #         raw_diff = event.exit_ts - event.entry_ts
#     #         holding_secs = (raw_diff.total_seconds() if hasattr(raw_diff, "total_seconds") else raw_diff)
#     #         # attach to the report
#     #         event.holding_time = holding_secs
#     #
#     #         # update counters
#     #         self.num_trades += 1
#     #         self.total_holding_secs += holding_secs
#
#
#     def observe(self, e) -> bool:
#         # only care about BARs and our TradeReport events
#         #return isinstance(e, (BAR, TradeReport))
#         match = isinstance(e, (BAR, TradeReport))
#         if match:
#             #print(f"[Recorder.observe] saw {type(e).__name__} @ {getattr(e, 'timestamp', None)}")
#             return match
#
#     def main(self, e):
#         # 1) build benchmark price series
#         if isinstance(e, BAR):
#             # append every bar
#             self.underlying.append((e.timestamp, e.C))
#
#             # once we’ve collected a few, print them for sanity‑checking
#             if len(self.underlying) < 5:
#                 print(f"DBG BAR #{len(self.underlying)+1}: ts={e.timestamp}   price={e.C:.2f}")
#
#             return  # don’t fall through to trade‑report logic
#
#         # 2) record final trade metrics
#         if isinstance(e, TradeReport):
#             #print(f"[Recorder.main] recording TradeReport: entry={e.entry_price} exit={e.exit_price} qty={e.qty}")
#             # 1) compute holding time in seconds directly here
#             raw_diff = e.exit_ts - e.entry_ts
#             holding_secs = (raw_diff.total_seconds()if hasattr(raw_diff, "total_seconds")else raw_diff)
#             # 2) update counters
#             self.num_trades += 1
#             self.total_holding_secs += holding_secs
#             rec = {
#                 "entry_ts"       : e.entry_ts,
#                 "exit_ts"        : e.exit_ts,
#                 "trade_type": e.trade_type,
#                 "entry_price"    : e.entry_price,
#                 "exit_price"     : e.exit_price,
#                 "qty"            : e.qty,
#                 "pnl"            : e.pnl,
#                 "return_pct"     : e.return_pct,
#                 "inventory_after": e.inventory_after,
#                 "cash_after": e.cash_after,  # ← new
#                 "holding_time_s": holding_secs,
#                 "exit_reason": e.exit_reason,
#                 "inventory_delta": abs(e.qty),
#                 "trade_volume": e.trade_volume,
#                 "slippage_cost": e.slippage_cost,
#                 "slippage_pct": e.slippage_pct,
#                 # ← new
#                 # "adv_at_fill": e.adv_at_fill,
#                 # "part_rate": e.part_rate,
#                 # "slippage_pct": e.slippage_pct,
#             }
#             print("→ rec:", rec)
#             self.closed_trades.append(rec)

