# backtester/recorder.py

import json
from typing import List, Tuple, Dict

from core.events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
from core.fsm      import Agent
from core.plumbing import emit


class Recorder(Agent):
    """
    1) On BAR:         snapshot underlying price for benchmark curve
    2) On TradeReport: store each trade's metrics (entry/exit ts, PnL, return_pct, etc.)
    """
    def __init__(self):
        super().__init__(name="RECORDER")

        # time-series of the underlying price (for buy-&-hold benchmark)
        self.underlying   : List[Tuple[float, float]] = []

        # per-trade records, each dict must include:
        #   entry_ts, exit_ts, entry_price, exit_price, qty, pnl, return_pct, inventory_after
        self.closed_trades: List[Dict[str, float]]   = []

    def observe(self, e) -> bool:
        # only care about BARs and our TradeReport events
        #return isinstance(e, (BAR, TradeReport))
        match = isinstance(e, (BAR, TradeReport))
        if match:
            #print(f"[Recorder.observe] saw {type(e).__name__} @ {getattr(e, 'timestamp', None)}")
            return match

    def main(self, e):
        # 1) build benchmark price series
        if isinstance(e, BAR):
            # append every bar
            self.underlying.append((e.timestamp, e.C))

            # once we’ve collected a few, print them for sanity‑checking
            if len(self.underlying) < 5:
                print(f"DBG BAR #{len(self.underlying)+1}: ts={e.timestamp}   price={e.C:.2f}")

            return  # don’t fall through to trade‑report logic

        # 2) record final trade metrics
        if isinstance(e, TradeReport):
            #print(f"[Recorder.main] recording TradeReport: entry={e.entry_price} exit={e.exit_price} qty={e.qty}")
            rec = {
                "entry_ts"       : e.entry_ts,
                "exit_ts"        : e.exit_ts,
                "entry_price"    : e.entry_price,
                "exit_price"     : e.exit_price,
                "qty"            : e.qty,
                "pnl"            : e.pnl,
                "return_pct"     : e.return_pct,
                "inventory_after": e.inventory_after,
                "cash_after": e.cash_after,  # ← new
                "inventory_delta": abs(e.qty),
                "inventory_held_in_trade": e.inventory_held_in_trade,
                # ← new
                # "adv_at_fill": e.adv_at_fill,
                # "part_rate": e.part_rate,
                # "slippage_pct": e.slippage_pct,
            }
            print("→ rec:", rec)
            self.closed_trades.append(rec)




# # backtester/recorder.py
#
# import json
# import matplotlib.pyplot as plt
# from typing import List, Tuple
#
# from core.events    import BAR, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
#
# class Recorder(Agent):
#     """
#     1) On BAR:            snapshot underlying price for market curve
#     2) On TradeFill:      update cash & inventory
#     3) On InvUpdate:      compute & record strategy equity
#     4) On TradeReport:    store per-trade metrics
#     """
#     def __init__(self, starting_cash: float = 0.0):
#         super().__init__(name="RECORDER")
#
#         # P/L bookkeeping
#         self.cash      = starting_cash
#         self.inventory       = 0.0
#
#         # time-series
#         self.underlying      : List[Tuple[float,float]] = []
#         self.equity_curve    : List[Tuple[float,float]] = []
#
#         # per-trade records
#         self.closed_trades   : List[dict] = []
#
#     def observe(self, e) -> bool:
#         return isinstance(e, (BAR, TradeFill, InvUpdate, TradeReport))
#
#     def main(self, e):
#         # 1) Underlying price: one point per BAR
#         if isinstance(e, BAR):
#             self.underlying.append((e.timestamp, e.C))
#             return
#
#         # 2) Fill: update cash & inventory
#         if isinstance(e, TradeFill):
#             # e.qty signed: positive for buy, negative for sell
#             self.cash      -= e.price * e.qty
#             self.inventory += e.qty
#             return
#
#         # 3) Inventory update: snapshot equity
#         if isinstance(e, InvUpdate):
#             last_price = self.underlying[-1][1]
#             equity     = self.cash + self.inventory * last_price
#             self.equity_curve.append((e.timestamp, equity))
#             return
#
#         # 4) Final trade report: append to closed_trades & re-emit if desired
#         if isinstance(e, TradeReport):
#             # store the raw dict
#             rec = {
#                 "entry_ts":        e.entry_ts,
#                 "exit_ts":         e.exit_ts,
#                 "entry_price":     e.entry_price,
#                 "exit_price":      e.exit_price,
#                 "qty":             e.qty,
#                 "pnl":             e.pnl,
#                 "return_pct":      e.return_pct,
#                 "inventory_after": e.inventory_after,
#             }
#             self.closed_trades.append(rec)
#
#             # (optionally) echo it to downstream reporters
#             emit(TradeReport(
#                 timestamp=e.timestamp,
#                 entry_ts=rec["entry_ts"],
#                 exit_ts=rec["exit_ts"],
#                 entry_price=rec["entry_price"],
#                 exit_price=rec["exit_price"],
#                 qty=rec["qty"],
#                 pnl=rec["pnl"],
#                 return_pct=rec["return_pct"],
#                 inventory_after=rec["inventory_after"],
#             ))
#
#     # (you can add helper methods here, e.g. to dump a chart, compute stats, etc.)