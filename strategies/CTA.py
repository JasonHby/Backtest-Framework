# strategies/CTA.py

from __future__ import annotations
from collections import deque
from typing import Deque, List, Optional

from core.fsm        import Agent
from core.events     import BAR, COMM
from core.plumbing   import emit
from core.indicators import ema, true_range

class CTA(Agent):
    """
    CTA‐style two‐EMA crossover + ATR stop.

    Positions history (self.positions) is a list of one element per bar:
      +1  = long
       0  = flat
      -1  = short
    If qty is None, the agent emits “ALL” for full‐cash sizing.
    """
    def __init__(
        self,
        symbol:   str,
        short:    int               = 20,
        long:     int               = 50,
        qty:      Optional[float]   = None,   # None => full‐cash (“ALL”)
        stop_atr: float             = 2.5,
        atr_len:  int               = 24
    ):
        super().__init__(name=f"CTA_{symbol}")

        # strategy parameters
        self.symbol   = symbol
        self.alpha_s  = 2.0 / (short + 1)
        self.alpha_l  = 2.0 / (long  + 1)
        self.qty      = qty
        self.stop_atr = stop_atr
        self.atr_len  = atr_len

        # internal state
        self._tr_buf     : Deque[float] = deque(maxlen=atr_len)
        self.ema_s       = None
        self.ema_l       = None
        self.prev_diff   = 0.0
        self.entry_price = None
        self.stop_price  = None
        self.just_exited = False

        # one entry per bar: +1 long, 0 flat, -1 short
        self.positions: List[int] = [0]

    # def reset(self) -> None:
    #     # clear ATR buffer & EMA seeds
    #     self._tr_buf.clear()
    #     self.ema_s = None
    #     self.ema_l = None
    #     # clear crossover flags & position history
    #     self.prev_diff = 0.0
    #     self.entry_price = None
    #     self.stop_price = None
    #     self.just_exited = False
    #     self.positions = [0]

    def observe(self, e: BAR) -> bool:
        return isinstance(e, BAR) and e.security == self.symbol

    def preprocess(self, e: BAR) -> None:
        # 1) ATR buffer
        tr = true_range(e)
        self._tr_buf.append(tr)

        # 2) seed / update EMAs
        price = e.C
        if self.ema_s is None:
            self.ema_s = price
            self.ema_l = price
        else:
            self.ema_s = ema(self.ema_s, price, self.alpha_s)
            self.ema_l = ema(self.ema_l, price, self.alpha_l)

    def main(self, e: BAR) -> None:
        price = e.C
        pos   = self.positions[-1]
        diff  = (self.ema_s or price) - (self.ema_l or price)

        # ─── 1) EXIT on strict EMA flip or ATR stop ──────────────────────────────
        exit_signal = False
        if pos == 1 and self.prev_diff > 0 and diff < 0:
            exit_signal = True
            side = "SELL"
            print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
        elif pos == -1 and self.prev_diff < 0 and diff > 0:
            exit_signal = True
            side = "BUY"
            print(f"[CTA.v] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
        elif pos != 0 and len(self._tr_buf) == self.atr_len:
            curr_atr   = sum(self._tr_buf) / self.atr_len
            stop_price = self.entry_price - pos * self.stop_atr * curr_atr
            hit = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
            if hit:
                exit_signal = True
                side = "SELL" if pos == 1 else "BUY"

        if exit_signal and not self.just_exited:
            q = "ALL" if self.qty is None else self.qty
            emit(COMM(e.timestamp, self.name, f"{side} {q}"))
            self.positions.append(0)
            self.entry_price = self.stop_price = None
            self.just_exited = True
            self.prev_diff   = diff
            return

        # ─── 2) ENTRY on strict EMA crossover when flat ─────────────────────────
        if pos == 0 and not self.just_exited:
            if self.prev_diff < 0 < diff:
                print(f"[CTA.cross] BUY @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f}")
                q = "ALL" if self.qty is None else self.qty
                emit(COMM(e.timestamp, self.name, f"BUY {q}"))
                self.positions.append(+1)
                self.entry_price = price
                self.stop_price  = (
                    price - self.stop_atr * (sum(self._tr_buf) / self.atr_len)
                    if len(self._tr_buf) == self.atr_len else None
                )
            elif self.prev_diff > 0 > diff:
                print(f"[CTA.cross] SELL @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f}")
                q = "ALL" if self.qty is None else self.qty
                emit(COMM(e.timestamp, self.name, f"SELL {q}"))
                self.positions.append(-1)
                self.entry_price = price
                self.stop_price  = (
                    price + self.stop_atr * (sum(self._tr_buf) / self.atr_len)
                    if len(self._tr_buf) == self.atr_len else None
                )
            else:
                self.positions.append(0)
        else:
            # ─── 3) HOLD current position ─────────────────────────────────────────
            self.positions.append(pos)

        # ─── 4) book‐keep diff for next bar ─────────────────────────────────────
        self.prev_diff = diff

    def postprocess(self, e: BAR) -> None:
        # only clear just_exited at end of bar
        self.just_exited = False



# # strategies/CTA.py
#
# from __future__ import annotations
# from collections import deque
# from typing import Deque, List, Optional
#
# from core.fsm        import Agent
# from core.events     import BAR, COMM
# from core.plumbing   import emit
# from core.indicators import ema, true_range
#
# class CTA(Agent):
#     """
#     CTA‐style two‐EMA crossover + ATR stop.
#
#     Positions history (self.positions) is a list of one element per bar:
#       +1  = long
#        0  = flat
#       -1  = short
#     """
#     def __init__(
#         self,
#         symbol:   str,
#         short:    int              = 10,
#         long:     int              = 30,
#         qty:      float | None  = None,  # None → full‐cash ("ALL")
#         stop_atr: float            = 2.0,
#         atr_len:  int              = 14
#     ):
#         super().__init__(name=f"CTA_{symbol}")
#         # strategy parameters
#         self.symbol   = symbol
#         self.alpha_s  = 2.0 / (short + 1)
#         self.alpha_l  = 2.0 / (long  + 1)
#         self.qty      = qty
#         self.stop_atr = stop_atr
#         self.atr_len  = atr_len
#
#         # internal state
#         self._tr_buf     : Deque[float] = deque(maxlen=atr_len)
#         self.ema_s       = None
#         self.ema_l       = None
#         self.prev_diff   = 0.0
#         self.entry_price = None
#         self.stop_price  = None
#         self.just_exited = False
#
#         # position history: one entry per bar consumed
#         # +1 = long, 0 = flat, -1 = short
#         self.positions: List[int] = [0]
#
#     def observe(self, e: BAR) -> bool:
#         return isinstance(e, BAR) and e.security == self.symbol
#
#     def preprocess(self, e: BAR) -> None:
#         # update ATR buffer
#         tr = true_range(e)
#         self._tr_buf.append(tr)
#
#         # seed or update EMAs
#         price = e.C
#         if self.ema_s is None:
#             self.ema_s = price
#             self.ema_l = price
#         else:
#             self.ema_s = ema(self.ema_s, price, self.alpha_s)
#             self.ema_l = ema(self.ema_l, price, self.alpha_l)
#
#     def main(self, e: BAR) -> None:
#         price = e.C
#         pos   = self.positions[-1]
#         diff  = (self.ema_s or price) - (self.ema_l or price)
#
#         # 1) EXIT (strict EMA flip or ATR stop)
#         exit_signal = False
#         # strict reverse‐cross
#         if pos == 1 and self.prev_diff > 0 and diff < 0:
#             exit_signal = True
#             side = "SELL"
#             print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         elif pos == -1 and self.prev_diff < 0 and diff > 0:
#             exit_signal = True
#             side = "BUY"
#             print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         # ATR protective stop
#         elif pos != 0 and len(self._tr_buf) == self.atr_len:
#             curr_atr = sum(self._tr_buf) / self.atr_len
#             stop_price = self.entry_price - pos * self.stop_atr * curr_atr
#             hit = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
#             if hit:
#                 exit_signal = True
#                 side = "SELL" if pos == 1 else "BUY"
#
#         if exit_signal and not self.just_exited:
#             q = "ALL" if self.qty is None else self.qty
#             emit(COMM(e.timestamp, self.name, f"{side} {q}"))
#             self.positions.append(0)
#             self.entry_price = self.stop_price = None
#             self.just_exited = True
#             self.prev_diff   = diff
#             return
#
#         # 2) ENTRY on strict EMA crossover if flat
#         if pos == 0 and not self.just_exited:
#             if self.prev_diff < 0 and diff > 0:
#                 print(f"[CTA.cross] BUY @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f}")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"BUY {q}"))
#                 self.positions.append(+1)
#                 self.entry_price = price
#                 self.stop_price  = (
#                     price - self.stop_atr * (sum(self._tr_buf) / self.atr_len)
#                     if len(self._tr_buf) == self.atr_len else None
#                 )
#             elif self.prev_diff > 0 and diff < 0:
#                 print(f"[CTA.cross] SELL @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f}")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"SELL {q}"))
#                 self.positions.append(-1)
#                 self.entry_price = price
#                 self.stop_price  = (
#                     price + self.stop_atr * (sum(self._tr_buf) / self.atr_len)
#                     if len(self._tr_buf) == self.atr_len else None
#                 )
#             else:
#                 self.positions.append(0)
#         else:
#             # HOLD current position
#             self.positions.append(pos)
#
#         # 3) update prev_diff for next bar
#         self.prev_diff = diff
#
#     def postprocess(self, e: BAR) -> None:
#         # clear just_exited so next bar can re‑enter
#         self.just_exited = False




# from __future__ import annotations
# from collections import deque
# from typing import Deque, List
#
# from core.fsm        import Agent
# from core.events     import BAR, COMM
# from core.plumbing   import emit
# from core.indicators import ema, true_range
#
#
# class CTA(Agent):
#     """
#     CTA‐style two‐EMA crossover + ATR stop.
#
#     Positions history (self.positions) is a list of one element per bar:
#       +1  = long
#        0  = flat
#       -1  = short
#     """
#     def __init__(
#         self,
#         symbol:   str,
#         short:    int    = 10,
#         long:     int    = 30,
#         qty:      float  = 1.0,
#         stop_atr: float  = 2.0,
#         atr_len:  int    = 14
#     ):
#         super().__init__(name=f"CTA_{symbol}")
#
#         # strategy parameters
#         self.symbol   = symbol
#         self.alpha_s  = 2.0 / (short + 1)
#         self.alpha_l  = 2.0 / (long  + 1)
#         self.qty = qty
#         self.stop_atr = stop_atr
#         self.atr_len  = atr_len
#
#         # internal state
#         self._tr_buf    : Deque[float] = deque(maxlen=atr_len)
#         self.ema_s      = None
#         self.ema_l      = None
#         self.prev_diff  = 0.0
#         self.entry_price = None
#         self.stop_price  = None
#         self.just_exited = False
#
#         # FSM scaffolding (unused transitions)
#         self.states  = ["INIT", "LONG", "SHORT"]
#         self.current = "INIT"
#
#         # position history: one entry per bar consumed
#         # +1 = long, 0 = flat, -1 = short
#         self.positions: List[int] = []
#         self.positions.append(0)  # start flat
#
#     def observe(self, e: BAR) -> bool:
#         return isinstance(e, BAR) and e.security == self.symbol
#
#     def preprocess(self, e: BAR) -> None:
#         # 1) Build ATR buffer
#         # DEBUG → confirm we’re actually getting every BAR
#         #print(f"[CTA] saw BAR for {e.security} @ {e.C:.2f} (ts={e.timestamp})")
#         tr = true_range(e)
#         self._tr_buf.append(tr)
#         #print(f"[CTA.preprocess] saw BAR for {e.security} @ {e.C:.2f} (ts={e.timestamp})")
#
#         # 2) Update EMAs
#         price = e.C
#         if self.ema_s is None:
#             # seed on first bar
#             self.ema_s = price
#             self.ema_l = price
#         else:
#             self.ema_s = ema(self.ema_s, price, self.alpha_s)
#             self.ema_l = ema(self.ema_l, price, self.alpha_l)
#             #print(f"[CTA]    ema_s={self.ema_s:.2f}, ema_l={self.ema_l:.2f}, diff={(self.ema_s - self.ema_l):.4f}")
#
#     def main(self, e: BAR) -> None:
#         price = e.C
#         pos   = self.positions[-1]
#         diff  = (self.ema_s or price) - (self.ema_l or price)
#
#         # 1) EXIT (reverse‐cross OR ATR stop), only if you're in a trade
#         exit_signal = False
#         #print(f"[CTA.main] ts={e.timestamp}  pos={pos}  diff={diff:.4f}  prev_diff={self.prev_diff:.4f}  just_exited={self.just_exited}")
#         # a) reverse‐cross (strict flip)
#         if pos == 1 and self.prev_diff > 0 >diff:
#             exit_signal = True
#             side = "SELL"
#             print(f"[CTA.Rev] ts={e.timestamp}  pos={pos}  diff={diff:.4f}  prev_diff={self.prev_diff:.4f}  just_exited={self.just_exited}")
#         elif pos == -1 and self.prev_diff < 0 <diff:
#             exit_signal = True
#             side = "BUY"
#             print(f"[CTA.Rev] ts={e.timestamp}  pos={pos}  diff={diff:.4f}  prev_diff={self.prev_diff:.4f}  just_exited={self.just_exited}")
#
#         # b) ATR protective stop
#         elif pos != 0 and len(self._tr_buf) == self.atr_len:
#             curr_atr   = sum(self._tr_buf) / self.atr_len
#             stop_price = self.entry_price - pos * self.stop_atr * curr_atr
#             hit = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
#             if hit:
#                 exit_signal = True
#                 side = "SELL" if pos == 1 else "BUY"
#
#         if exit_signal and not self.just_exited:
#             emit(COMM(e.timestamp, self.name, f"{side} {self.qty}"))
#             self.positions.append(0)
#             self.entry_price = self.stop_price = None
#             self.just_exited = True
#             self.prev_diff   = diff
#             return
#
#         # 2) ENTRY on EMA cross, only if flat and not just exited
#         if pos == 0 and not self.just_exited:
#             if self.prev_diff < 0 <diff:
#                 print(f"[CTA.cross] BUY @ {price:.2f}   prev_diff={self.prev_diff:.4f}  diff={diff:.4f}")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"BUY {q}"))
#                 self.positions.append(+1)
#                 self.entry_price = price
#                 # set your stop price
#                 self.stop_price  = price - self.stop_atr * (sum(self._tr_buf) / self.atr_len) \
#                                    if len(self._tr_buf) == self.atr_len else None
#
#
#             elif self.prev_diff > 0 >diff:
#                 print(f"[CTA.cross] SELL @ {price:.2f}   prev_diff={self.prev_diff:.4f}  diff={diff:.4f}")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"SELL {q}"))
#                 self.positions.append(-1)
#                 self.entry_price = price
#                 self.stop_price  = price + self.stop_atr * (sum(self._tr_buf) / self.atr_len) \
#                                    if len(self._tr_buf) == self.atr_len else None
#
#             else:
#                 # no signal, stay flat
#                 self.positions.append(0)
#
#         else:
#             # 3) HOLD whatever you had
#             self.positions.append(pos)
#
#         # 4) book‐keep for next bar
#         self.prev_diff = diff
#
#     def postprocess(self, e: BAR) -> None:
#         # only here do we clear the "just exited" flag,
#         # so that the *next* bar can see entries again
#         self.just_exited = False

    # def main(self, e: BAR) -> None:
    #     #print(f"[CTA.main] bar #{len(self.positions)}: price={e.C:.2f}, prev_diff={self.prev_diff:.4f}")
    #     # 0) clear the just_exited flag *at the start* of the next bar
    #     if self.just_exited:
    #         self.just_exited = False
    #
    #     price = e.C
    #     pos   = self.positions[-1]
    #     diff  = self.ema_s - self.ema_l
    #
    #     #print(f"[CTA.main] ts={e.timestamp}  pos={pos}  diff={diff:.4f}  prev_diff={self.prev_diff:.4f}  just_exited={self.just_exited}")
    #
    #     # print(
    #     #                f"[CTA.main]  ts={e.timestamp}  price={price:.2f}  "
    #     #     f"ema_s={self.ema_s:.2f}  ema_l={self.ema_l:.2f}  "
    #     #     f"diff={diff:.4f}  prev_diff={self.prev_diff:.4f}"
    #     #     )
    #
    #     # 1) EXIT on reverse cross FIRST (only if you’re in a trade)
    #     if pos == 1 and self.prev_diff > 0 >= diff:
    #         emit(COMM(e.timestamp, self.name, f"SELL {self.qty}"))
    #         self.positions.append(0)
    #         self.entry_price = self.stop_price = None
    #         self.just_exited = True
    #         # *do not* fall through to entry logic
    #         self.prev_diff = diff
    #         return
    #
    #     if pos == -1 and self.prev_diff < 0 <= diff:
    #         emit(COMM(e.timestamp, self.name, f"BUY {self.qty}"))
    #         self.positions.append(0)
    #         self.entry_price = self.stop_price = None
    #         self.just_exited = True
    #         self.prev_diff = diff
    #         return
    #
    #     # 2) ATR‐based protective stop
    #     curr_atr = None
    #     if len(self._tr_buf) == self.atr_len:
    #         curr_atr = sum(self._tr_buf) / self.atr_len
    #
    #     # --- 0) Protective stop (exit if hit) -------------------
    #     if not self.just_exited and pos != 0 and curr_atr is not None:
    #         hit_long = (
    #             pos == 1
    #             and self.stop_price is not None
    #             and price <= self.stop_price
    #         )
    #         hit_short = (
    #             pos == -1
    #             and self.stop_price is not None
    #             and price >= self.stop_price
    #         )
    #
    #         if hit_long or hit_short:
    #             # choose the opposite side to exit
    #             if pos == 1:
    #                 emit(COMM(e.timestamp, self.name, f"SELL {self.qty}"))
    #             else:
    #                 emit(COMM(e.timestamp, self.name, f"BUY {self.qty}"))
    #
    #             # now we’re flat
    #             self.positions.append(0)
    #             self.entry_price = self.stop_price = None
    #             self.just_exited = True
    #             return
    #
    #     # 3) ENTRY on EMA cross (only if flat and not just_exited)
    #     if pos == 0 and not self.just_exited:
    #         # bullish cross?
    #         if self.prev_diff <= 0 < diff:
    #             print(f"[CTA.cross] BUY @ {price:.2f}   prev_diff={self.prev_diff:.4f}  diff={diff:.4f}")
    #             emit(COMM(e.timestamp, self.name, f"BUY {self.qty}"))
    #             self.positions.append(+1)
    #             self.entry_price = price
    #             if curr_atr is not None:
    #                 self.stop_price = price - self.stop_atr * curr_atr
    #         # bearish cross?
    #         elif self.prev_diff >= 0 > diff:
    #             print(f"[CTA.cross] SELL @ {price:.2f}   prev_diff={self.prev_diff:.4f}  diff={diff:.4f}")
    #             emit(COMM(e.timestamp, self.name, f"SELL {self.qty}"))
    #             self.positions.append(-1)
    #             self.entry_price = price
    #             if curr_atr is not None:
    #                 self.stop_price = price + self.stop_atr * curr_atr
    #         else:
    #             # stay flat
    #             self.positions.append(0)
    #
    #     else:
    #         # 4) HOLD your current position
    #         self.positions.append(pos)
    #
    #     # remember for next bar
    #     self.prev_diff = diff
    #
    # def postprocess(self, e: BAR) -> None:
    #     # allow re-entry after one bar
    #     self.just_exited = False