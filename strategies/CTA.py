# strategies/CTA.py

from __future__ import annotations
from collections import deque
from typing import Deque, List, Optional, Tuple

from core.fsm        import Agent
from core.events     import BAR, COMM, InvUpdate
from core.plumbing   import emit
from core.indicators import ema, true_range
from pathlib import Path
import json


class CTA(Agent):
    """
    CTA strategy: two-EMA crossover + ATR risk control + take-profit (partial / fixed / giveback).

    positions: one value per bar in {+1, 0, -1}
    qty=None => send "ALL" to size by full cash (handled in ExecutionAgent).
    """

    def __init__(
        self,
        symbol:   str,
        short:    int               = 20,
        long:     int               = 50,
        qty:      Optional[float]   = None,     # None => full-cash (“ALL”)
        stop_atr: float             = 2.5,
        atr_len:  int               = 24,
        allow_long: bool            = True,
        allow_short: bool           = True,

        # ---- TP / Breakeven / Giveback config ----
        take_profit_r:   Optional[float] = None,  # fixed full-exit take-profit threshold in R
        take_profit_r1:  Optional[float] = None,  # partial take-profit threshold in R (fires once)
        take_profit_frac1: float         = 0.5,   # partial fraction 0~1
        breakeven_r:     Optional[float] = None,  # arm breakeven after this many R
        giveback_k:      Optional[float] = None,  # full exit when drawdown from extreme >= k * ATR
        prefer_giveback: bool            = False, # True: giveback has priority over fixed TP
    ):
        super().__init__(name=f"CTA_{symbol}")

        # strategy parameters
        self.symbol       = symbol
        self.alpha_s      = 2.0 / (short + 1)
        self.alpha_l      = 2.0 / (long  + 1)
        self.short_period = short
        self.long_period  = long
        self.qty          = qty
        self.stop_atr     = stop_atr
        self.atr_len      = atr_len
        self.allow_long   = allow_long
        self.allow_short  = allow_short
        self.bar_count: int = 0

        # exits config
        self.take_profit_r     = take_profit_r
        self.take_profit_r1    = take_profit_r1
        self.take_profit_frac1 = take_profit_frac1
        self.breakeven_r       = breakeven_r
        self.giveback_k        = giveback_k
        self.prefer_giveback   = prefer_giveback

        # internal state
        self._tr_buf  : Deque[float] = deque(maxlen=atr_len)
        self.ema_s    = None
        self.ema_l    = None
        self.prev_diff= 0.0

        self.entry_price: Optional[float] = None
        self.stop_price : Optional[float] = None
        self.just_exited: bool = False

        # per-trade run-time vars
        self.initial_risk     : Optional[float] = None   # price distance R (= stop_atr * ATR_at_entry)
        self.high_since_entry : Optional[float] = None   # extreme for long
        self.low_since_entry  : Optional[float] = None   # extreme for short
        self.breakeven_armed  : bool = False             # if breakeven clamp is armed
        self.tp1_fired        : bool = False             # if partial TP already fired once
        self.current_units    : float = 0.0              # real-time position from InvUpdate

        # one entry per bar: +1 long, 0 flat, -1 short
        self.positions: List[int] = [0]

        # debug log
        self._debug_flips: list[dict] = []
        # "BUY" or "SELL" if we want to reverse next bar at open
        self.pending_reversal: Optional[str] = None
        print(f"[CTA.init] allow_long={allow_long} ({type(allow_long)}), allow_short={allow_short} ({type(allow_short)})")

    # ---------- helpers ----------
    def dump_debug(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(json.dumps(self._debug_flips, indent=2))

    # ---------- framework I/O ----------
    def observe(self, e) -> bool:
        # listen to BAR of this symbol, and InvUpdate (for current position)
        return (isinstance(e, BAR) and e.security == self.symbol) or isinstance(e, InvUpdate)

    def preprocess(self, e: BAR) -> None:
        if not isinstance(e, BAR):
            return
        self.bar_count += 1

        # 1) update ATR buffer
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

    def main(self, e) -> None:
        # consume position snapshot first (used for partial TP sizing)
        if isinstance(e, InvUpdate):
            self.current_units = e.position
            return

        # bar logic
        price = e.C
        pos   = self.positions[-1]
        diff  = (self.ema_s or price) - (self.ema_l or price)
        fast_ready = (self.bar_count >= self.short_period)
        slow_ready = (self.bar_count >= self.long_period)

        prior_pos = pos

        # update entry extremes for giveback exit
        if pos == 1:
            self.high_since_entry = max(self.high_since_entry or price, price)
        elif pos == -1:
            self.low_since_entry = min(self.low_since_entry or price, price)

        # ===== exit priority: ATR stop -> partial TP -> (giveback vs fixed TP) -> EMA flip =====
        exit_signal = False
        side: Optional[str] = None
        reason = ""

        # (1) ATR stop (with breakeven clamp)
        if pos != 0 and len(self._tr_buf) == self.atr_len:
            curr_atr = sum(self._tr_buf) / self.atr_len
            stop_price = self.entry_price - pos * self.stop_atr * curr_atr

            # breakeven clamp after armed
            if self.breakeven_armed:
                if pos == 1:
                    stop_price = max(stop_price, self.entry_price)
                else:
                    stop_price = min(stop_price, self.entry_price)

            self.stop_price = stop_price
            hit_sl = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
            if hit_sl:
                # === 区分 breakeven vs 正常 stop ===
                eps = 1e-9  # 浮点容差
                is_breakeven_exit = self.breakeven_armed and (abs(stop_price - self.entry_price) <= eps)
                reason = "breakeven" if is_breakeven_exit else "stop_loss"
                exit_signal, side = True, ("SELL" if pos == 1 else "BUY")
                #exit_signal, side, reason = True, ("SELL" if pos == 1 else "BUY"), "stop_loss"

        # (2) arm breakeven (only adjusts stop; does not exit)
        if (not exit_signal and pos != 0 and
            self.breakeven_r is not None and self.initial_risk is not None and
            not self.breakeven_armed):
            moved = (price - self.entry_price) if pos == 1 else (self.entry_price - price)
            if moved >= self.breakeven_r * self.initial_risk:
                self.breakeven_armed = True

        # (2.5) partial take-profit (non-terminal, only once)
        if (not exit_signal and pos != 0 and
            self.take_profit_r1 is not None and
            self.initial_risk is not None and
            not self.tp1_fired):
            moved = (price - self.entry_price) if pos == 1 else (self.entry_price - price)
            if moved >= self.take_profit_r1 * self.initial_risk:
                units = abs(self.current_units)
                if units > 0:
                    close_qty = units * float(self.take_profit_frac1)
                    side_p = "SELL" if pos == 1 else "BUY"
                    emit(COMM(e.timestamp, self.name, f"{side_p} {close_qty} #reason:take_profit1"))
                    self.tp1_fired = True

        # (3) giveback vs fixed full exit order
        def try_giveback() -> Tuple[bool, Optional[str], str]:
            if (pos != 0 and self.giveback_k is not None and
                len(self._tr_buf) == self.atr_len and
                (self.high_since_entry is not None or self.low_since_entry is not None)):
                curr_atr = sum(self._tr_buf) / self.atr_len
                retr = (self.high_since_entry - price) if pos == 1 else (price - self.low_since_entry)
                print(f"[CTA.giveback?] pos={pos} entry={self.entry_price} "
                      f"hi={self.high_since_entry} lo={self.low_since_entry} "
                      f"price={price} retr={retr:.4f} kATR={self.giveback_k * curr_atr:.4f}")
                if retr >= self.giveback_k * curr_atr:
                    return True, ("SELL" if pos == 1 else "BUY"), "giveback"
            return False, None, ""

        def try_take_profit_full() -> Tuple[bool, Optional[str], str]:
            if (pos != 0 and self.take_profit_r is not None and self.initial_risk is not None):
                moved = (price - self.entry_price) if pos == 1 else (self.entry_price - price)
                if moved >= self.take_profit_r * self.initial_risk:
                    return True, ("SELL" if pos == 1 else "BUY"), "take_profit"
            return False, None, ""

        if not exit_signal:
            if self.prefer_giveback:
                ok, s, r = try_giveback()
                if ok:
                    exit_signal, side, reason = ok, s, r
                else:
                    ok, s, r = try_take_profit_full()
                    if ok:
                        exit_signal, side, reason = ok, s, r
            else:
                ok, s, r = try_take_profit_full()
                if ok:
                    exit_signal, side, reason = ok, s, r
                else:
                    ok, s, r = try_giveback()
                    if ok:
                        exit_signal, side, reason = ok, s, r

        # (4) EMA flip (only if nothing above triggered)
        if not exit_signal and fast_ready and slow_ready:
            if pos == 1 and self.prev_diff > 0 > diff:
                exit_signal, side, reason = True, "SELL", "ema_flip"
                print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
            elif pos == -1 and self.prev_diff < 0 < diff:
                exit_signal, side, reason = True, "BUY", "ema_flip"
                print(f"[CTA.v] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")

        # ===== execute exit =====
        if exit_signal:
            record = {
                "timestamp": e.timestamp,
                "event": "exit_signal",
                "prior_pos": prior_pos,
                "prev_diff": self.prev_diff,
                "diff": diff,
                "reason": reason,
                "just_exited_before": self.just_exited,
                "side": side,
                "executed": False,
                "price": price,
            }

            if not self.just_exited:
                q = "ALL" if self.qty is None else self.qty
                emit(COMM(e.timestamp, self.name, f"{side} {q} #reason:{reason}"))
                self.positions.append(0)
                self.entry_price = self.stop_price = None

                # clear per-trade state
                self.initial_risk     = None
                self.high_since_entry = None
                self.low_since_entry  = None
                self.breakeven_armed  = False
                self.tp1_fired        = False

                self.just_exited = True
                self.prev_diff   = diff
                record["executed"] = True

                # schedule reversal next bar if exit was due to EMA flip
                if reason == "ema_flip":
                    if side == "SELL" and self.allow_short:
                        self.pending_reversal = "SELL"
                    elif side == "BUY" and self.allow_long:
                        self.pending_reversal = "BUY"

            self._debug_flips.append(record)

            if record["executed"]:
                return

        # ===== pending reversal: enter next bar at open =====
        if self.pending_reversal and pos == 0:
            entry_price = getattr(e, "O", e.C)
            reversal_side = self.pending_reversal
            q = "ALL" if self.qty is None else self.qty

            # log
            side_label = "long" if reversal_side == "BUY" else "short"
            self._debug_flips.append({
                "timestamp": e.timestamp,
                "event": "entry_signal",
                "side": side_label,
                "prev_diff": self.prev_diff,
                "diff": diff,
                "reason": "reverse_ema_flip",
                "via": "pending_reversal",
                "at_open": True,
            })

            # order
            emit(COMM(e.timestamp, self.name, f"{reversal_side} {q} #reason:reverse_ema_flip #at_open"))

            # internal book-keeping
            self.positions.append(+1 if reversal_side == "BUY" else -1)
            self.entry_price = entry_price

            # initialize ATR stop and R
            if len(self._tr_buf) == self.atr_len:
                curr_atr = sum(self._tr_buf) / self.atr_len
                self.stop_price = (entry_price - self.stop_atr * curr_atr) if reversal_side == "BUY" \
                                  else (entry_price + self.stop_atr * curr_atr)
                self.initial_risk = self.stop_atr * curr_atr
            else:
                self.stop_price   = None
                self.initial_risk = None

            # reset extremes / flags
            self.high_since_entry = entry_price
            self.low_since_entry  = entry_price
            self.breakeven_armed  = False
            self.tp1_fired        = False

            self.pending_reversal = None
            self.prev_diff = diff
            return

        # ===== entry: strict EMA crossover =====
        if pos == 0 and not self.just_exited and fast_ready and slow_ready:
            long_entry  = (self.prev_diff < 0 < diff) and self.allow_long
            short_entry = (self.prev_diff > 0 > diff) and self.allow_short

            if long_entry or short_entry:
                side_txt = "BUY" if long_entry else "SELL"
                self._debug_flips.append({
                    "timestamp": e.timestamp,
                    "event": "entry_signal",
                    "side": "long" if long_entry else "short",
                    "prev_diff": self.prev_diff,
                    "diff": diff,
                })
                print(f"[CTA.cross] {side_txt} @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f}")

                q = "ALL" if self.qty is None else self.qty
                emit(COMM(e.timestamp, self.name, f"{side_txt} {q} #reason:entry_cross"))
                self.positions.append(+1 if long_entry else -1)
                self.entry_price = price

                # initialize ATR stop and R
                if len(self._tr_buf) == self.atr_len:
                    curr_atr = sum(self._tr_buf) / self.atr_len
                    self.stop_price = (price - self.stop_atr * curr_atr) if long_entry \
                                      else (price + self.stop_atr * curr_atr)
                    self.initial_risk = self.stop_atr * curr_atr
                else:
                    self.stop_price   = None
                    self.initial_risk = None

                # initialize extremes / flags
                self.high_since_entry = price
                self.low_since_entry  = price
                self.breakeven_armed  = False
                self.tp1_fired        = False
            else:
                self.positions.append(0)
        else:
            # hold current position
            self.positions.append(pos)

        # store diff for next bar
        self.prev_diff = diff

    def postprocess(self, e: BAR) -> None:
        # allow exits again next bar
        self.just_exited = False

#No give back,only take profit
# from __future__ import annotations
# from collections import deque
# from typing import Deque, List, Optional
#
# from core.fsm        import Agent
# from core.events     import BAR, COMM
# from core.plumbing   import emit
# from core.indicators import ema, true_range
# from pathlib import Path
# import json
#
# class CTA(Agent):
#     """
#     CTA‐style two‐EMA crossover + ATR stop.
#
#     Positions history (self.positions) is a list of one element per bar:
#       +1  = long
#        0  = flat
#       -1  = short
#     If qty is None, the agent emits “ALL” for full‐cash sizing.
#     """
#     def __init__(
#         self,
#         symbol:   str,
#         short:    int               = 20,
#         long:     int               = 50,
#         qty:      Optional[float]   = None,   # None => full‐cash (“ALL”)
#         stop_atr: float             = 2.5,
#         atr_len:  int               = 24,
#         allow_long: bool = True ,  # new
#         allow_short: bool = True,  # new
#         take_profit_r: Optional[float] = None,  # NEW: take profit with coeff R
#         breakeven_r:  Optional[float] = None,  # NEW: breakeven clamp
#     ):
#         super().__init__(name=f"CTA_{symbol}")
#
#         # strategy parameters
#         self.symbol     = symbol
#         self.alpha_s    = 2.0 / (short + 1)
#         self.alpha_l    = 2.0 / (long  + 1)
#         self.short_period = short
#         self.long_period = long
#         self.qty        = qty
#         self.stop_atr   = stop_atr
#         self.atr_len    = atr_len
#         self.allow_long = allow_long
#         self.allow_short= allow_short
#         self.bar_count: int = 0
#         self.take_profit_r = take_profit_r
#         self.breakeven_r = breakeven_r
#
#         # internal state
#         self._tr_buf     : Deque[float] = deque(maxlen=atr_len)
#         self.ema_s       = None
#         self.ema_l       = None
#         self.prev_diff   = 0.0
#         self.entry_price = None
#         self.stop_price  = None
#         self.just_exited = False
#         # per-trade runtime
#         self.initial_risk: Optional[float] = None  # R 的价格距离（= stop_atr * ATR_entry）
#         self.high_since_entry: Optional[float] = None
#         self.low_since_entry: Optional[float] = None
#         self.breakeven_armed: bool = False
#
#
#         # one entry per bar: +1 long, 0 flat, -1 short
#         self.positions: List[int] = [0]
#         # Debugging log
#         self._debug_flips: list[dict] = []
#         # "BUY" or "SELL" if an EMA-flip reversal is queued
#         self.pending_reversal: Optional[str] = None
#
#     def dump_debug(self, path: str | Path) -> None:
#         """
#         Write the in-memory entry/exit debug trace to a JSON file.
#         """
#         p = Path(path)
#         p.write_text(json.dumps(self._debug_flips, indent=2))
#
#     def observe(self, e: BAR) -> bool:
#         return isinstance(e, BAR) and e.security == self.symbol
#
#     def preprocess(self, e: BAR) -> None:
#         self.bar_count += 1
#         # 1) ATR buffer
#         tr = true_range(e)
#         self._tr_buf.append(tr)
#
#         # 2) seed / update EMAs
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
#         fast_ready = (self.bar_count >= self.short_period)
#         slow_ready = (self.bar_count >= self.long_period)
#
#         # remember prior for this bar
#         prior_pos = pos
#
#         # initialize R
#         if pos!=0 and self.initial_risk is None and len(self._tr_buf) == self.atr_len:
#             curr_atr = sum(self._tr_buf) / self.atr_len
#             self.initial_risk = self.stop_atr * curr_atr
#
#         # Extreme value tracking after entry
#         if pos == 1:
#             self.high_since_entry = max(self.high_since_entry or price, price)
#         elif pos == -1:
#             self.low_since_entry = min(self.low_since_entry or price, price)
#
#         # # ─── 1) EXIT on strict EMA flip or ATR stop ──────────────────────────────
#         # exit_signal = False
#         # reason = ""
#         # if fast_ready and slow_ready and pos == 1 and self.prev_diff > 0 and diff < 0:
#         #     exit_signal = True
#         #     side = "SELL"
#         #     reason = "ema_flip"
#         #     print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         # elif fast_ready and slow_ready and pos == -1 and self.prev_diff < 0 and diff > 0:
#         #     exit_signal = True
#         #     side = "BUY"
#         #     reason = "ema_flip"
#         #     print(f"[CTA.v] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         # elif pos != 0 and len(self._tr_buf) == self.atr_len:
#         #     curr_atr   = sum(self._tr_buf) / self.atr_len
#         #     stop_price = self.entry_price - pos * self.stop_atr * curr_atr
#         #     self.stop_price = stop_price
#         #     hit = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
#         #     if hit:
#         #         exit_signal = True
#         #         side = "SELL" if pos == 1 else "BUY"
#         #         reason = "stop_loss"
#
#         # ─── 1) EXIT priority: ATR stop → take-profit → EMA flip ────────────────
#         exit_signal = False
#         reason = ""
#         side: Optional[str] = None
#
#         # (1) ATR 止损（含保本钳制）
#         if pos != 0 and len(self._tr_buf) == self.atr_len:
#             curr_atr = sum(self._tr_buf) / self.atr_len
#             stop_price = self.entry_price - pos * self.stop_atr * curr_atr
#         # 保本钳制：先有一定浮盈后，把 ATR 止损钳到开仓价以外
#             if self.breakeven_armed:
#                 if pos == 1:
#                     stop_price = max(stop_price, self.entry_price)
#                 else:
#                     stop_price = min(stop_price, self.entry_price)
#             self.stop_price = stop_price
#             hit_sl = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
#             if hit_sl:
#                 exit_signal, side, reason = True, ("SELL" if pos == 1 else "BUY"), "stop_loss"
#         # (2) 到达保本阈值则“武装保本”，仅修改止损位置，不平仓
#         if (not exit_signal and pos != 0 and
#             self.breakeven_r is not None and self.initial_risk is not None and
#             not self.breakeven_armed):
#             moved = (price - self.entry_price) if pos == 1 else (self.entry_price - price)
#             if moved >= self.breakeven_r * self.initial_risk:
#                 self.breakeven_armed = True
#         # (3) R 倍止盈（全平）
#         if (not exit_signal and pos != 0 and
#             self.take_profit_r is not None and self.initial_risk is not None):
#             moved = (price - self.entry_price) if pos == 1 else (self.entry_price - price)
#             if moved >= self.take_profit_r * self.initial_risk:
#                 exit_signal, side, reason = True, ("SELL" if pos == 1 else "BUY"), "take_profit"
#         # (4) EMA 翻转（仅在前面都未触发时考虑；维持你的反手逻辑）
#         if not exit_signal and fast_ready and slow_ready:
#             if pos == 1 and self.prev_diff > 0 > diff:
#                 exit_signal, side, reason = True, "SELL", "ema_flip"
#                 print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#             elif pos == -1 and self.prev_diff < 0 < diff:
#                 exit_signal, side, reason = True, "BUY", "ema_flip"
#                 print(f"[CTA.v] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#
#         if exit_signal:
#             # build base record of an eligible exit (whether executed or suppressed)
#             record = {
#                 "timestamp": e.timestamp,
#                 "event": "exit_signal",
#                 "prior_pos": prior_pos,
#                 "prev_diff": self.prev_diff,
#                 "diff": diff,
#                 "reason": reason,
#                 "just_exited_before": self.just_exited,
#                 "side": side,
#                 "executed": False,  # will flip if we actually do it
#                 "price": price,
#             }
#
#             if not self.just_exited:
#                 q = "ALL" if self.qty is None else self.qty
#                 cmd = f"{side} {q} #reason:{reason}"
#                 emit(COMM(e.timestamp, self.name, cmd))
#                 self.positions.append(0)
#                 self.entry_price = self.stop_price = None
#                 # 清理本笔新增状态
#                 self.initial_risk = None
#                 self.high_since_entry = None
#                 self.low_since_entry = None
#                 self.breakeven_armed = False
#                 self.just_exited = True
#                 self.prev_diff = diff
#                 record["executed"] = True
#
#                 # === schedule opposite reversal for next bar only if this was an EMA flip ===
#                 if reason == "ema_flip":
#                     if side == "SELL" and self.allow_short:
#                         # we were long, schedule a short reversal
#                         self.pending_reversal = "SELL"
#                     elif side == "BUY" and self.allow_long:
#                         # we were short, schedule a long reversal
#                         self.pending_reversal = "BUY"
#
#             # always append the record (eligible or executed)
#             self._debug_flips.append(record)
#
#             # if we executed the exit, bail out so we don't also take an entry this bar
#             if record["executed"]:
#                 return
#
#         # --- handle a pending EMA-flip reversal at next bar's open ---
#         if self.pending_reversal and pos == 0:
#             # determine open price; fallback to close if O is missing
#             entry_price = getattr(e, "O", e.C)
#             reversal_side = self.pending_reversal  # "BUY" or "SELL"
#             q = "ALL" if self.qty is None else self.qty
#
#             # record the reversal entry in debug flips
#             side_label = "long" if reversal_side == "BUY" else "short"
#             self._debug_flips.append({
#                 "timestamp": e.timestamp,
#                 "event": "entry_signal",
#                 "side": side_label,
#                 "prev_diff": self.prev_diff,
#                 "diff": diff,
#                 "reason": "reverse_ema_flip",
#                 "via": "pending_reversal",
#                 "at_open": True,
#             })
#
#             # emit the reversal order
#             emit(COMM(e.timestamp, self.name, f"{reversal_side} {q} #reason:reverse_ema_flip #at_open"))
#
#             # update internal state as if we entered on this bar's open
#             self.positions.append(+1 if reversal_side == "BUY" else -1)
#             self.entry_price = entry_price
#
#             # compute stop_price based on ATR, if ready
#             if len(self._tr_buf) == self.atr_len:
#                 curr_atr = sum(self._tr_buf) / self.atr_len
#                 if reversal_side == "BUY":
#                     self.stop_price = entry_price - self.stop_atr * curr_atr
#                 else:
#                     self.stop_price = entry_price + self.stop_atr * curr_atr
#             else:
#                 self.stop_price = None
#
#             # 初始化 R 与极值 & 取消保本标记
#             if len(self._tr_buf) == self.atr_len:
#                 # 这里可以直接复用上面的 curr_atr
#                 self.initial_risk = self.stop_atr * curr_atr
#             else:
#                 self.initial_risk = None
#             self.high_since_entry = self.entry_price
#             self.low_since_entry = self.entry_price
#             self.breakeven_armed = False
#
#             # clear pending reversal
#             self.pending_reversal = None
#
#             # ensure prev_diff gets updated for the next bar
#             self.prev_diff = diff
#             return
#
#         # ─── 2) ENTRY on strict EMA crossover when flat ─────────────────────────
#         if pos == 0 and not self.just_exited and fast_ready and slow_ready:
#             long_entry = (self.prev_diff < 0 < diff) and self.allow_long
#             short_entry = (self.prev_diff > 0 > diff) and self.allow_short
#             if long_entry:
#                 self._debug_flips.append({
#                     "timestamp": e.timestamp,
#                     "event": "entry_signal",
#                     "side": "long",
#                     "prev_diff": self.prev_diff,
#                     "diff": diff,
#                 })
#                 print(f"[CTA.cross] BUY @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f} (long allowed={self.allow_long})")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"BUY {q}"))
#                 self.positions.append(+1)
#                 self.entry_price = price
#                 self.stop_price  = (
#                     price - self.stop_atr * (sum(self._tr_buf) / self.atr_len)
#                     if len(self._tr_buf) == self.atr_len else None
#                 )
#                 # 初始化 R 与极值
#                 if len(self._tr_buf) == self.atr_len:
#                     curr_atr = sum(self._tr_buf) / self.atr_len
#                     self.initial_risk = self.stop_atr * curr_atr
#                 else:
#                     self.initial_risk = None
#                     self.high_since_entry = self.entry_price
#                     self.low_since_entry = self.entry_price
#                     self.breakeven_armed = False
#             elif short_entry:
#                 self._debug_flips.append({
#                     "timestamp": e.timestamp,
#                     "event": "entry_signal",
#                     "side": "short",
#                     "prev_diff": self.prev_diff,
#                     "diff": diff,
#                 })
#                 print(f"[CTA.cross] SELL @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f} (short allowed={self.allow_short})")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"SELL {q}"))
#                 self.positions.append(-1)
#                 self.entry_price = price
#                 self.stop_price  = (
#                     price + self.stop_atr * (sum(self._tr_buf) / self.atr_len)
#                     if len(self._tr_buf) == self.atr_len else None
#                 )
#                 # 初始化 R 与极值
#                 if len(self._tr_buf) == self.atr_len:
#                     curr_atr = sum(self._tr_buf) / self.atr_len
#                     self.initial_risk = self.stop_atr * curr_atr
#                 else:
#                     self.initial_risk = None
#                     self.high_since_entry = self.entry_price
#                     self.low_since_entry = self.entry_price
#                     self.breakeven_armed = False
#             else:
#                 self.positions.append(0)
#         else:
#             # ─── 3) HOLD current position ─────────────────────────────────────────
#             self.positions.append(pos)
#
#         # ─── 4) book‐keep diff for next bar ─────────────────────────────────────
#         self.prev_diff = diff
#
#     def postprocess(self, e: BAR) -> None:
#         # only clear just_exited at end of bar
#         self.just_exited = False


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
# from pathlib import Path
# import json
#
# class CTA(Agent):
#     """
#     CTA‐style two‐EMA crossover + ATR stop.
#
#     Positions history (self.positions) is a list of one element per bar:
#       +1  = long
#        0  = flat
#       -1  = short
#     If qty is None, the agent emits “ALL” for full‐cash sizing.
#     """
#     def __init__(
#         self,
#         symbol:   str,
#         short:    int               = 20,
#         long:     int               = 50,
#         qty:      Optional[float]   = None,   # None => full‐cash (“ALL”)
#         stop_atr: float             = 2.5,
#         atr_len:  int               = 24,
#         allow_long: bool = True,  # new
#         allow_short: bool = True,  # new
#     ):
#         super().__init__(name=f"CTA_{symbol}")
#
#         # strategy parameters
#         self.symbol   = symbol
#         self.alpha_s  = 2.0 / (short + 1)
#         self.alpha_l  = 2.0 / (long  + 1)
#         self.qty      = qty
#         self.stop_atr = stop_atr
#         self.atr_len  = atr_len
#         self.allow_long = allow_long
#         self.allow_short = allow_short
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
#         # one entry per bar: +1 long, 0 flat, -1 short
#         self.positions: List[int] = [0]
#         # Debugging log
#         self._debug_flips: list[dict] = []
#         print(f"[CTA init] allow_long={allow_long} allow_short={allow_short}")
#
#     # def set_mode(
#     #         self,
#     #         *,
#     #         allow_long: Optional[bool] = None,
#     #         allow_short: Optional[bool] = None,
#     # ) -> None:
#     #     """
#     #     Update entry gating at runtime.
#     #     Pass only the flags you want to change.
#     #     """
#     #     if allow_long is not None:
#     #         self.allow_long = allow_long
#     #     if allow_short is not None:
#     #         self.allow_short = allow_short
#     #
#     def dump_debug(self, path: str | Path) -> None:
#          """
#         Write the in-memory entry/exit debug trace to a JSON file.
#          """
#          p = Path(path)
#          p.write_text(json.dumps(self._debug_flips, indent=2))
#
#     def observe(self, e: BAR) -> bool:
#         return isinstance(e, BAR) and e.security == self.symbol
#
#     def preprocess(self, e: BAR) -> None:
#         # 1) ATR buffer
#         tr = true_range(e)
#         self._tr_buf.append(tr)
#
#         # 2) seed / update EMAs
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
#         # remember prior for this bar
#         prior_pos = pos
#
#         # ─── 1) EXIT on strict EMA flip or ATR stop ──────────────────────────────
#         exit_signal = False
#         reason = ""
#         if pos == 1 and self.prev_diff > 0 and diff < 0:
#             exit_signal = True
#             side = "SELL"
#             reason = "ema_flip"
#             print(f"[CTA.Rev] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         elif pos == -1 and self.prev_diff < 0 and diff > 0:
#             exit_signal = True
#             side = "BUY"
#             reason = "ema_flip"
#             print(f"[CTA.v] ts={e.timestamp} pos={pos} diff={diff:.4f} prev_diff={self.prev_diff:.4f}")
#         elif pos != 0 and len(self._tr_buf) == self.atr_len:
#             curr_atr   = sum(self._tr_buf) / self.atr_len
#             stop_price = self.entry_price - pos * self.stop_atr * curr_atr
#             self.stop_price = stop_price
#             hit = (pos == 1 and price <= stop_price) or (pos == -1 and price >= stop_price)
#             if hit:
#                 exit_signal = True
#                 side = "SELL" if pos == 1 else "BUY"
#                 reason = "stop_loss"
#
#         if exit_signal:
#             # build base record of an eligible exit (whether executed or suppressed)
#             record = {
#                 "timestamp": e.timestamp,
#                 "event": "exit_signal",
#                 "prior_pos": prior_pos,
#                 "prev_diff": self.prev_diff,
#                 "diff": diff,
#                 "reason": reason,
#                 "just_exited_before": self.just_exited,
#                 "side": side,
#                 "executed": False,  # will flip if we actually do it
#                 "price": price,
#             }
#
#             if not self.just_exited:
#                 q = "ALL" if self.qty is None else self.qty
#                 cmd = f"{side} {q} #reason:{reason}"
#                 emit(COMM(e.timestamp, self.name, cmd))
#                 self.positions.append(0)
#                 self.entry_price = self.stop_price = None
#                 self.just_exited = True
#                 self.prev_diff = diff
#                 record["executed"] = True
#
#             # always append the record (eligible or executed)
#             self._debug_flips.append(record)
#
#             # if we executed the exit, bail out so we don't also take an entry this bar
#             if record["executed"]:
#                 return
#         # ─── 2) ENTRY on strict EMA crossover when flat ─────────────────────────
#         if pos == 0 and not self.just_exited:
#             long_entry = (self.prev_diff < 0 < diff) and self.allow_long
#             short_entry = (self.prev_diff > 0 > diff) and self.allow_short
#             if long_entry:
#                 self._debug_flips.append({
#                     "timestamp": e.timestamp,
#                     "event": "entry_signal",
#                     "side": "long",
#                     "prev_diff": self.prev_diff,
#                     "diff": diff,
#                 })
#                 print(f"[CTA.cross] BUY @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f} (long allowed={self.allow_long})")
#                 q = "ALL" if self.qty is None else self.qty
#                 emit(COMM(e.timestamp, self.name, f"BUY {q}"))
#                 self.positions.append(+1)
#                 self.entry_price = price
#                 self.stop_price  = (
#                     price - self.stop_atr * (sum(self._tr_buf) / self.atr_len)
#                     if len(self._tr_buf) == self.atr_len else None
#                 )
#             elif short_entry:
#                 self._debug_flips.append({
#                     "timestamp": e.timestamp,
#                     "event": "entry_signal",
#                     "side": "short",
#                     "prev_diff": self.prev_diff,
#                     "diff": diff,
#                 })
#                 print(f"[CTA.cross] SELL @ {price:.2f} prev_diff={self.prev_diff:.4f} diff={diff:.4f} (short allowed={self.allow_short})")
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
#             # ─── 3) HOLD current position ─────────────────────────────────────────
#             self.positions.append(pos)
#
#         # ─── 4) book‐keep diff for next bar ─────────────────────────────────────
#         self.prev_diff = diff
#
#     def postprocess(self, e: BAR) -> None:
#         # only clear just_exited at end of bar
#         self.just_exited = False



# # strategies/CTA.py (minute k line)
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
