# core/executor.py
# (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe,
#  close-intent hygiene, and price/quantity slippage costs only)

from typing import Dict, Optional
from core.events import (
    Event, BAR, COMM,
    TradeFill, InvUpdate, TradeReport, PartialExitReport,
)
from core.fsm import Agent
from core.plumbing import emit


def parse_comm_text(text: str):
    """
    Parse a COMM text like:
      "BUY ALL #reason:reverse_ema_flip #at_open"
    Returns (side, qty_text, reason, flags_set)
    """
    parts = text.strip().split()
    side = None
    qty_text = None
    reason = "manual"
    flags = set()

    non_flag = []
    for token in parts:
        if token.startswith("#"):
            if token.startswith("#reason:"):
                reason = token.split(":", 1)[1]
            else:
                flags.add(token.lstrip("#"))
        else:
            non_flag.append(token)

    if len(non_flag) >= 2:
        side = non_flag[0].upper()
        qty_text = non_flag[1]
    elif len(non_flag) == 1:
        side = non_flag[0].upper()
        qty_text = "ALL"
    else:
        raise ValueError(f"Cannot parse COMM text: '{text}'")

    return side, qty_text, reason, flags


class ExecutionAgent(Agent):
    """
    Turns COMM 'BUY qty' / 'SELL qty' into simulated fills + P/L.

    Slippage model (matches calibration):
        fill = base * (1 ± k * |qty| / ADV_t)
      - BUY uses +, SELL uses −
      - ρ = |qty| / ADV_t (non-negative)

    What we record (PER TRADE):
      • entry_slip_cost         : 仅入场切片的 |fill-base| * entry_abs 数量累加（现金口径）
      • quantity_effect_cost    : 分批出场时按“未结的 ΔQ（签名） * 本次基线单位收益”逐笔结算并累加
                                   —— 单位收益 = sign_entry * (base_exit - base_entry)
                                   —— ΔQ 为“无滑点应得数量 − 实际入场数量”的**签名**缺口；
                                      多头缺口 >0（买少了），空头缺口 <0（卖多了）
      • total_effect_cost       = entry_slip_cost + quantity_effect_cost
    """

    def __init__(
        self,
        starting_cash: float,
        shortfall_coeff: float = 0.0,
        adv_map: Optional[Dict[float, float]] = None,
    ):
        super().__init__(name="EXEC")

        # portfolio
        self.cash = float(starting_cash)
        self.position = 0.0

        # slippage params
        self.k = float(shortfall_coeff)
        self.adv_map = adv_map or {}

        # last-seen market data
        self.last_price: Optional[float] = None
        self.last_open: Optional[float] = None

        # trade-in-progress (actual)
        self.entry_price: Optional[float] = None
        self.entry_ts: Optional[float] = None
        self.entry_qty: float = 0.0            # sign of entry side (+ long / - short)
        self.trade_vol: float = 0.0            # Σ(fill * qty) for the open trade
        self._realized_pnl_accum: float = 0.0
        self._total_entry_qty: float = 0.0     # absolute open quantity (constant for the trade)
        self._cash_at_entry: Optional[float] = None

        # slippage stats (for display, not used in costs)
        self._slip_accum: float = 0.0          # Σ |fill-base| * |qty| (all fills)
        self._slip_pct_accum: float = 0.0
        self._slip_n: int = 0

        # VWAP audit of actual exit fills
        self._vwap_exit_num: float = 0.0       # Σ exit_abs * fill_price
        self._vwap_exit_den: float = 0.0       # Σ exit_abs

        # ===== Baseline refs used ONLY for quantity effect =====
        self._entry_base_price: Optional[float] = None  # base at entry (first entry bar)

        # ===== Three-column cost accumulators =====
        #self._entry_slip_cost_accum: float = 0.0     # 入场切片的下单滑点现金（绝对值）
        self._lost_qty_open_signed: float = 0.0      # 仍未结的“签名 ΔQ”（>0 多头买少了；<0 空头卖多了）
        self._lost_qty_total_signed: float = 0.0     # 累计 ΔQ（签名）仅审计用
        self._qty_effect_cost_accum: float = 0.0     # 分批出场逐笔结算并累加的数量效应现金

        # close-intent only-reduce-no-flip
        self._close_intent_reasons = {
            "stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"
        }

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, COMM))

    def main(self, e: Event) -> None:
        # 1) BAR
        if isinstance(e, BAR):
            self.last_price = float(getattr(e, "C", 0.0) or 0.0)
            self.last_open  = float(getattr(e, "O", 0.0) or 0.0) if hasattr(e, "O") else None
            return

        # 2) COMM
        if not isinstance(e, COMM):
            return

        # parse
        try:
            side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
        except ValueError as ve:
            print(f"[EXEC] malformed COMM '{e.text}': {ve}")
            return
        side_sign = 1 if side_txt.upper() == "BUY" else -1

        # base price choice
        use_open   = ("at_open" in flags)
        base_price = (self.last_open if (use_open and self.last_open) else self.last_price)
        if not base_price:
            print(f"[EXEC] skip COMM: no ref price")
            return

        # 3) sizing with close-intent hygiene
        fixed_cash = None
        old_pos    = self.position
        is_close   = (reason in self._close_intent_reasons)

        if qty_txt.upper() == "ALL":
            if is_close:
                if old_pos == 0:
                    return
                qty = abs(old_pos)
            else:
                if old_pos * side_sign < 0:
                    qty = abs(old_pos)                    # pure close
                else:
                    fixed_cash = self.cash                # ALL open with cash
                    qty = self.cash / base_price
        else:
            qty = float(qty_txt)
            if is_close:
                if old_pos == 0:
                    return
                if old_pos * side_sign >= 0:
                    side_sign = -1 if old_pos > 0 else 1  # force reduce direction
                qty = min(qty, abs(old_pos))
            else:
                if old_pos * side_sign < 0:
                    qty = min(qty, abs(old_pos))          # no flip in one shot

        signed_qty_req = side_sign * qty

        # 4) slippage → fill price (use requested qty for slip, then fix ALL cash)
        adv_i = self.adv_map.get(e.timestamp) if self.adv_map else None
        if self.k > 0 and adv_i and adv_i > 0 and signed_qty_req != 0:
            rho_abs   = abs(signed_qty_req) / adv_i
            slip_frac = self.k * rho_abs
            #slip_frac=0
            fill_price = base_price * (1.0 + (1.0 if signed_qty_req > 0 else -1.0) * slip_frac)
        else:
            fill_price = base_price

        # "ALL" open → recompute qty with slipped price (final qty!)
        if fixed_cash is not None:
            qty = fixed_cash / fill_price
            signed_qty_req = side_sign * qty

        # 5) per-fill slippage stats (use FINAL qty)
        slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty_req)
        self._slip_accum     += slip_cost_fill
        self._slip_pct_accum += abs((fill_price / base_price) - 1.0)
        self._slip_n         += 1

        trade_val   = signed_qty_req * fill_price
        prev_pos    = self.position

        # A) first open slice of a trade → init baselines & cost accumulators
        if self.entry_price is None and prev_pos == 0 and signed_qty_req != 0:
            self._cash_at_entry         = self.cash
            self._entry_base_price      = base_price
            self._vwap_exit_num         = 0.0
            self._vwap_exit_den         = 0.0
            # reset 3-column costs
            #self._entry_slip_cost_accum = 0.0
            self._lost_qty_open_signed  = 0.0
            self._lost_qty_total_signed = 0.0
            self._qty_effect_cost_accum = 0.0

        # B) does this fill reduce absolute position? (exit slice)
        reduced = 0.0
        if self.entry_price is not None and prev_pos != 0 and (prev_pos * signed_qty_req) < 0:
            reduced = min(abs(prev_pos), abs(signed_qty_req))
            if reduced > 0.0:
                # actual VWAP for audit (optional)
                self._vwap_exit_num += reduced * fill_price
                self._vwap_exit_den += reduced

                # === quantity effect settlement (per slice) ===
                # r = 当前这次减仓占“出场前持仓”的比例 ∈ (0,1]
                r = reduced / abs(prev_pos)
                before = self._lost_qty_open_signed  # 更新前未结 ΔQ（签名）
                # 本次应结 ΔQ（签名）
                dec = r * before
                # 本次基线“单位收益”（多空对称）： sign_entry * (base_exit - base_entry)
                sign_entry = 1.0 if (self.entry_qty > 0) else -1.0
                entry_base = self._entry_base_price if self._entry_base_price is not None else base_price
                unit_ret_cash_slice = sign_entry * (base_price - entry_base)
                # slice 数量效应现金
                qty_effect_slice = unit_ret_cash_slice * dec
                self._qty_effect_cost_accum += qty_effect_slice
                # 未结 ΔQ 递减
                self._lost_qty_open_signed -= dec
                after = self._lost_qty_open_signed
                print(
                    f"[QEFF.exit] reduced={reduced:.6f} prev_pos={prev_pos:.6f} r={r:.4f} "
                    f"dec={dec:+.6f} unit_ret={unit_ret_cash_slice:+.2f} "
                    f"slice_qcost={qty_effect_slice:+.2f} "
                    f"openΔQ: {before:+.6f} → {after:+.6f}"
                )

        # C) emit TradeFill for calibration audit
        emit(TradeFill(timestamp=e.timestamp, price=fill_price, qty=signed_qty_req,
                       value=trade_val, base_price=base_price, adv_at_fill=adv_i))

        # book
        self.cash     -= trade_val
        self.position += signed_qty_req
        new_pos        = self.position

        # D) entry-side slice? (absolute position increased)
        added_abs = max(0.0, abs(new_pos) - abs(prev_pos))
        if added_abs > 0.0:
            # 1) 入场切片的下单滑点现金（绝对）
            #self._entry_slip_cost_accum += abs(fill_price - base_price) * added_abs
            # 2) 更新“签名 ΔQ”：
            #    ΔQ_slice = (同样现金按 base 可得数量) − (实际数量)
            #              = (added_abs*fill/base) − added_abs = added_abs*(fill/base - 1)
            #    多头：fill>=base → ΔQ>0（买少了）
            #    空头：fill<=base → ΔQ<0（卖多了）
            before = self._lost_qty_open_signed
            delta_q_signed = added_abs * (fill_price / base_price - 1.0)

            if delta_q_signed != 0.0:
                self._lost_qty_open_signed  += delta_q_signed
                self._lost_qty_total_signed += delta_q_signed
            after = self._lost_qty_open_signed
            print(
                f"[QEFF.entry] base={base_price:.2f} fill={fill_price:.2f} "
                f"added={added_abs:.6f} dQ={delta_q_signed:+.6f} "
                f"openΔQ: {before:+.6f} → {after:+.6f} "
                f"(k={self.k}, adv={adv_i if adv_i is not None else 'None'})"
            )

        # E) overshoot: same fill closes old and opens new
        if prev_pos * signed_qty_req < 0 and abs(signed_qty_req) > abs(prev_pos):
            # remainder opens a new trade at this fill_price
            self.entry_price = fill_price
            self.entry_ts    = e.timestamp
            self.entry_qty   = self.position
            self.trade_vol   = fill_price * self.position
            self._total_entry_qty = abs(self.position)

            # new trade baselines
            self._cash_at_entry    = self.cash
            self._entry_base_price = base_price

            # reset costs for NEW trade
            #self._entry_slip_cost_accum = 0.0
            self._lost_qty_open_signed  = 0.0
            self._lost_qty_total_signed = 0.0
            self._qty_effect_cost_accum = 0.0

            # remainder qty that opens the new trade (带方向)
            remainder = signed_qty_req + prev_pos
            # 为“新交易”播种入场滑点累计：只记余量部分的切片
            self._slip_accum = (fill_price - base_price) * remainder
            self._slip_pct_accum = abs((fill_price / base_price) - 1.0)
            self._slip_n = 1

        # F) PartialExitReport (optional but useful)
        if reduced > 0 and self.position != 0:
            # 粗略的 per-slice realized（仅诊断用途）
            dir_sign = 1.0 if prev_pos > 0 else -1.0
            slice_realized = dir_sign * reduced * (fill_price - (self.entry_price or fill_price))
            emit(PartialExitReport(
                entry_ts=self.entry_ts,
                exit_ts=e.timestamp,
                entry_price=self.entry_price,
                exit_price=fill_price,
                qty=reduced,
                pnl=slice_realized,
                reason=reason,
                trade_type=("long" if prev_pos > 0 else "short"),
                cash_after=self.cash,
                position_after=self.position,
                slippage_cost=abs(fill_price - base_price) * reduced,
                slippage_pct=abs((fill_price / base_price) - 1.0) * 100.0,
            ))

        # G) clean new entry (no overshoot)
        if self.entry_price is None and signed_qty_req != 0 and self.position != 0:
            self.entry_price = fill_price
            self.entry_ts    = e.timestamp
            self.entry_qty   = signed_qty_req
            self.trade_vol   = fill_price * signed_qty_req
            self._total_entry_qty = abs(signed_qty_req)

            # reset 3-column costs for TRUE new trade
            self._entry_base_price      = base_price
            #self._entry_slip_cost_accum = 0.0
            # self._lost_qty_open_signed  = 0.0
            # self._lost_qty_total_signed = 0.0
            # self._qty_effect_cost_accum = 0.0

            # reset slip stats
            # self._slip_accum = 0.0
            # self._slip_pct_accum = 0.0
            # self._slip_n = 0

        # H) inventory snapshot
        emit(InvUpdate(timestamp=e.timestamp,
                       position=self.position,
                       inventory_value=self.position * (self.last_price or fill_price)))

        # I) fully closed → final TradeReport (NO recon; only three costs)
        if self.entry_price is not None and self.position == 0:
            pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
            denom_notional = abs(self.entry_price * self._total_entry_qty)
            ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0

            slippage_pct = (self._slip_pct_accum / self._slip_n) * 100.0 if self._slip_n else 0.0
            trade_type   = "long" if self.entry_qty > 0 else "short"

            exit_price_last = fill_price
            exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last

            #entry_slip_cost      = self._entry_slip_cost_accum
            quantity_effect_cost = self._qty_effect_cost_accum
            total_effect_cost    = self._slip_accum + quantity_effect_cost

            if abs(self._lost_qty_open_signed) > 1e-9:
                print(f"[WARN] openΔQ not fully settled at close: {self._lost_qty_open_signed:+.6f}")

            tr = TradeReport(
                timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
                entry_ts=self.entry_ts,
                exit_ts=e.timestamp,
                entry_price=self.entry_price,
                exit_price=exit_price_last,
                qty=abs(self._total_entry_qty),
                pnl=pnl,
                return_pct=ret_pct,
                inventory_after=0.0,
                cash_after=self.cash,
                trade_volume=self.trade_vol,
                slippage_cost=self._slip_accum,   # 总绝对滑点（进+出），仅展示
                slippage_pct=slippage_pct,
                exit_reason=reason,
                trade_type=trade_type,
                exit_price_vwap=exit_price_vwap,
                # 直接写入三列成本
                #entry_slip_cost=entry_slip_cost,
                quantity_effect_cost=quantity_effect_cost,
                total_effect_cost=total_effect_cost,
            )
            # # 三列成本挂到事件（Recorder 会读）
            # setattr(tr, "entry_slip_cost", float(entry_slip_cost))
            # setattr(tr, "quantity_effect_cost", float(quantity_effect_cost))
            # setattr(tr, "total_effect_cost", float(total_effect_cost))
            # print(f"[QEFF.final] entry_slip={self._entry_slip_cost_accum:.2f} ...")
            print(f"[QEFF.final] slip_sum={self._slip_accum:.2f} "
                  f"qty_effect={self._qty_effect_cost_accum:.2f} "
                  f"lost_Q_total={self._lost_qty_total_signed:.6f}")

            emit(tr)

            # reset for next trade
            self.entry_price = None
            self.entry_ts = None
            self.entry_qty = 0.0
            self.trade_vol = 0.0
            self._realized_pnl_accum = 0.0
            self._total_entry_qty = 0.0
            self._cash_at_entry   = None

            self._slip_accum = 0.0
            self._slip_pct_accum = 0.0
            self._slip_n = 0

            self._vwap_exit_num = 0.0
            self._vwap_exit_den = 0.0

            self._entry_base_price = None
            #self._entry_slip_cost_accum = 0.0
            self._lost_qty_open_signed  = 0.0
            self._lost_qty_total_signed = 0.0
            self._qty_effect_cost_accum = 0.0




# # core/executor.py
# # (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe,
# #  close-intent hygiene, and precise price/quantity slippage attribution)
#
# from typing import Dict, Optional
# from core.events import (
#     Event, BAR, COMM,
#     TradeFill, InvUpdate, TradeReport, PartialExitReport,
# )
# from core.fsm import Agent
# from core.plumbing import emit
#
#
# def parse_comm_text(text: str):
#     """
#     Parse a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L.
#
#     Slippage convention (matches calibration):
#         fill = base * (1 ± k * |qty| / ADV_t)
#       - BUY uses +, SELL uses −
#       - ρ = |qty| / ADV_t (non-negative)
#
#     Features
#     --------
#     • Supports '#at_open'
#     • Close-intent hygiene (stop_loss/giveback/take_profit/take_profit1/ema_flip → only reduce/close, no flip)
#     • Robust partial exits and VWAP audit (actual & baseline)
#     • Cash-path PnL settlement at close
#     • Overshoot-safe: same fill may close old and open new; resets per-trade stats correctly
#     • No-slip baseline (closed-form) for reconciliation
#     • Precise slippage attribution into price vs quantity effects (cash)
#     """
#
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None,
#     ):
#         super().__init__(name="EXEC")
#
#         # portfolio state
#         self.cash = float(starting_cash)
#         self.position = 0.0
#
#         # slippage params
#         self.k = float(shortfall_coeff)
#         self.adv_map = adv_map or {}
#
#         # last-seen market data
#         self.last_price: Optional[float] = None
#         self.last_open: Optional[float] = None
#         self.last_mid: Optional[float] = None
#
#         # trade-in-progress (actual)
#         self.entry_price: Optional[float] = None
#         self.entry_ts: Optional[float] = None
#         self.entry_qty: float = 0.0            # sign of entry side
#         self.trade_vol: float = 0.0            # Σ(fill * qty) for the open trade
#         self._realized_pnl_accum: float = 0.0  # realized partial exits (diagnostic)
#         self._total_entry_qty: float = 0.0     # absolute open quantity (constant for the trade)
#         self._cash_at_entry: Optional[float] = None
#
#         # slippage aggregation for the current trade
#         self._slip_accum: float = 0.0          # Σ |fill-base| × |qty|  (cash, absolute)
#         self._slip_pct_accum: float = 0.0      # Σ |(fill/base) - 1|
#         self._slip_n: int = 0                  # # of fills for this trade
#
#         # VWAP audit of actual exit fills
#         self._vwap_exit_num: float = 0.0       # Σ qty * fill_price
#         self._vwap_exit_den: float = 0.0       # Σ qty
#
#         # ===== No-slippage baseline (closed-form) =====
#         self._entry_base_price: Optional[float] = None   # base at entry
#         self._vwap_exit_base_num: float = 0.0            # Σ qty * base_price (exit slices)
#         self._vwap_exit_base_den: float = 0.0            # Σ qty
#
#         # ===== Precise slippage attribution (cash) =====
#         self._price_effect_signed: float = 0.0  # Σ qty * (fill - base)  — price effect (signed cash)
#         self._entry_cash_used: float = 0.0      # Σ |qty| * fill (only entry-side)
#         self._entry_qty_total: float = 0.0      # Σ |qty| (only entry-side)
#
#         # close-intent reasons (only reduce/close, never flip on same command)
#         self._close_intent_reasons = {
#             "stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"
#         }
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # 1) On BAR, update prices
#         if isinstance(e, BAR):
#             self.last_price = float(getattr(e, "C", 0.0) or 0.0)
#             self.last_open = float(getattr(e, "O", 0.0) or 0.0) if hasattr(e, "O") else None
#             self.last_mid = 0.5 * (float(getattr(e, "O", 0.0) or 0.0) + self.last_price)
#             return
#
#         # 2) Only handle COMM beyond here
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # reference price (open if #at_open is present and available)
#         use_open = "at_open" in flags
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#         if base_price in (None, 0.0):
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # 3) Determine quantity (ALLOW “ALL”), with close-intent hygiene
#         fixed_cash = None
#         is_close_intent = (reason in self._close_intent_reasons)
#         old_pos = self.position
#
#         if qty_txt.upper() == "ALL":
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#                     return
#                 qty = abs(old_pos)
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = abs(old_pos)  # pure close
#                 else:
#                     fixed_cash = self.cash
#                     qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (reason={reason}).")
#                     return
#                 if old_pos * side_sign >= 0:
#                     # force reduce direction
#                     side_sign = -1 if old_pos > 0 else 1
#                 qty = min(qty, abs(old_pos))
#             else:
#                 # reverse → clamp to pure close; no flip
#                 if old_pos * side_sign < 0:
#                     qty = min(qty, abs(old_pos))
#
#         signed_qty = side_sign * qty
#         # 4) Compute fill price with slippage — matches calibration
#         adv_i = self.adv_map.get(e.timestamp) if self.adv_map else None
#         if self.k > 0 and adv_i and adv_i > 0:
#             rho_abs = abs(signed_qty) / adv_i
#             slip_frac = self.k * rho_abs
#             #slip_frac = 0.0
#             fill_price = base_price * (1.0 + (1.0 if signed_qty > 0 else -1.0) * slip_frac)
#         else:
#             slip_frac = 0.0
#             fill_price = base_price
#
#         # If "ALL" entry with fixed cash, recompute qty with slipped price
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             print(f"[EXEC] adjusted {'BUY' if side_sign>0 else 'SELL'} ALL → qty={qty:.4f} after slip")
#
#         # === Per-fill slippage stats (use FINAL signed_qty!) ===
#         slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty)
#         slip_pct_fill = abs((fill_price / base_price) - 1.0)
#         self._slip_accum += slip_cost_fill
#         self._slip_pct_accum += slip_pct_fill
#         self._slip_n += 1
#         # precise price effect (signed cash)
#         self._price_effect_signed += signed_qty * (fill_price - base_price)
#
#         trade_val = signed_qty * fill_price
#
#         # A) Snapshot cash at entry BEFORE first opening fill
#         opening_trade = (self.entry_price is None and self.position == 0 and signed_qty != 0)
#         if opening_trade:
#             self._cash_at_entry = self.cash
#             # baseline — record base at entry
#             self._entry_base_price = base_price
#             # reset exit-base accumulators just in case
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#         # B) How much of this fill actually closes old position?
#         old_pos = self.position
#         reduced = 0.0
#         slice_realized = 0.0
#         slice_slip = 0.0
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = min(abs(old_pos), abs(signed_qty))
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0
#                 slice_realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += slice_realized
#                 # VWAP audit (actual)
#                 self._vwap_exit_num += reduced * fill_price
#                 self._vwap_exit_den += reduced
#                 # baseline VWAP (on base)
#                 self._vwap_exit_base_num += reduced * base_price
#                 self._vwap_exit_base_den += reduced
#                 # slice slippage in cash (for partial-exit report)
#                 slice_slip = abs(reduced * (fill_price - base_price))
#
#         # C) Book the fill (+ emit TradeFill with base/ADV for calibration audit)
#         emit(TradeFill(
#             timestamp=e.timestamp,
#             price=fill_price,
#             qty=signed_qty,
#             value=trade_val,
#             base_price=base_price,     # for Calibration
#             adv_at_fill=adv_i          # for Calibration
#         ))
#         self.cash -= trade_val
#         self.position += signed_qty
#         position_after = self.position
#
#         # entry-side portion (increased |position|) → record entry cash/qty
#         added_abs = max(0.0, abs(position_after) - abs(old_pos))
#         if added_abs > 0.0:
#             self._entry_cash_used += added_abs * fill_price
#             self._entry_qty_total += added_abs
#
#         # D) If this was a pure partial exit (still have position), emit slice report
#         if reduced > 0 and self.position != 0:
#             emit(PartialExitReport(
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=reduced,
#                 pnl=slice_realized,
#                 reason=reason,
#                 trade_type=("long" if old_pos > 0 else "short"),
#                 cash_after=self.cash,
#                 position_after=self.position,
#                 slippage_cost=slice_slip,
#                 slippage_pct=slip_pct_fill
#             ))
#
#         # E) Overshoot handling: same fill closes old and opens new side
#         if old_pos * signed_qty < 0 and abs(signed_qty) > abs(old_pos):
#             # remainder becomes a fresh entry at this fill_price
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = self.position
#             self.trade_vol = fill_price * self.position
#             self._total_entry_qty = abs(self.position)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # baseline accumulators for the NEW trade
#             self._cash_at_entry = self.cash
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slippage & attribution stats for the NEW trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#             self._price_effect_signed = 0.0
#             self._entry_cash_used = 0.0
#             self._entry_qty_total = 0.0
#
#         # inventory snapshot
#         emit(InvUpdate(timestamp=e.timestamp, position=self.position,
#                        inventory_value=self.position * (self.last_price or fill_price)))
#
#         # G) Mark a clean new entry (no overshoot)
#         if self.entry_price is None and signed_qty != 0 and self.position != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # baseline entry/base reset
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slip & attribution stats for a TRUE new trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#             self._price_effect_signed = 0.0
#             self._entry_cash_used = 0.0
#             self._entry_qty_total = 0.0
#
#         # H) Fully closed → emit TradeReport (cash-path PnL; keep last real exit price)
#         if self.entry_price is not None and self.position == 0:
#             # Authoritative PnL = cash delta over the full trade
#             pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
#
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#
#             # Slippage pct (avg of per-fill absolute slip %)
#             slippage_pct = (self._slip_pct_accum / self._slip_n) * 100.0 if self._slip_n else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#
#             # Actual exit price(s)
#             exit_price_last = fill_price
#             exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last
#
#             # ===== No-slip baseline close-out =====
#             exit_base_vwap = (self._vwap_exit_base_num / self._vwap_exit_base_den) \
#                              if self._vwap_exit_base_den > 0 else base_price
#             entry_base = self._entry_base_price if self._entry_base_price is not None else self.entry_price
#             sign = 1.0 if self.entry_qty > 0 else -1.0
#             pnl_no_slip = sign * abs(self._total_entry_qty) * (exit_base_vwap - entry_base)
#             ret_no_slip_pct = (pnl_no_slip / denom_notional) * 100.0 if denom_notional else 0.0
#             delta_ret_pct = ret_no_slip_pct - ret_pct
#
#             # ===== Precise attribution =====
#             price_effect_cash = self._price_effect_signed
#             Q0 = (self._entry_cash_used / entry_base) if entry_base else 0.0
#             Q1 = self._entry_qty_total
#             quantity_effect_cash = sign * (Q0 - Q1) * (exit_base_vwap - entry_base)
#             slippage_impact_total = price_effect_cash + quantity_effect_cash
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=exit_price_last,
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#                 exit_price_vwap=exit_price_vwap,
#                 # —— no-slip baseline ——
#                 pnl_no_slip=pnl_no_slip,
#                 return_no_slip_pct=ret_no_slip_pct,
#                 delta_ret_pct=delta_ret_pct,
#                 entry_base_price=entry_base,
#                 exit_base_vwap=exit_base_vwap,
#                 # —— 精确归因（直接写入，不再 setattr）——
#                 price_effect_cash=price_effect_cash,
#                 quantity_effect_cash=quantity_effect_cash,
#                 slippage_impact_total=slippage_impact_total,
#             )
#
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0
#             self._cash_at_entry = None
#
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#             self._entry_base_price = None
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             self._price_effect_signed = 0.0
#             self._entry_cash_used = 0.0
#             self._entry_qty_total = 0.0


# # path: core/executor.py
# # (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe, no-slip baseline)
#
# from typing import Dict, Optional
# from core.events import (
#     Event, BAR, COMM,
#     TradeFill, InvUpdate, TradeReport, PartialExitReport
# )
#     # 如果你的导入路径不同，以你的工程为准
# from core.fsm import Agent
# from core.plumbing import emit
#
#
# def parse_comm_text(text: str):
#     """
#     Parse a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L.
#
#     Slippage convention (matches calibration):
#         fill = base * (1 ± k * |qty| / ADV_t)
#       - BUY uses +, SELL uses −
#       - ρ = |qty| / ADV_t (non-negative)
#
#     Also:
#       • Supports '#at_open'
#       • Handles partial exits with per-slice records
#       • Final TradeReport PnL is settled by cash-path delta
#       • Overshoot-safe (closing and opening in one fill)
#       • Adds a closed-form *no-slippage baseline* PnL for reconciliation:
#           - record entry_base_price on open
#           - accumulate exit_base_vwap from per-slice base_price
#           - at close: pnl_no_slip = sign * qty_total * (exit_base_vwap - entry_base_price)
#     """
#
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#
#         # portfolio state
#         self.cash = starting_cash
#         self.position = 0.0
#
#         # slippage params
#         self.k = shortfall_coeff
#         self.adv_map = adv_map or {}
#
#         # last-seen market data
#         self.last_price: Optional[float] = None
#         self.last_open: Optional[float] = None
#         self.last_mid: Optional[float] = None
#
#         # trade-in-progress (actual)
#         self.entry_price: Optional[float] = None
#         self.entry_ts: Optional[float] = None
#         self.entry_qty: float = 0.0            # sign of entry side
#         self.trade_vol: float = 0.0            # Σ(fill * qty) for the open trade
#         self._realized_pnl_accum: float = 0.0  # realized partial exits (diagnostic)
#         self._total_entry_qty: float = 0.0     # absolute open quantity (constant for the trade)
#         self._cash_at_entry: Optional[float] = None
#
#         # slippage aggregation for the current trade
#         self._slip_accum: float = 0.0          # Σ |fill-base| × |qty|  (cash)
#         self._slip_pct_accum: float = 0.0      # Σ |(fill/base) - 1|
#         self._slip_n: int = 0                  # # of fills for this trade
#
#         # VWAP audit of actual exit fills
#         self._vwap_exit_num: float = 0.0       # Σ qty * fill_price
#         self._vwap_exit_den: float = 0.0       # Σ qty
#
#         # ===== No-slippage baseline (closed-form) =====
#         self._entry_base_price: Optional[float] = None   # base at entry
#         self._vwap_exit_base_num: float = 0.0            # Σ qty * base_price (exit slices)
#         self._vwap_exit_base_den: float = 0.0            # Σ qty
#
#         # ===== Precise slippage attribution (cash) =====
#         self._price_effect_signed: float = 0.0  # Σ qty * (fill - base)  — price effect (signed cash)
#         self._entry_cash_used: float = 0.0  # Σ |qty| * fill (only entry-side)
#         self._entry_qty_total: float = 0.0  # Σ |qty| (only entry-side)
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # 1) On BAR, update prices
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_open = getattr(e, "O", None)
#             self.last_mid = 0.5 * (getattr(e, "O", 0.0) + e.C)
#             return
#
#         # 2) Only handle COMM beyond here
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         lp = self.last_price if self.last_price is not None else 0.0
#         print(f"[EXEC] got COMM '{e.text}' @ last_price={lp:.2f}")
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # reference price (open if #at_open is present and available)
#         use_open = "at_open" in flags
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#         if base_price is None or base_price == 0.0:
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # 3) Determine quantity (ALLOW “ALL”), with close-intent hygiene
#         fixed_cash = None
#         # ==== 恢复：只平/不反手 的 close-intent 语义 ====
#         close_intent_reasons = {
#             "stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"
#         }
#         is_close_intent = (reason in close_intent_reasons)
#         old_pos = self.position
#
#         if qty_txt.upper() == "ALL":
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#                     return
#                 qty = abs(old_pos)
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = abs(old_pos)  # pure close
#                 else:
#                     fixed_cash = self.cash
#                     qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (reason={reason}).")
#                     return
#                 if old_pos * side_sign >= 0:
#                     # 强制把方向改成“减仓方向”
#                     side_sign = -1 if old_pos > 0 else 1
#                 qty = min(qty, abs(old_pos))
#             else:
#                 # 反向信号 → 最多把旧仓减到 0，不反手
#                 if old_pos * side_sign < 0:
#                     qty = min(qty, abs(old_pos))
#
#         signed_qty = side_sign * qty
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # 4) Compute fill price with slippage — matches calibration
#         adv_i = self.adv_map.get(e.timestamp) if self.adv_map else None
#
#         if self.k > 0 and adv_i and adv_i > 0:
#             rho_abs = abs(signed_qty) / adv_i
#             slip_frac = self.k * rho_abs
#             #slip_frac = 0.0# ≥ 0
#             fill_price = base_price * (1.0 + (1.0 if signed_qty > 0 else -1.0) * slip_frac)
#         else:
#             slip_frac = 0.0
#             fill_price = base_price
#
#         # Per-fill slippage for metrics (cash & pct; pct uses absolute)
#         slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty)
#         slip_pct_fill = abs((fill_price / base_price) - 1.0)
#
#         # Accumulate to current-trade totals (written to TradeReport at close)
#         self._slip_accum += slip_cost_fill
#         self._slip_pct_accum += slip_pct_fill
#         self._slip_n += 1
#
#         # If "ALL" entry with fixed cash, recompute qty with slipped price
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             print(f"[EXEC] adjusted {'BUY' if side_sign>0 else 'SELL'} ALL → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price
#
#         # A) Snapshot cash at entry BEFORE first opening fill
#         opening_trade = (self.entry_price is None and self.position == 0 and signed_qty != 0)
#         if opening_trade:
#             self._cash_at_entry = self.cash
#             # NEW: no-slip baseline — record base at entry
#             self._entry_base_price = base_price
#             # reset exit-base accumulators just in case
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#         # B) How much of this fill actually closes old position?
#         old_pos = self.position
#         reduced = 0.0
#         slice_realized = 0.0
#         slice_slip = 0.0
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = min(abs(old_pos), abs(signed_qty))
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0
#                 slice_realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += slice_realized
#                 # VWAP audit (actual)
#                 self._vwap_exit_num += reduced * fill_price
#                 self._vwap_exit_den += reduced
#                 # NEW: VWAP on base for no-slip baseline
#                 self._vwap_exit_base_num += reduced * base_price
#                 self._vwap_exit_base_den += reduced
#                 # slice slippage in cash (for partial-exit report)
#                 slice_slip = abs(reduced * (fill_price - base_price))
#
#         # C) Book the fill (+ emit TradeFill with base/ADV for calibration audit)
#         emit(TradeFill(
#             timestamp=e.timestamp,
#             price=fill_price,
#             qty=signed_qty,
#             value=trade_val,
#             base_price=base_price,     # 给 Calibration 用
#             adv_at_fill=adv_i          # 给 Calibration 用
#         ))
#         self.cash -= trade_val
#         self.position += signed_qty
#
#         # D) If this was a pure partial exit (still have position), emit slice report
#         if reduced > 0 and self.position != 0:
#             print(f"[EXEC.partial] old_pos={old_pos:.6f} signed={signed_qty:.6f} "
#                   f"reduced={reduced:.6f} new_pos={self.position:.6f} reason={reason}")
#             emit(PartialExitReport(
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=reduced,
#                 pnl=slice_realized,
#                 reason=reason,
#                 trade_type=("long" if old_pos > 0 else "short"),
#                 cash_after=self.cash,
#                 position_after=self.position,
#                 slippage_cost=slice_slip,
#                 slippage_pct=slip_pct_fill
#             ))
#
#         # E) Overshoot handling: same fill closes old and opens new side
#         if old_pos * signed_qty < 0 and abs(signed_qty) > abs(old_pos):
#             # remainder becomes a fresh entry at this fill_price
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = self.position
#             self.trade_vol = fill_price * self.position
#             self._total_entry_qty = abs(self.position)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: reset baseline accumulators for the NEW trade
#             self._cash_at_entry = self.cash
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slippage stats for the NEW trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#         print(
#             f"[EXEC.overshoot] remainder={self.position:+.6f} "
#             f"opened={'long' if self.position > 0 else 'short'} @ {fill_price:.2f} "
#             f"(closed {old_pos:+.6f} with {signed_qty:+.6f}, reason={reason})"
#         )
#
#         # F) Inventory snapshot
#         inventory_value = self.position * (self.last_price if self.last_price is not None else 0.0)
#         emit(InvUpdate(timestamp=e.timestamp, position=self.position, inventory_value=inventory_value))
#
#         # G) Mark a clean new entry (no overshoot)
#         if self.entry_price is None and signed_qty != 0 and self.position != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: baseline entry/base reset
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slip stats for a TRUE new trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             print(f"[EXEC] opened position → {'long' if self.position > 0 else 'short'} "
#                   f"@ {self.entry_price:.2f} (reason={reason})")
#
#         # H) Fully closed → emit TradeReport (cash-path PnL; keep last real exit price)
#         if self.entry_price is not None and self.position == 0:
#             # Authoritative PnL = cash delta over the full trade
#             pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
#
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#
#             # Slippage pct (avg of per-fill absolute slip %)
#             slippage_pct = (self._slip_pct_accum / self._slip_n) * 100.0 if self._slip_n else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#
#             # Actual exit price(s)
#             exit_price_last = fill_price
#             exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last
#
#             # ===== No-slip baseline close-out =====
#             exit_base_vwap = (self._vwap_exit_base_num / self._vwap_exit_base_den) \
#                              if self._vwap_exit_base_den > 0 else base_price
#             entry_base = self._entry_base_price if self._entry_base_price is not None else self.entry_price
#             sign = 1.0 if self.entry_qty > 0 else -1.0
#             pnl_no_slip = sign * abs(self._total_entry_qty) * (exit_base_vwap - entry_base)
#             ret_no_slip_pct = (pnl_no_slip / denom_notional) * 100.0 if denom_notional else 0.0
#             delta_ret_pct = ret_no_slip_pct - ret_pct
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=exit_price_last,        # last real exit price (for charts)
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#                 exit_price_vwap=exit_price_vwap,
#
#                 # ======== no-slip baseline & reconciliation helpers ========
#                 pnl_no_slip=pnl_no_slip,
#                 return_no_slip_pct=ret_no_slip_pct,
#                 delta_ret_pct=delta_ret_pct,              # (no-slip − actual)
#                 entry_base_price=entry_base,
#                 exit_base_vwap=exit_base_vwap
#             )
#
#             print(f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                   f"→ TradeReport entry={self.entry_price:.2f} "
#                   f"exit(last)={exit_price_last:.2f} vwap={exit_price_vwap:.2f}")
#
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0
#             self._cash_at_entry = None
#
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: baseline reset
#             self._entry_base_price = None
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0




# # core/executor.py
# # (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe, no-slip baseline)
#
# from typing import Dict, Optional
# from core.events import (
#     Event, BAR, COMM,
#     TradeFill, InvUpdate, TradeReport, PartialExitReport
# )
# from core.fsm import Agent
# from core.plumbing import emit
#
#
# def parse_comm_text(text: str):
#     """
#     Parse a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L.
#
#     Slippage convention (matches calibration):
#         fill = base * (1 ± k * |qty| / ADV_t)
#       - BUY uses +, SELL uses −
#       - ρ = |qty| / ADV_t (non-negative)
#
#     Also:
#       • Supports '#at_open'
#       • Handles partial exits with per-slice records
#       • Final TradeReport PnL is settled by cash-path delta
#       • Overshoot-safe (closing and opening in one fill)
#       • Adds a closed-form *no-slippage baseline* PnL for reconciliation:
#           - record entry_base_price on open
#           - accumulate exit_base_vwap from per-slice base_price
#           - at close: pnl_no_slip = sign * qty_total * (exit_base_vwap - entry_base_price)
#     """
#
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#
#         # portfolio state
#         self.cash = starting_cash
#         self.position = 0.0
#
#         # slippage params
#         self.k = shortfall_coeff
#         self.adv_map = adv_map or {}
#
#         # last-seen market data
#         self.last_price: Optional[float] = None
#         self.last_open: Optional[float] = None
#         self.last_mid: Optional[float] = None
#
#         # trade-in-progress (actual)
#         self.entry_price: Optional[float] = None
#         self.entry_ts: Optional[float] = None
#         self.entry_qty: float = 0.0            # sign of entry side
#         self.trade_vol: float = 0.0            # Σ(fill * qty) for the open trade
#         self._realized_pnl_accum: float = 0.0  # realized partial exits (diagnostic)
#         self._total_entry_qty: float = 0.0     # absolute open quantity (constant for the trade)
#         self._cash_at_entry: Optional[float] = None
#
#         # slippage aggregation for the current trade
#         self._slip_accum: float = 0.0          # Σ |fill-base| × |qty|  (cash)
#         self._slip_pct_accum: float = 0.0      # Σ |(fill/base) - 1|
#         self._slip_n: int = 0                  # # of fills for this trade
#
#         # VWAP audit of actual exit fills
#         self._vwap_exit_num: float = 0.0       # Σ qty * fill_price
#         self._vwap_exit_den: float = 0.0       # Σ qty
#
#         # ===== No-slippage baseline (closed-form) =====
#         self._entry_base_price: Optional[float] = None   # base at entry
#         self._vwap_exit_base_num: float = 0.0            # Σ qty * base_price (exit slices)
#         self._vwap_exit_base_den: float = 0.0            # Σ qty
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # 1) On BAR, update prices
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_open = getattr(e, "O", None)
#             self.last_mid = 0.5 * (getattr(e, "O", 0.0) + e.C)
#             return
#
#         # 2) Only handle COMM beyond here
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         lp = self.last_price if self.last_price is not None else 0.0
#         print(f"[EXEC] got COMM '{e.text}' @ last_price={lp:.2f}")
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # reference price (open if #at_open is present and available)
#         use_open = "at_open" in flags
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#         if base_price is None or base_price == 0.0:
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # 3) Determine quantity (ALLOW “ALL”), with close-intent hygiene
#         fixed_cash = None
#         close_intent_reasons = {
#             "stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"
#         }
#         is_close_intent = (reason in close_intent_reasons)
#         old_pos = self.position
#
#         if qty_txt.upper() == "ALL":
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#                     return
#                 qty = abs(old_pos)
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = abs(old_pos)  # pure close
#                 else:
#                     fixed_cash = self.cash
#                     qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (reason={reason}).")
#                     return
#                 if old_pos * side_sign >= 0:
#                     side_sign = -1 if old_pos > 0 else 1
#                 qty = min(qty, abs(old_pos))
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = min(qty, abs(old_pos))
#
#         signed_qty = side_sign * qty
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # 4) Compute fill price with slippage — matches calibration
#         adv_i = self.adv_map.get(e.timestamp) if self.adv_map else None
#
#         if self.k > 0 and adv_i and adv_i > 0:
#             rho_abs = abs(signed_qty) / adv_i
#             slip_frac = self.k * rho_abs                     # ≥ 0
#             fill_price = base_price * (1.0 + (1.0 if signed_qty > 0 else -1.0) * slip_frac)
#         else:
#             slip_frac = 0.0
#             fill_price = base_price
#
#         # Per-fill slippage for metrics (cash & pct; pct uses absolute)
#         slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty)
#         slip_pct_fill = abs((fill_price / base_price) - 1.0)
#
#         # Accumulate to current-trade totals (written to TradeReport at close)
#         self._slip_accum += slip_cost_fill
#         self._slip_pct_accum += slip_pct_fill
#         self._slip_n += 1
#
#         # If "ALL" entry with fixed cash, recompute qty with slipped price
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             print(f"[EXEC] adjusted {'BUY' if side_sign>0 else 'SELL'} ALL → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price
#
#         # A) Snapshot cash at entry BEFORE first opening fill
#         opening_trade = (self.entry_price is None and self.position == 0 and signed_qty != 0)
#         if opening_trade:
#             self._cash_at_entry = self.cash
#             # NEW: no-slip baseline — record base at entry
#             self._entry_base_price = base_price
#             # reset exit-base accumulators just in case
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#         # B) How much of this fill actually closes old position?
#         old_pos = self.position
#         reduced = 0.0
#         slice_realized = 0.0
#         slice_slip = 0.0
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = min(abs(old_pos), abs(signed_qty))
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0
#                 slice_realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += slice_realized
#                 # VWAP audit (actual)
#                 self._vwap_exit_num += reduced * fill_price
#                 self._vwap_exit_den += reduced
#                 # NEW: VWAP on base for no-slip baseline
#                 self._vwap_exit_base_num += reduced * base_price
#                 self._vwap_exit_base_den += reduced
#                 # slice slippage in cash (for partial-exit report)
#                 slice_slip = abs(reduced * (fill_price - base_price))
#
#         # C) Book the fill (+ emit TradeFill with base/ADV for calibration audit)
#         emit(TradeFill(
#             timestamp=e.timestamp,
#             price=fill_price,
#             qty=signed_qty,
#             value=trade_val,
#             base_price=base_price,     # optional (if your TradeFill supports it)
#             adv_at_fill=adv_i          # optional (if your TradeFill supports it)
#         ))
#         self.cash -= trade_val
#         self.position += signed_qty
#
#         # D) If this was a pure partial exit (still have position), emit slice report
#         if reduced > 0 and self.position != 0:
#             print(f"[EXEC.partial] old_pos={old_pos:.6f} signed={signed_qty:.6f} "
#                   f"reduced={reduced:.6f} new_pos={self.position:.6f} reason={reason}")
#             emit(PartialExitReport(
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=reduced,
#                 pnl=slice_realized,
#                 reason=reason,
#                 trade_type=("long" if old_pos > 0 else "short"),
#                 cash_after=self.cash,
#                 position_after=self.position,
#                 slippage_cost=slice_slip,
#                 slippage_pct=slip_pct_fill
#             ))
#
#         # E) Overshoot handling: same fill closes old and opens new side
#         if old_pos * signed_qty < 0 and abs(signed_qty) > abs(old_pos):
#             # remainder becomes a fresh entry at this fill_price
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = self.position
#             self.trade_vol = fill_price * self.position
#             self._total_entry_qty = abs(self.position)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: reset baseline accumulators for the NEW trade
#             self._cash_at_entry = self.cash
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slippage stats for the NEW trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#         print(
#             f"[EXEC.overshoot] remainder={self.position:+.6f} "
#             f"opened={'long' if self.position > 0 else 'short'} @ {fill_price:.2f} "
#             f"(closed {old_pos:+.6f} with {signed_qty:+.6f}, reason={reason})"
#         )
#
#         # F) Inventory snapshot (mark-to-market inventory value not used in PnL)
#         inventory_value = self.position * (self.last_price if self.last_price is not None else 0.0)
#         emit(InvUpdate(timestamp=e.timestamp, position=self.position, inventory_value=inventory_value))
#
#         # G) Mark a clean new entry (no overshoot)
#         if self.entry_price is None and signed_qty != 0 and self.position != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: baseline entry/base reset
#             self._entry_base_price = base_price
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0
#
#             # reset slip stats for a TRUE new trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             print(f"[EXEC] opened position → {'long' if self.position > 0 else 'short'} "
#                   f"@ {self.entry_price:.2f} (reason={reason})")
#
#         # H) Fully closed → emit TradeReport (cash-path PnL; keep last real exit price)
#         if self.entry_price is not None and self.position == 0:
#             # Authoritative PnL = cash delta over the full trade
#             pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
#
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#
#             # Slippage pct (avg of per-fill absolute slip %)
#             slippage_pct = (self._slip_pct_accum / self._slip_n) * 100.0 if self._slip_n else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#
#             # Actual exit price(s)
#             exit_price_last = fill_price
#             exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last
#
#             # ===== No-slip baseline close-out =====
#             exit_base_vwap = (self._vwap_exit_base_num / self._vwap_exit_base_den) \
#                              if self._vwap_exit_base_den > 0 else base_price
#             entry_base = self._entry_base_price if self._entry_base_price is not None else self.entry_price
#             sign = 1.0 if self.entry_qty > 0 else -1.0
#             pnl_no_slip = sign * abs(self._total_entry_qty) * (exit_base_vwap - entry_base)
#             ret_no_slip_pct = (pnl_no_slip / denom_notional) * 100.0 if denom_notional else 0.0
#             delta_ret_pct = ret_no_slip_pct - ret_pct
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=exit_price_last,        # last real exit price (for charts)
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#                 exit_price_vwap=exit_price_vwap,
#
#                 # ======== NEW: no-slip baseline & reconciliation helpers ========
#                 pnl_no_slip=pnl_no_slip,
#                 return_no_slip_pct=ret_no_slip_pct,
#                 delta_ret_pct=delta_ret_pct,              # (no-slip − actual)
#                 entry_base_price=entry_base,
#                 exit_base_vwap=exit_base_vwap
#             )
#
#             print(f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                   f"→ TradeReport entry={self.entry_price:.2f} "
#                   f"exit(last)={exit_price_last:.2f} vwap={exit_price_vwap:.2f}")
#
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0
#             self._cash_at_entry = None
#
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#
#             # NEW: baseline reset
#             self._entry_base_price = None
#             self._vwap_exit_base_num = 0.0
#             self._vwap_exit_base_den = 0.0



# # core/executor.py (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe)
#
# from typing import Dict, Optional
# from core.events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport, PartialExitReport
# from core.fsm import Agent
# from core.plumbing import emit
#
# def parse_comm_text(text: str):
#     """
#     Parses a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L,
#     charging calibrated slippage with the SAME convention as calibration:
#         fill = base * (1 ± k * |qty| / ADV_t)
#       - BUY uses +, SELL uses −
#       - ρ = |qty| / ADV_t  (non-negative)
#     Supports '#at_open'. Handles partial exits with per-slice records.
#     Final TradeReport PnL is settled by cash-path delta. Overshoot-safe.
#     """
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash = starting_cash
#         self.position = 0.0
#         # slippage params
#         self.k = shortfall_coeff
#         self.adv_map = adv_map or {}
#         # last-seen market data
#         self.last_price = None
#         self.last_mid = None
#         self.last_open = None
#         # trade-in-progress
#         self.entry_price = None
#         self.entry_ts = None
#         self.entry_qty = 0.0
#         self.trade_vol = 0.0
#         self._slip_accum = 0.0              # 本笔交易累计滑点“现金成本”（绝对值求和）
#         self._slip_pct_accum = 0.0          # 本笔交易每次成交的 |滑点比例| 之和
#         self._slip_n = 0                    # 本笔交易累计成交次数
#         # partial-exit accounting
#         self._realized_pnl_accum = 0.0
#         self._total_entry_qty = 0.0
#         self._cash_at_entry = None
#         self._vwap_exit_num = 0.0
#         self._vwap_exit_den = 0.0
#         # --- shadow (no-slippage) cash path ---
#         self._shadow_cash = starting_cash
#         self._shadow_cash_at_entry = None
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # 1) On BAR, update prices
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_open = getattr(e, "O", None)
#             self.last_mid = 0.5 * (getattr(e, "O", 0.0) + e.C)
#             return
#
#         # 2) Only handle COMM beyond here
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         lp = self.last_price if self.last_price is not None else 0.0
#         print(f"[EXEC] got COMM '{e.text}' @ last_price={lp:.2f}")
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # reference price (open if #at_open is present and available)
#         use_open = "at_open" in flags
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#         if base_price is None or base_price == 0.0:
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # 3) Determine quantity (ALLOW “ALL”)
#         fixed_cash = None
#         # Reasons that imply a **close-only** intent.
#         # IMPORTANT: Do NOT include 'reverse_ema_flip' here (that one is an entry/reversal).
#         close_intent_reasons = {"stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"}
#         is_close_intent = (reason in close_intent_reasons)
#         old_pos = self.position
#
#         if qty_txt.upper() == "ALL":
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#                     return
#                 qty = abs(old_pos)
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = abs(old_pos)  # 纯平
#                 else:
#                     fixed_cash = self.cash
#                     qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#
#             # 数值型 close-intent 的稳健处理
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (reason={reason}).")
#                     return
#                 # 强制按“只平仓”处理，并对数量做 clamp
#                 if old_pos * side_sign >= 0:
#                     side_sign = -1 if old_pos > 0 else 1
#                 qty = min(qty, abs(old_pos))
#             else:
#                 # 非 close-intent 的原有保护：反向时最多只把旧仓平掉
#                 if old_pos * side_sign < 0:
#                     qty = min(qty, abs(old_pos))
#
#         signed_qty = side_sign * qty
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # 4) Compute fill price with slippage —— 与校准一致
#         adv_i = self.adv_map.get(e.timestamp) if self.adv_map else None
#
#         if self.k > 0 and adv_i and adv_i > 0:
#             # ρ 用绝对值；方向由 side_sign 决定（买加价、卖减价）
#             rho_abs = abs(signed_qty) / adv_i
#             #slip_frac = self.k * rho_abs                     # ≥ 0
#             slip_frac = 0.0
#             fill_price = base_price * (1.0 + (1.0 if signed_qty > 0 else -1.0) * slip_frac)
#         else:
#             slip_frac = 0.0
#             fill_price = base_price
#
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             print(f"[EXEC] adjusted {'BUY' if side_sign>0 else 'SELL'} ALL → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price
#         # --- shadow path: execute the same qty at base_price (no slippage) ---
#         shadow_trade_val = signed_qty * base_price
#
#         # 本次成交的滑点（现金与百分比）；用于 partial/Report 与 metrics
#         slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty)  # 现金口径
#         #fill_slip_cash = abs(trade_val - shadow_trade_val)
#         slip_pct_fill = abs((fill_price / base_price) - 1.0)  # 比例口径（绝对值）
#
#         # 累加到“本笔交易”的统计（交易完全平仓时写入 TradeReport）
#         self._slip_accum += slip_cost_fill
#         self._slip_pct_accum += slip_pct_fill
#         self._slip_n += 1
#
#         # A) Snapshot cash at entry BEFORE applying first opening fill
#         opening_trade = (self.entry_price is None and self.position == 0 and signed_qty != 0)
#         if opening_trade:
#             self._cash_at_entry = self.cash
#             self._shadow_cash_at_entry = self._shadow_cash
#
#         # B) How much of this fill actually closes old position?
#         old_pos = self.position
#         reduced = 0.0
#         slice_realized = 0.0
#         slice_slip = 0.0
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = min(abs(old_pos), abs(signed_qty))
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0
#                 slice_realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += slice_realized
#                 # accumulate VWAP for audit
#                 self._vwap_exit_num += reduced * fill_price
#                 self._vwap_exit_den += reduced
#                 slice_slip = abs(reduced * (fill_price - base_price))
#
#         # C) Book the fill (+ emit TradeFill with base/ADV for calibration)
#         emit(TradeFill(
#             timestamp = e.timestamp,
#             price     = fill_price,
#             qty       = signed_qty,
#             value     = trade_val,
#             base_price = base_price,        # NEW: give calibration the true base
#             adv_at_fill = adv_i             # NEW: and the ADV at fill time
#         ))
#         self.cash -= trade_val
#         self._shadow_cash -= shadow_trade_val
#         self.position += signed_qty
#
#         # D) If this was a pure partial exit (still have position), emit slice report
#         if reduced > 0 and self.position != 0:
#             print(f"[EXEC.partial] old_pos={old_pos:.6f} signed={signed_qty:.6f} "
#                   f"reduced={reduced:.6f} new_pos={self.position:.6f} reason={reason}")
#             emit(PartialExitReport(
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=reduced,
#                 pnl=slice_realized,
#                 reason=reason,
#                 trade_type=("long" if old_pos > 0 else "short"),
#                 cash_after=self.cash,
#                 position_after=self.position,
#                 slippage_cost=slice_slip,
#                 slippage_pct=slip_pct_fill    # NEW: add pct for diagnostics
#             ))
#
#         # E) Overshoot handling: same fill closes old and opens new side
#         if old_pos * signed_qty < 0 and abs(signed_qty) > abs(old_pos):
#             # remainder becomes a fresh entry at this fill_price
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = self.position
#             self.trade_vol = fill_price * self.position
#             self._total_entry_qty = abs(self.position)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#             self._cash_at_entry = self.cash  # snapshot for the NEW trade
#             self._shadow_cash_at_entry = self._shadow_cash
#             # reset slip stats for the NEW trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#         print(
#             f"[EXEC.overshoot] remainder={self.position:+.6f} "
#             f"opened={'long' if self.position > 0 else 'short'} @ {fill_price:.2f} "
#             f"(closed {old_pos:+.6f} with {signed_qty:+.6f}, reason={reason})"
#         )
#
#         # F) Inventory snapshot
#         inventory_value = self.position * (self.last_price if self.last_price is not None else 0.0)
#         emit(InvUpdate(timestamp=e.timestamp, position=self.position, inventory_value=inventory_value))
#
#         # G) Mark a clean new entry (no overshoot)
#         if self.entry_price is None and signed_qty != 0 and self.position != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#             # reset slip stats for a TRUE new trade
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#
#             print(f"[EXEC] opened position → {'long' if self.position > 0 else 'short'} "
#                   f"@ {self.entry_price:.2f} (reason={reason})")
#
#         # H) Fully closed → emit TradeReport (cash-path PnL; keep last real exit price)
#         if self.entry_price is not None and self.position == 0:
#             pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#             # --- shadow (no-slippage) diagnostics ---
#             pnl_no_slip = (
#                         self._shadow_cash - self._shadow_cash_at_entry) if self._shadow_cash_at_entry is not None else 0.0
#             ret_no_slip_pct = (pnl_no_slip / denom_notional) * 100.0 if denom_notional else 0.0
#             delta_ret_pct = ret_no_slip_pct - ret_pct
#
#             # 推荐：用“本笔交易平均滑点比例”作为 slippage_pct
#             slippage_pct = (self._slip_pct_accum / self._slip_n) * 100.0 if self._slip_n else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#             exit_price_last = fill_price
#             exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last
#
#             diff_shadow_real = pnl_no_slip - pnl
#             if abs(diff_shadow_real - self._slip_accum) > 1e-6:
#                 print(
#                     f"[WARN] Reconcile mismatch: shadow-real={diff_shadow_real:.6f} vs slip_accum={self._slip_accum:.6f}")
#
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=exit_price_last,      # last real exit price (for charts)
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 pnl_no_slip=pnl_no_slip,
#                 return_no_slip_pct=ret_no_slip_pct,
#                 delta_ret_pct=delta_ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#                 exit_price_vwap=exit_price_vwap  # VWAP over all exit slices (for audit)
#             )
#             print(f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% → TradeReport entry={self.entry_price:.2f} exit(last)={exit_price_last:.2f} vwap={exit_price_vwap:.2f}")
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._slip_accum = 0.0
#             self._slip_pct_accum = 0.0
#             self._slip_n = 0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0
#             self._cash_at_entry = None
#             self._shadow_cash_at_entry = None
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0



# # core/executor.py (robust partial exits, VWAP audit, cash-path PnL, overshoot-safe)
#
# from typing import Dict, Optional
# from core.events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport, PartialExitReport
# from core.fsm import Agent
# from core.plumbing import emit
#
# def parse_comm_text(text: str):
#     """
#     Parses a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L,
#     charging calibrated slippage = k * (signed_qty / ADV_t).
#     Supports '#at_open'. Handles partial exits with per-slice records.
#     Final TradeReport PnL is settled by cash-path delta. Overshoot-safe.
#     """
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash = starting_cash
#         self.position = 0.0
#         # slippage params
#         self.k = shortfall_coeff
#         self.adv_map = adv_map or {}
#         # last-seen market data
#         self.last_price = None
#         self.last_mid = None
#         self.last_open = None
#         # trade-in-progress
#         self.entry_price = None
#         self.entry_ts = None
#         self.entry_qty = 0.0
#         self.trade_vol = 0.0
#         self._slip_accum = 0.0
#         # partial-exit accounting
#         self._realized_pnl_accum = 0.0
#         self._total_entry_qty = 0.0
#         self._cash_at_entry = None
#         self._vwap_exit_num = 0.0
#         self._vwap_exit_den = 0.0
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # 1) On BAR, update prices
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_open = getattr(e, "O", None)
#             self.last_mid = 0.5 * (getattr(e, "O", 0.0) + e.C)
#             return
#
#         # 2) Only handle COMM beyond here
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         lp = self.last_price if self.last_price is not None else 0.0
#         print(f"[EXEC] got COMM '{e.text}' @ last_price={lp:.2f}")
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # reference price (open if #at_open is present and available)
#         use_open = "at_open" in flags
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#         if base_price is None or base_price == 0.0:
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # 3) Determine quantity (ALLOW “ALL”)
#         fixed_cash = None
#         # Reasons that imply a **close-only** intent.
#         # IMPORTANT: Do NOT include 'reverse_ema_flip' here (that one is an entry/reversal).
#         close_intent_reasons = {"stop_loss", "giveback", "take_profit", "take_profit1", "ema_flip"}
#         is_close_intent = (reason in close_intent_reasons)
#         old_pos = self.position
#
#         if qty_txt.upper() == "ALL":
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#                     return
#                 qty = abs(old_pos)
#             else:
#                 if old_pos * side_sign < 0:
#                     qty = abs(old_pos)  # 纯平
#                 else:
#                     fixed_cash = self.cash
#                     qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#
#             # ⚠️ 新增：数值型 close-intent 的稳健处理
#             if is_close_intent:
#                 if old_pos == 0:
#                     print(f"[EXEC] ignoring '{e.text}' while flat (reason={reason}).")
#                     return
#                 # 强制按“只平仓”处理，并对数量做 clamp
#                 if old_pos * side_sign >= 0:
#                     side_sign = -1 if old_pos > 0 else 1
#                 qty = min(qty, abs(old_pos))
#             else:
#                 # 非 close-intent 的原有保护：反向时最多只把旧仓平掉
#                 if old_pos * side_sign < 0:
#                     qty = min(qty, abs(old_pos))
#
#         # if qty_txt.upper() == "ALL":
#         #     if is_close_intent:
#         #         # Explicit close: close position only; if already flat, ignore the command.
#         #         if self.position == 0:
#         #             print(f"[EXEC] ignoring '{e.text}' while flat (ALL with reason={reason}).")
#         #             return
#         #         qty = abs(self.position)
#         #     else:
#         #         # Entry-type ALL (no reason or 'reverse_ema_flip'):
#         #         # - if opposite to current position → pure close
#         #         # - otherwise (flat or same side) → open using all cash
#         #         if self.position * side_sign < 0:
#         #             qty = abs(self.position)
#         #         else:
#         #             fixed_cash = self.cash
#         #             qty = self.cash / base_price
#         # else:
#         #     qty = float(qty_txt)
#         #     # Clamp to available position when this order closes in the opposite direction.
#         #     if self.position * side_sign < 0:
#         #         qty = min(qty, abs(self.position))
#         #
#         signed_qty = side_sign * qty
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # if qty_txt.upper() == "ALL":
#         #     if self.position * side_sign < 0:
#         #         # opposite side: pure close
#         #         qty = abs(self.position)
#         #     else:
#         #         fixed_cash = self.cash
#         #         qty = self.cash / base_price
#         # else:
#         #     qty = float(qty_txt)
#         #     # defensive clamp when it's a close
#         #     if self.position * side_sign < 0:
#         #         qty = min(qty, abs(self.position))
#         # signed_qty = side_sign * qty
#         # print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # 4) Compute fill price with slippage
#         if self.k > 0 and self.adv_map:
#             adv_i = self.adv_map.get(e.timestamp, 1.0)
#             rho = signed_qty / adv_i if adv_i else 0.0
#             slip_pct = self.k * rho
#             fill_price = base_price * (1 + slip_pct)
#         else:
#             fill_price = base_price
#
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             print(f"[EXEC] adjusted {'BUY' if side_sign>0 else 'SELL'} ALL → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price
#         slip_dollars = abs(signed_qty * (fill_price - base_price))
#         self._slip_accum += slip_dollars
#
#         # A) Snapshot cash at entry BEFORE applying first opening fill
#         opening_trade = (self.entry_price is None and self.position == 0 and signed_qty != 0)
#         if opening_trade:
#             self._cash_at_entry = self.cash
#
#         # B) How much of this fill actually closes old position?
#         old_pos = self.position
#         reduced = 0.0
#         slice_realized = 0.0
#         slice_slip = 0.0
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = min(abs(old_pos), abs(signed_qty))
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0
#                 slice_realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += slice_realized
#                 # accumulate VWAP for audit
#                 self._vwap_exit_num += reduced * fill_price
#                 self._vwap_exit_den += reduced
#                 slice_slip = abs(reduced * (fill_price - base_price))
#
#         # C) Book the fill
#         emit(TradeFill(timestamp=e.timestamp, price=fill_price, qty=signed_qty, value=trade_val))
#         self.cash -= trade_val
#         self.position += signed_qty
#
#         # D) If this was a pure partial exit (still have position), emit slice report
#         if reduced > 0 and self.position != 0:
#             print(f"[EXEC.partial] old_pos={old_pos:.6f} signed={signed_qty:.6f} "
#                   f"reduced={reduced:.6f} new_pos={self.position:.6f} reason={reason}")
#             emit(PartialExitReport(
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=reduced,
#                 pnl=slice_realized,
#                 reason=reason,
#                 trade_type=("long" if old_pos > 0 else "short"),
#                 cash_after=self.cash,
#                 position_after=self.position,
#                 slippage_cost=slice_slip
#             ))
#
#         # E) Overshoot handling: same fill closes old and opens new side
#         if old_pos * signed_qty < 0 and abs(signed_qty) > abs(old_pos):
#             # remainder becomes a fresh entry at this fill_price
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = self.position
#             self.trade_vol = fill_price * self.position
#             self._total_entry_qty = abs(self.position)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#             self._cash_at_entry = self.cash  # snapshot for the NEW trade
#         print(
#             f"[EXEC.overshoot] remainder={self.position:+.6f} "
#             f"opened={'long' if self.position > 0 else 'short'} @ {fill_price:.2f} "
#             f"(closed {old_pos:+.6f} with {signed_qty:+.6f}, reason={reason})"
#         )
#
#         # F) Inventory snapshot
#         inventory_value = self.position * (self.last_price if self.last_price is not None else 0.0)
#         emit(InvUpdate(timestamp=e.timestamp, position=self.position, inventory_value=inventory_value))
#
#         # G) Mark a clean new entry (no overshoot)
#         if self.entry_price is None and signed_qty != 0 and self.position != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0
#             # cash snapshot should already be taken as opening_trade
#             #print(f"[EXEC] opened position at {self.entry_price:.2f}")
#             print(f"[EXEC] opened position → {'long' if self.position > 0 else 'short'} "
#                   f"@ {self.entry_price:.2f} (reason={reason})")
#
#         # H) Fully closed → emit TradeReport (cash-path PnL; keep last real exit price)
#         if self.entry_price is not None and self.position == 0:
#             pnl = (self.cash - self._cash_at_entry) if self._cash_at_entry is not None else self._realized_pnl_accum
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#             slippage_pct = (self._slip_accum / abs(self.trade_vol)) * 100 if self.trade_vol else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#             exit_price_last = fill_price
#             exit_price_vwap = (self._vwap_exit_num / self._vwap_exit_den) if self._vwap_exit_den > 0 else exit_price_last
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=exit_price_last,      # last real exit price (for charts)
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#                 exit_price_vwap=exit_price_vwap  # VWAP over all exit slices (for audit)
#             )
#             print(f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% → TradeReport entry={self.entry_price:.2f} exit(last)={exit_price_last:.2f} vwap={exit_price_vwap:.2f}")
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._slip_accum = 0.0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0
#             self._cash_at_entry = None
#             self._vwap_exit_num = 0.0
#             self._vwap_exit_den = 0.0




# # core/executor.py (with slippage and #at_open support)
#
# from typing import Dict, Optional
# from core.events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm import Agent
# from core.plumbing import emit
#
#
# def parse_comm_text(text: str):
#     """
#     Parses a COMM text like:
#       "BUY ALL #reason:reverse_ema_flip #at_open"
#     Returns (side, qty_text, reason, flags_set)
#     """
#     parts = text.strip().split()
#     side = None
#     qty_text = None
#     reason = "manual"
#     flags = set()
#
#     # First two non-flag tokens are expected to be side and qty
#     non_flag = []
#     for token in parts:
#         if token.startswith("#"):
#             if token.startswith("#reason:"):
#                 reason = token.split(":", 1)[1]
#             else:
#                 flags.add(token.lstrip("#"))
#         else:
#             non_flag.append(token)
#     if len(non_flag) >= 2:
#         side = non_flag[0].upper()
#         qty_text = non_flag[1]
#     elif len(non_flag) == 1:
#         side = non_flag[0].upper()
#         qty_text = "ALL"
#     else:
#         raise ValueError(f"Cannot parse COMM text: '{text}'")
#
#     return side, qty_text, reason, flags
#
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L,
#     charging calibrated slippage = k * (signed_qty / ADV_i) off the base price.
#     Supports a “#at_open” tag to execute the fill against the current bar's open
#     instead of the last close.
#     """
#     def __init__(
#         self,
#         starting_cash: float,
#         shortfall_coeff: float = 0.0,
#         adv_map: Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash = starting_cash      # USD
#         self.position = 0.0            # asset units
#         # slippage parameters
#         self.k = shortfall_coeff       # calibrated impact coefficient
#         self.adv_map = adv_map or {}    # {timestamp: ADV_i}
#         # last-seen market data
#         self.last_price = None          # USD (close-based)
#         self.last_mid = None            # USD
#         self.last_open = None           # USD (open-based)
#         # trade-in-progress
#         self.entry_price = None
#         self.entry_ts = None
#         self.entry_qty = 0.0
#         self.trade_vol = 0.0            # USD notional at entry
#         self._slip_accum = 0.0
#         self._realized_pnl_accum = 0.0  # 累计“部分平仓”产生的已实现盈亏
#         self._total_entry_qty = 0.0 # 初始开仓数量（用于return%的分母）
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # ——————— 1) On each BAR, record prices ———————
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_open = getattr(e, "O", None)
#             self.last_mid = 0.5 * (getattr(e, "O", 0.0) + e.C)
#             return
#
#         # ——————— 2) Only handle COMM events beyond here ———————
#         if not isinstance(e, COMM):
#             return
#
#         # parse the COMM text
#         try:
#             side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
#         except ValueError as ve:
#             print(f"[EXEC] malformed COMM '{e.text}': {ve}")
#             return
#
#         # Safe debug print (guard if last_price is None)
#         lp = self.last_price if self.last_price is not None else 0.0
#         print(f"[EXEC] got COMM '{e.text}' @ last_price={lp:.2f}")
#
#         side = side_txt.upper()
#         side_sign = 1 if side == "BUY" else -1
#
#         # Determine whether to use open for this fill
#         use_open = "at_open" in flags  # corresponds to "#at_open"
#         base_label = "open" if use_open and self.last_open is not None else "close"
#         base_price = self.last_open if use_open and self.last_open is not None else self.last_price
#
#         if base_price is None or base_price == 0.0:
#             print(f"[EXEC] skipping COMM '{e.text}' because no reference price available yet.")
#             return
#
#         # ——————— 3) Determine quantity (ALLOW “ALL”) ———————
#         fixed_cash = None
#         if qty_txt.upper() == "ALL":
#             if self.position * side_sign < 0:
#                 # opposite side: close existing
#                 qty = abs(self.position)
#             elif side_sign > 0:
#                 fixed_cash = self.cash
#                 qty = self.cash / base_price
#             else:
#                 fixed_cash = self.cash
#                 qty = self.cash / base_price
#         else:
#             qty = float(qty_txt)
#         signed_qty = side_sign * qty
#
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f} (using {base_label} price)")
#
#         # ——————— 4) Compute fill price with slippage ———————
#         if self.k > 0 and self.adv_map:
#             adv_i = self.adv_map.get(e.timestamp, 1.0)
#             rho = signed_qty / adv_i if adv_i else 0.0
#             slip_pct = self.k * rho
#             fill_price = base_price * (1 + slip_pct)
#         else:
#             fill_price = base_price  # no slippage / calibration pass
#
#         # — if this was a BUY/SELL ALL, recompute qty so you spend exactly fixed_cash
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             if side_sign > 0:
#                 print(f"[EXEC] adjusted BUY ALL → qty={qty:.4f} after slip")
#             else:
#                 print(f"[EXEC] adjusted SELL ALL (short-all) → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price  # dollars exchanged
#
#         # --- accumulate slippage cost (consistent with base_price) ---
#         slip_dollars = abs(signed_qty * (fill_price - base_price))
#         self._slip_accum += slip_dollars
#
#         # ——————— 5) Emit TradeFill & update cash/position ———————
#         old_pos = self.position
#         emit(TradeFill(timestamp=e.timestamp, price=fill_price, qty=signed_qty, value=trade_val))
#         self.cash -= trade_val
#         self.position += signed_qty
#         # 若这笔成交与原持仓方向相反 => 减仓；把“减少的那部分”的盈亏计入
#         if self.entry_price is not None and old_pos != 0 and (old_pos * signed_qty) < 0:
#             reduced = max(0.0, abs(old_pos) - abs(self.position))  # 本次实际减掉的数量
#             if reduced > 0:
#                 dir_sign = 1.0 if old_pos > 0 else -1.0  # 原持仓方向
#                 realized = dir_sign * reduced * (fill_price - self.entry_price)
#                 self._realized_pnl_accum += realized
#
#         # ——————— 6) Inventory snapshot ———————
#         inventory_value = self.position * (self.last_price if self.last_price is not None else 0.0)
#         emit(InvUpdate(
#             timestamp=e.timestamp,
#             position=self.position,
#             inventory_value=inventory_value
#         ))
#
#         # ——————— 7) Track entry if opening a new position ———————
#         if self.entry_price is None and signed_qty != 0:
#             self.entry_price = fill_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.trade_vol = fill_price * signed_qty
#             self._total_entry_qty = abs(signed_qty)
#             self._realized_pnl_accum = 0.0
#             print(f"[EXEC] opened position at {self.entry_price:.2f}")
#
#         # ——————— 8) On position close, emit TradeReport ———————
#         if self.entry_price is not None and self.position == 0:
#             pnl = self._realized_pnl_accum
#             # 用“初始名义”做分母计算收益率，避免部分平仓导致分母变化
#             denom_notional = abs(self.entry_price * self._total_entry_qty)
#             ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
#             slippage_pct = (self._slip_accum / abs(self.trade_vol)) * 100 if self.trade_vol else 0.0
#             trade_type = "long" if self.entry_qty > 0 else "short"
#
#             tr = TradeReport(
#                 timestamp=self.entry_ts if self.entry_ts is not None else e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=fill_price,
#                 qty=abs(self._total_entry_qty),
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=0.0,
#                 cash_after=self.cash,
#                 trade_volume=self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#                 trade_type=trade_type,
#             )
#             print(
#                 f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                 f"→ TradeReport entry={self.entry_price:.2f} exit={fill_price:.2f}"
#             )
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.trade_vol = 0.0
#             self._slip_accum = 0.0
#             self._realized_pnl_accum = 0.0
#             self._total_entry_qty = 0.0



# # core/executor.py (with slippage)
#
# from typing import Dict, Optional
# from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
# class ExecutionAgent(Agent):
#     """
# #     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L,
# #     charging calibrated slippage = k * (signed_qty / ADV_i) off the mid‐price.
# #     """
#     def __init__(
#         self,
#         starting_cash:   float,
#         shortfall_coeff: float               = 0.0,
#         adv_map:         Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash       = starting_cash      # USD
#         self.position   = 0.0                # asset units
#         # slippage parameters
#         self.k          = shortfall_coeff    # calibrated impact coefficient
#         self.adv_map    = adv_map or {}      # {timestamp: ADV_i}
#         # last‐seen market data
#         self.last_price = None               # USD
#         self.last_mid   = None               # USD
#         # trade‐in‐progress
#         self.entry_price= None
#         self.entry_ts   = None
#         self.entry_qty  = 0.0
#         self.trade_vol   = 0.0                # USD notional at entry
#         self._slip_accum = 0.0
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # ——————— 1) On each BAR, record price & mid ———————
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             self.last_mid   = 0.5 * (e.O + e.C)
#             return
#
#         # ——————— 2) Only handle COMM events beyond here ———————
#         if not isinstance(e, COMM):
#             return
#         print(f"DEBUG COMM in executor → text={e.text!r}, ts={e.timestamp}")
#
#         #
#          # parse the order text: “BUY 1.0” or “SELL ALL”
#         #side_txt, qty_txt = e.text.split()
#         # parse the order text, stripping off any “#reason:xyz” suffix
#         parts = e.text.split("#reason:")
#         cmd = parts[0].strip()
#         reason = parts[1].strip() if len(parts) > 1 else "manual"
#         side_txt, qty_txt = cmd.split()
#
#
#         print(f"[EXEC] got COMM “{e.text}” @ last_price={self.last_price:.2f}")
#         side      = side_txt.upper()
#         side_sign =  1 if side=="BUY" else -1
#
#
#          # ——————— 3) Determine quantity (ALLOW “ALL”) ———————
#         fixed_cash = None
#         if qty_txt.upper() == "ALL":
#             # if side_sign > 0:
#             #     # BUY ALL: we’ll spend exactly self.cash _after_ slip
#             #     fixed_cash = self.cash
#             #     qty = self.cash / self.last_price
#             # else:
#             #     # SELL ALL: just dump your entire position
#             #     qty = abs(self.position)
#             # if we're on the opposite side, this is a close: buy back or sell off exactly our position
#             # 1) if we're on the opposite side, close it out
#
#             if self.position * side_sign < 0:
#                 qty = abs(self.position)
#              # 2) otherwise ALL = full‐cash sizing in that direction
#             elif side_sign > 0:
#                 fixed_cash = self.cash
#                 qty = self.cash / self.last_price
#             else:
#                 fixed_cash = self.cash
#                 qty = self.cash / self.last_price
#         else:
#             qty = float(qty_txt)
#         signed_qty = side_sign * qty
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f}")
#          # ——————— 4) Compute fill price ———————
#         if self.k>0 and self.adv_map:
#         # Compute slippage‐adjusted fill price
#             adv_i     = self.adv_map.get(e.timestamp, 1.0)
#             rho       = signed_qty / adv_i if adv_i else 0.0
#             slip_pct  = self.k * rho
#             fill_price = self.last_price * (1 + slip_pct)
#         else:
#          # calibration exec
#             fill_price = self.last_price
#          # — if this was a BUY ALL, recompute qty so you spend exactly fixed_cash
#         if fixed_cash is not None:
#             qty = fixed_cash / fill_price
#             signed_qty = side_sign * qty
#             if side_sign > 0:
#                 print(f"[EXEC] adjusted BUY ALL → qty={qty:.4f} after slip")
#             else:
#                 print(f"[EXEC] adjusted SELL ALL (short-all) → qty={qty:.4f} after slip")
#
#         trade_val = signed_qty * fill_price  # dollars exchanged
#         # --- accumulate slippage cost ---
#         # slippage_amt = signed_qty * (fill_price - self.last_price)
#         slip_dollars = abs(signed_qty * (fill_price - (self.last_price or fill_price)))
#         self._slip_accum += slip_dollars
#
#         # ——————— 5) Emit TradeFill & update cash/position ———————
#         emit(TradeFill(timestamp=e.timestamp, price=fill_price, qty=signed_qty, value=trade_val))
#         self.cash     -= trade_val
#         self.position += signed_qty
#
#         # ——————— 6) Inventory snapshot ———————
#         emit(InvUpdate(
#                 timestamp=e.timestamp,
#                 position        = self.position,
#                 inventory_value = self.position * self.last_price
#                 ))
#
#          # ——————— 7) Track entry if opening a new position ———————
#         if self.entry_price is None and signed_qty != 0:
#             self.entry_price = fill_price
#             self.entry_ts    = e.timestamp
#             self.entry_qty   = signed_qty
#             self.trade_vol    = fill_price * signed_qty
#             print(f"[EXEC] opened position at {self.entry_price:.2f}")
#
#          # ——————— 8) On position close, emit TradeReport ———————
#         if self.entry_price is not None and self.position == 0:
#             pnl     = self.entry_qty * (fill_price - self.entry_price)
#             ret_pct = (pnl / abs(self.entry_price * self.entry_qty)) * 100.0 \
#                       if self.entry_price and self.entry_qty else 0.0
#             slippage_pct = (self._slip_accum / abs(self.trade_vol)) * 100 if self.trade_vol else 0.0
#             # adv_i = self.adv_map.get(e.timestamp, 0.0)
#             # pos_turnover = self.trade_vol / self.starting_cash if self.starting_cash else 0.0
#
#             tr = TradeReport(
#                 timestamp               = e.timestamp,
#                 entry_ts                = self.entry_ts,
#                 exit_ts                 = e.timestamp,
#                 entry_price             = self.entry_price,
#                 exit_price              = fill_price,
#                 qty                     = abs(self.entry_qty),
#                 pnl                     = pnl,
#                 return_pct              = ret_pct,
#                 inventory_after         = 0.0,
#                 cash_after              = self.cash,
#                 trade_volume = self.trade_vol,
#                 slippage_cost=self._slip_accum,
#                 slippage_pct=slippage_pct,
#                 exit_reason=reason,
#             )
#             print(
#                 f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                 f"→ TradeReport entry={self.entry_price:.2f} exit={fill_price:.2f}"
#             )
#             emit(tr)
#
#             # reset for next trade
#             self.entry_price = None
#             self.entry_ts    = None
#             self.entry_qty   = 0.0
#             self.trade_vol    = 0.0
#             self._slip_accum = 0.0






# # core/executor.py (without slippage)
#
# from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
# class ExecutionAgent(Agent):
#     """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L.
#     Supports full‑cash sizing via “ALL”, tracks entry size for correct P/L,
#     and emits TradeReport with cash, inventory, and notional‐held details.
#     """
#     def __init__(self, starting_cash: float = 0.0):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash              = starting_cash   # USD
#         self.position          = 0.0             # BTC (units)
#         # last‑seen price
#         self.last_price        = None            # USD
#         # currently open trade
#         self.entry_price       = None            # USD
#         self.entry_ts          = None            # timestamp
#         self.entry_qty         = 0.0             # +units for long, −units for short
#         self.inv_held          = 0.0             # USD notional locked at entry
#
#     def observe(self, e: Event) -> bool:
#         # listen to price bars and COMM orders
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # ——————— 1) Update last_price on each BAR ———————
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             return
#
#         # ——————— 2) Only handle COMM events from here ———————
#         if not isinstance(e, COMM):
#             return
#
#         # parse COMM text: e.g. "BUY 1.0" or "SELL ALL"
#         side, qty_txt = e.text.split()
#         print(f"[EXEC] got COMM “{e.text}” @ last_price={self.last_price:.2f}")
#
#         # ——————— 3) Parse quantity (ALLOW “ALL”) ———————
#         if qty_txt.upper() == "ALL":
#             # invest all cash at current price
#             if side == "BUY":
#                 qty = (self.cash / self.last_price)
#             else:
#                 qty = abs(self.position)
#         else:
#             qty = float(qty_txt)
#         signed_qty = +qty if side.upper() == "BUY" else -qty
#
#         print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f}")
#
#         # ——————— 4) Execute fill: emit TradeFill, update cash & position ———————
#         trade_val = signed_qty * self.last_price  # dollars exchanged
#         emit(TradeFill(
#             timestamp=e.timestamp,
#             price=self.last_price,
#             qty=signed_qty,
#             value=trade_val
#         ))
#         self.cash     -= trade_val
#         self.position += signed_qty
#
#         # ——————— 5) Broadcast inventory snapshot ———————
#         emit(InvUpdate(
#             timestamp=e.timestamp,
#             position        = self.position,
#             inventory_value = self.position * self.last_price
#         ))
#
#         # ——————— 6) On entry, record size & notional held ———————
#         if self.entry_price is None and signed_qty != 0:
#             self.entry_price = self.last_price
#             self.entry_ts    = e.timestamp
#             self.entry_qty   = signed_qty
#             self.inv_held    = self.entry_price * signed_qty
#             print(f"[EXEC] opened position at {self.entry_price:.2f}")
#
#         # ——————— 7) On exit (position→0), emit TradeReport ———————
#         if self.entry_price is not None and self.position == 0:
#             # P/L = entry_qty * (exit_price - entry_price)
#             pnl     = self.entry_qty * (self.last_price - self.entry_price)
#             # return % = P/L ÷ (dollars at risk)
#             ret_pct = pnl / (self.entry_price * self.entry_qty) * 100.0
#
#             tr = TradeReport(
#                 timestamp               = e.timestamp,
#                 entry_ts                = self.entry_ts,
#                 exit_ts                 = e.timestamp,
#                 entry_price             = self.entry_price,
#                 exit_price              = self.last_price,
#                 qty                     = abs(self.entry_qty),
#                 pnl                     = pnl,
#                 return_pct              = ret_pct,
#                 inventory_after         = self.position * self.last_price,
#                 cash_after              = self.cash,
#                 inventory_held_in_trade = self.inv_held
#             )
#             print(
#                 f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                 f"→ TradeReport entry={self.entry_price:.2f} exit={self.last_price:.2f}"
#             )
#             emit(tr)
#
#             # reset entry state
#             self.entry_price = None
#             self.entry_ts    = None
#             self.entry_qty   = 0.0
#             self.inv_held    = 0.0
