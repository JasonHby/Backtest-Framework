# # core/executor.py
#
# from typing import Dict, Optional, List, Tuple
# from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
# class ExecutionAgent(Agent):
#     """
#     COMM → next‐BAR open fills + P/L.
#     Calibration pass (k=0) enqueues with slip_pct=0.
#     Real pass (k>0 + adv_map) enqueues with slip_pct=k·ρ.
#     All fills occur at the *next* BAR.open.
#     """
#     def __init__(
#         self,
#         starting_cash:   float,
#         shortfall_coeff: float               = 0.0,
#         adv_map:         Optional[Dict[float, float]] = None
#     ):
#         super().__init__(name="EXEC")
#         # portfolio state
#         self.cash        = starting_cash
#         self.position    = 0.0
#
#         # slippage parameters
#         self.k           = shortfall_coeff
#         self.adv_map     = adv_map or {}
#
#         # last‐seen market data
#         self.last_price  = None  # most recent bar close
#         self.last_mid    = None  # midpoint of that bar
#
#         # trade‐in‐progress
#         self.entry_price = None
#         self.entry_ts    = None
#         self.entry_qty   = 0.0
#         self.inv_held    = 0.0
#
#         # buffer orders until next bar’s open
#         self._pending_orders: List[Tuple[float, float]] = []  # (slip_pct, signed_qty)
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         print(f"[DEBUG] main() got event {type(e).__name__} ts={getattr(e, 'timestamp', None)}")
#         # ————— 1) BAR event: execute all queued fills at this bar’s open —————
#         if isinstance(e, BAR):
#             # Show the raw bar data
#             print(f"[DEBUG][BAR] BAR ts={e.timestamp}  open={e.O:.2f}  close={e.C:.2f}  pending={self._pending_orders}")
#             for slip_pct, signed_qty in self._pending_orders:
#                 print(f"[DEBUG][BAR] → Filling qty={signed_qty:.4f} at fill_price = open* (1+slip_pct)")
#                 print(f"[DEBUG][BAR]    open={e.O:.2f}, slip_pct={slip_pct:.6f}")
#                 fill_price = e.O * (1 + slip_pct)
#                 print(f"[DEBUG][BAR]    computed fill_price={fill_price:.2f}")
#                 trade_val  = signed_qty * fill_price
#
#                 emit(TradeFill(
#                     timestamp=e.timestamp,
#                     price=fill_price,
#                     qty=signed_qty,
#                     value=trade_val
#                 ))
#                 self.cash     -= trade_val
#                 self.position += signed_qty
#
#                 emit(InvUpdate(
#                     timestamp        = e.timestamp,
#                     position         = self.position,
#                     inventory_value  = self.position * self.last_price
#                 ))
#
#                 # entry
#                 if self.entry_price is None and signed_qty != 0:
#                     self.entry_price = fill_price
#                     self.entry_ts    = e.timestamp
#                     self.entry_qty   = signed_qty
#                     self.inv_held    = fill_price * signed_qty
#
#                 # exit & report
#                 if self.entry_price is not None and self.position == 0:
#                     pnl     = self.entry_qty * (fill_price - self.entry_price)
#                     ret_pct = (pnl / abs(self.entry_price * self.entry_qty)) * 100.0 \
#                               if self.entry_price and self.entry_qty else 0.0
#
#                     tr = TradeReport(
#                         timestamp               = e.timestamp,
#                         entry_ts                = self.entry_ts,
#                         exit_ts                 = e.timestamp,
#                         entry_price             = self.entry_price,
#                         exit_price              = fill_price,
#                         qty                     = abs(self.entry_qty),
#                         pnl                     = pnl,
#                         return_pct              = ret_pct,
#                         inventory_after         = 0.0,
#                         cash_after              = self.cash,
#                         inventory_held_in_trade = self.inv_held,
#                     )
#                     emit(tr)
#
#                     # reset for next trade
#                     self.entry_price = None
#                     self.entry_ts    = None
#                     self.entry_qty   = 0.0
#                     self.inv_held    = 0.0
#
#             # clear for next bar
#             self._pending_orders.clear()
#             print(f"[DEBUG][BAR] Pending after clear: {self._pending_orders}")
#
#             # update last‐bar data for next COMM
#             self.last_price = e.C
#             self.last_mid   = 0.5 * (e.O + e.C)
#             return
#
#         # ————— 2) COMM event: enqueue at next bar’s open —————
#         if isinstance(e, COMM):
#             # are we in the real pass?
#             is_real = (self.k > 0 and bool(self.adv_map))
#             print(f"[DEBUG][COMM] is_real={is_real}  k={self.k:.6f}  adv_map_keys={list(self.adv_map.keys())[:3]}")
#             side_txt, qty_txt = e.text.split()
#             side_sign = 1 if side_txt.upper() == "BUY" else -1
#
#             # determine quantity
#             if qty_txt.upper() == "ALL":
#                 qty = (self.cash / self.last_price) if side_sign > 0 else abs(self.position)
#             else:
#                 qty = float(qty_txt)
#             signed_qty = side_sign * qty
#
#             if is_real:
#                 # compute slippage fraction ρ = signed_qty/ADV
#                 adv_i    = self.adv_map.get(e.timestamp, 1.0)
#                 rho      = signed_qty / adv_i if adv_i else 0.0
#                 slip_pct = self.k * rho
#                 print(f"[DEBUG][COMM] Real pass → rho={rho:.6f}, slip_pct={slip_pct:.6f}")
#             else:
#                 # calibration pass: no slippage
#                 slip_pct = 0.0
#                 print(f"[DEBUG][COMM] Calibration pass → slip_pct=0.0")
#
#             # enqueue for fill at next BAR.open
#             self._pending_orders.append((slip_pct, signed_qty))
#             print(f"[DEBUG][COMM] Pending orders now: {self._pending_orders}")
#             return
#
#         # ignore other event types
#         return


# core/executor.py

from typing import Dict, Optional
from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
from core.fsm       import Agent
from core.plumbing  import emit

class ExecutionAgent(Agent):
    """
#     Turns COMM “BUY qty” / “SELL qty” into simulated fills + P/L,
#     charging calibrated slippage = k * (signed_qty / ADV_i) off the mid‐price.
#     """
    def __init__(
        self,
        starting_cash:   float,
        shortfall_coeff: float               = 0.0,
        adv_map:         Optional[Dict[float, float]] = None
    ):
        super().__init__(name="EXEC")
        # portfolio state
        self.cash       = starting_cash      # USD
        self.position   = 0.0                # asset units
        # slippage parameters
        self.k          = shortfall_coeff    # calibrated impact coefficient
        self.adv_map    = adv_map or {}      # {timestamp: ADV_i}
        # last‐seen market data
        self.last_price = None               # USD
        self.last_mid   = None               # USD
        # trade‐in‐progress
        self.entry_price= None
        self.entry_ts   = None
        self.entry_qty  = 0.0
        self.inv_held   = 0.0                # USD notional at entry

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, COMM))

    def main(self, e: Event) -> None:
        # ——————— 1) On each BAR, record price & mid ———————
        if isinstance(e, BAR):
            self.last_price = e.C
            self.last_mid   = 0.5 * (e.O + e.C)
            return

        # ——————— 2) Only handle COMM events beyond here ———————
        if not isinstance(e, COMM):
            return
#
         # parse the order text: “BUY 1.0” or “SELL ALL”
        side_txt, qty_txt = e.text.split()
        print(f"[EXEC] got COMM “{e.text}” @ last_price={self.last_price:.2f}")
        side      = side_txt.upper()
        side_sign =  1 if side=="BUY" else -1


         # ——————— 3) Determine quantity (ALLOW “ALL”) ———————
        if qty_txt.upper() == "ALL":
            if side_sign > 0:
                qty = (self.cash / self.last_price)
            else:
                qty = abs(self.position)
        else:
            qty = float(qty_txt)
        signed_qty = side_sign * qty
        print(f"[EXEC] sizing → qty={qty:.4f}, signed_qty={signed_qty:.4f}")

#         # # ——————— 4) Compute slippage‐adjusted fill price ———————
#         # adv_i     = self.adv_map.get(e.timestamp, 1.0)
#         # rho       = signed_qty / adv_i if adv_i else 0.0
#         # slip_pct  = self.k * rho
#         # fill_price= self.last_mid * (1 + side_sign * slip_pct)
#         # trade_val = signed_qty * fill_price  # dollars exchanged
#
         # ——————— 4) Compute fill price ———————
        if self.k>0 and self.adv_map:
        # Compute slippage‐adjusted fill price
            adv_i     = self.adv_map.get(e.timestamp, 1.0)
            rho       = signed_qty / adv_i if adv_i else 0.0
            slip_pct  = self.k * rho
            #fill_price= self.last_mid  * (1 + slip_pct)
            fill_price = self.last_price * (1 + slip_pct)
        else:
         # calibration exec
            fill_price = self.last_price

        trade_val = signed_qty * fill_price  # dollars exchanged

         # ——————— 5) Emit TradeFill & update cash/position ———————
        emit(TradeFill(timestamp=e.timestamp, price=fill_price, qty=signed_qty, value=trade_val))
        self.cash     -= trade_val
        self.position += signed_qty

        # ——————— 6) Inventory snapshot ———————
        emit(InvUpdate(
                timestamp=e.timestamp,
                position        = self.position,
                inventory_value = self.position * self.last_price
                ))

         # ——————— 7) Track entry if opening a new position ———————
        if self.entry_price is None and signed_qty != 0:
            self.entry_price = fill_price
            self.entry_ts    = e.timestamp
            self.entry_qty   = signed_qty
            self.inv_held    = fill_price * signed_qty
            print(f"[EXEC] opened position at {self.entry_price:.2f}")

         # ——————— 8) On position close, emit TradeReport ———————
        if self.entry_price is not None and self.position == 0:
            pnl     = self.entry_qty * (fill_price - self.entry_price)
            ret_pct = (pnl / abs(self.entry_price * self.entry_qty)) * 100.0 \
                      if self.entry_price and self.entry_qty else 0.0

            tr = TradeReport(
                timestamp               = e.timestamp,
                entry_ts                = self.entry_ts,
                exit_ts                 = e.timestamp,
                entry_price             = self.entry_price,
                exit_price              = fill_price,
                qty                     = abs(self.entry_qty),
                pnl                     = pnl,
                return_pct              = ret_pct,
                inventory_after         = 0.0,
                cash_after              = self.cash,
                inventory_held_in_trade = self.inv_held,
            )
            print(
                f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
                f"→ TradeReport entry={self.entry_price:.2f} exit={fill_price:.2f}"
            )
            emit(tr)

            # reset for next trade
            self.entry_price = None
            self.entry_ts    = None
            self.entry_qty   = 0.0
            self.inv_held    = 0.0






# # core/executor.py
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





# # core/executor.py
# from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
# class ExecutionAgent(Agent):
#     def __init__(self, starting_cash: float = 0.0):
#         super().__init__(name="EXEC")
#         self.cash        = starting_cash
#         self.position   = 0.0
#         self.inv_held      = 0.0
#         self.last_price  = None
#         self.entry_price = None
#         self.entry_ts    = None
#         self.entry_qty = 0.0
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, COMM))
#
#     def main(self, e: Event) -> None:
#         # —————————— handle BAR ——————————
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             #print(f"[EXEC] saw BAR @ {e.C:.2f}")
#             return
#
#         # —————————— handle COMM ——————————
#         # this is your buy/sell signal
#         if not isinstance(e, COMM):
#             return
#         print(f"[EXEC] got COMM “{e.text}” @ last_price={self.last_price:.2f}")
#         # parts   = e.text.split()
#         # side    = parts[0].upper()
#         # qty_txt = parts[1]
#         side, qty_txt = e.text.split()
#         # # parse qty (ALLOW “ALL”)
#         # try:
#         #     qty = float(qty_txt)
#         # except ValueError:
#         #     qty = (self.cash / self.last_price) if (self.cash and self.last_price) else 0.0
#         # signed_qty = +qty if side=="BUY" else -qty
#
#         # parse qty (ALLOW “ALL”)
#         if qty_txt.upper() == "ALL":
#             qty = (self.cash / self.last_price) if self.last_price else 0.0
#         else:
#             qty = float(qty_txt)
#         signed_qty = +qty if side == "BUY" else -qty
#
#         # ———————— 3) TradeFill & book‑keeping ————————
#         # compute the cash exchanged at this fill
#         trade_val = signed_qty * self.last_price
#
#         # emit the fill event (you can extend TradeFill to include `value`)
#         emit(TradeFill(
#             timestamp=e.timestamp,
#             price=self.last_price,
#             qty=signed_qty,
#             value=trade_val
#         ))
#
#         # update cash and position value
#         self.cash -= trade_val
#         self.position += signed_qty
#         # # —————————— TradeFill & book‑keeping ——————————
#         # emit(TradeFill(timestamp=e.timestamp, price=self.last_price, qty=signed_qty))
#         # self.cash      -= self.last_price * signed_qty
#         # self.inventory += signed_qty
#         # #print(f"[EXEC] → TradeFill qty={signed_qty:.3f},  cash={self.cash:.2f},  inv={self.inventory:.3f}")
#
#         # ———————— 4) inventory snapshot ————————
#         emit(InvUpdate(
#             timestamp=e.timestamp,
#             position=self.position,
#             inventory_value=self.position * self.last_price
#         ))
#         #print(f"[EXEC] → InvUpdate pos={self.position:.3f}  inv_val={self.inventory_value:.2f}")
#
#         # ———————— 5) Track entry if opening a new position ————————
#         if self.entry_price is None and signed_qty != 0:
#             self.entry_price = self.last_price
#             self.entry_ts = e.timestamp
#             self.entry_qty = signed_qty
#             self.inv_held = self.entry_price * signed_qty
#             print(f"[EXEC] opened position at {self.entry_price:.2f}")
#
#         # ———————— 6) Possibly emit TradeReport on closing ————————
#         if self.entry_price is not None and self.position == 0:
#             pnl     = self.entry_qty * (self.last_price - self.entry_price)
#             ret_pct = pnl / (self.entry_price * self.entry_qty) * 100.0
#             tr = TradeReport(
#                 timestamp=e.timestamp,
#                 entry_ts=self.entry_ts,
#                 exit_ts=e.timestamp,
#                 entry_price=self.entry_price,
#                 exit_price=self.last_price,
#                 qty=abs(self.entry_qty),  # size of the trade
#                 pnl=pnl,
#                 return_pct=ret_pct,
#                 inventory_after=self.position * self.last_price,
#                 cash_after=self.cash,
#                 inventory_held_in_trade=self.inv_held
#             )
#             print(
#                 f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}% "
#                 f"→ TradeReport entry={self.entry_price:.2f} exit={self.last_price:.2f}"
#             )
#             emit(tr)
#
#             # — reset for the next trade —
#             self.entry_price = None
#             self.entry_ts = None
#             self.entry_qty = 0.0
#             self.inv_held = 0.0

    # def main(self, e: Event) -> None:
    #     # 1) If it's a BAR, just update last_price
    #     if isinstance(e, BAR):
    #         self.last_price = e.C
    #         print(f"[EXEC] saw BAR @ {e.C:.2f}")
    #         return
    #
    #     # 2) Otherwise it's a COMM
    #     print(f"[EXEC] got COMM “{e.text}” @ last_price={self.last_price:.2f}")
    #     parts = e.text.split()
    #     side  = parts[0].upper()
    #     qty_txt = parts[1]
    #
    #     # parse qty (ALLOW “ALL”)
    #     try:
    #         qty = float(qty_txt)
    #     except ValueError:
    #         # if user said "ALL" and we have a price, use all cash
    #         qty = (self.cash / self.last_price) if self.cash and self.last_price else 0.0
    #
    #     signed_qty = +qty if side == "BUY" else -qty
    #     px = self.last_price
    #     ts = e.timestamp
    #
    #     # 3) Emit TradeFill with the correct signature
    #     emit(TradeFill(
    #         timestamp = ts,
    #         price=px,
    #         qty       = signed_qty,
    #     ))
    #
    #     # update book‐keeping
    #     self.cash      -= px * signed_qty
    #     self.inventory += signed_qty
    #     print(f"[EXEC] → TradeFill qty={signed_qty:.3f},  cash={self.cash:.2f},  inv={self.inventory:.3f}")
    #
    #     # 4) Emit InvUpdate with both timestamp and inventory
    #     emit(InvUpdate(
    #         timestamp = ts,
    #         inventory = self.inventory,
    #     ))
    #     print(f"[EXEC] → InvUpdate inventory={self.inventory:.3f}")
    #
    #     # 5) Track an opening price
    #     if self.entry_price is None and signed_qty != 0:
    #         self.entry_price = px
    #         self.entry_ts    = ts
    #         print(f"[EXEC] opened position at {self.entry_price:.2f}")
    #
    #     # 6) If we just flattened out, emit a TradeReport
    #     if self.entry_price is not None and self.inventory == 0:
    #         pnl     = signed_qty * (px - self.entry_price)
    #         ret_pct = (pnl / self.entry_price) * 100.0
    #         print(f"[EXEC] closing P/L={pnl:.2f}  {ret_pct:.2f}%")
    #         tr = TradeReport(
    #             timestamp       = ts,
    #             entry_ts        = self.entry_ts,
    #             exit_ts         = ts,
    #             entry_price     = self.entry_price,
    #             exit_price      = px,
    #             qty             = qty,
    #             pnl             = pnl,
    #             return_pct      = ret_pct,
    #             inventory_after = self.inventory,
    #         )
    #         print(f"[EXEC] -> TradeReport: entry={self.entry_price:.2f}  exit={px:.2f}  qty={qty:.3f}")
    #         emit(tr)
    #         # reset
    #         self.entry_price = None
    #         self.entry_ts    = None




# # core/executor.py
# from core.events    import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from core.fsm       import Agent
# from core.plumbing  import emit
#
# class ExecutionAgent(Agent):
#     def __init__(self, starting_cash: float = 0.0):
#         super().__init__(name="EXEC")
#         self.cash        = starting_cash
#         self.inventory   = 0.0
#         self.last_price  = None
#         self.entry_price = None
#         self.entry_ts    = None
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, BAR) or isinstance(e, COMM)
#
#     def main(self, e: Event) -> None:
#         # 1) If it's a BAR, just update last_price
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             return
#
#         # 2) Otherwise it must be a COMM, parse it
#         #    Format: "BUY qty" or "SELL qty", qty may be "ALL"
#         parts = e.text.split()
#         side, qty_txt = parts[0].upper(), parts[1]
#         try:
#             qty = float(qty_txt)
#         except ValueError:
#             qty = (self.cash / self.last_price) if self.last_price else 0.0
#         signed_qty = +qty if side == "BUY" else -qty
#         px = self.last_price
#         ts = e.timestamp
#
#         # 3) Emit a real TradeFill
#         emit(TradeFill(timestamp=ts, price=px, qty=signed_qty))
#         self.cash      -= px * signed_qty
#         self.inventory += signed_qty
#
#         # 4) Tell the recorder “inventory changed, snapshot equity now”
#         emit(InvUpdate(timestamp=ts))
#
#         # 5) Track entry → exit
#         if self.entry_price is None and signed_qty != 0:
#             # just opened a new position
#             self.entry_price = px
#             self.entry_ts    = ts
#
#         # if we just flattened out, emit a TradeReport
#         if self.entry_price is not None and self.inventory == 0:
#             pnl      = signed_qty * (px - self.entry_price)
#             ret_pct  = (pnl / self.entry_price) * 100
#             tr = TradeReport(
#                 timestamp        = ts,
#                 entry_ts         = self.entry_ts,
#                 exit_ts          = ts,
#                 entry_price      = self.entry_price,
#                 exit_price       = px,
#                 qty              = qty,
#                 pnl              = pnl,
#                 return_pct       = ret_pct,
#                 inventory_after  = self.inventory,
#             )
#             emit(tr)
#             # reset entry markers
#             self.entry_price = None
#             self.entry_ts    = None
#
#




# # core/executor.py
#
# import time
# from .events import Event, BAR, COMM, TradeFill, InvUpdate, TradeReport
# from .fsm    import Agent
# from .plumbing import emit
#
# class ExecutionAgent(Agent):
#     def __init__(self, starting_cash: float = 0.0):
#         super().__init__(name="EXEC")
#         self.cash        = starting_cash
#         self.inventory   = 0.0
#         self.last_price  = None
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, BAR) or isinstance(e, COMM)
#
#     def main(self, e) -> None:
#         # 1) If it's a bar, update last_price
#         if isinstance(e, BAR):
#             self.last_price = e.C
#             return
#
#         # 2) Otherwise if it's not a COMM, drop it
#         if not isinstance(e, COMM):
#             return
#
#         # 3) Now safe to read e.text
#         parts = e.text.split()
#         side  = parts[0].upper()
#
#         # ... rest of your all-in logic ...
#         try:
#             qty = float(parts[1])
#         except Exception:
#             if parts[1].upper() == "ALL" and self.last_price:
#                 qty = self.cash / self.last_price
#             else:
#                 qty = 0.0
#
#         px = self.last_price or 0.0
#         if side == "BUY":
#             self.inventory += qty
#             self.cash      -= qty * px
#             signed_qty     = +qty
#         elif side == "SELL":
#             self.inventory -= qty
#             self.cash      += qty * px
#             signed_qty     = -qty
#         else:
#             return
#
#         emit(TradeFill(time.time(), px, signed_qty))
#         emit(InvUpdate( time.time(), self.inventory))
