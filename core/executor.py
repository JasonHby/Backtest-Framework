from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from core.events import (
    BAR,
    COMM,
    Event,
    InvUpdate,
    OrderRequest,
    PartialExitReport,
    QuantityType,
    TradeFill,
    TradeReport,
)
from core.fsm import Agent
from core.portfolio import PortfolioManager
from core.plumbing import emit


def parse_comm_text(text: str) -> tuple[str, str, str, set[str]]:
    """
    Parse a legacy COMM payload like:
      "BUY ALL #reason:reverse_ema_flip #at_open"
    """
    parts = text.strip().split()
    reason = "manual"
    flags: set[str] = set()
    non_flag: list[str] = []

    for token in parts:
        if token.startswith("#"):
            if token.startswith("#reason:"):
                reason = token.split(":", 1)[1]
            else:
                flags.add(token.lstrip("#"))
        else:
            non_flag.append(token)

    if len(non_flag) >= 2:
        return non_flag[0].upper(), non_flag[1], reason, flags
    if len(non_flag) == 1:
        return non_flag[0].upper(), "ALL", reason, flags
    raise ValueError(f"Cannot parse COMM text: '{text}'")


@dataclass
class MarketState:
    last_price: Optional[float] = None
    last_open: Optional[float] = None


class ExecutionAgent(Agent):
    """
    Generic event-driven execution layer.

    Supports:
      - structured OrderRequest events
      - legacy COMM compatibility
      - multiple portfolio ids
      - all-cash entries and arbitrary partial exits
      - the existing ADV shortfall slippage model
    """

    def __init__(
        self,
        starting_cash: float,
        shortfall_coeff: float = 0.0,
        adv_map: Optional[Dict[float, float]] = None,
        default_portfolio_id: str = "main",
        default_security: Optional[str] = None,
    ):
        super().__init__(name="EXEC")
        self.starting_cash = float(starting_cash)
        self.k = float(shortfall_coeff)
        self.adv_map = adv_map or {}
        self.default_portfolio_id = default_portfolio_id
        self.default_security = default_security
        self.market: Dict[str, MarketState] = {}
        self.portfolio_manager = PortfolioManager(starting_cash=self.starting_cash)
        self.last_seen_security: Optional[str] = default_security
        self._close_intent_reasons = {
            "stop_loss",
            "breakeven",
            "giveback",
            "take_profit",
            "take_profit1",
            "ema_flip",
        }

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, COMM, OrderRequest))

    def main(self, e: Event) -> None:
        if isinstance(e, BAR):
            self.market[e.security] = MarketState(last_price=float(e.C), last_open=float(e.O))
            self.last_seen_security = e.security
            return

        if isinstance(e, COMM):
            order = self._legacy_comm_to_order(e)
            if order is None:
                return
        elif isinstance(e, OrderRequest):
            order = e
        else:
            return

        self._execute_order(order)

    def _resolve_security(self) -> Optional[str]:
        if self.default_security:
            return self.default_security
        if self.last_seen_security:
            return self.last_seen_security
        if len(self.market) == 1:
            return next(iter(self.market))
        return None

    def _legacy_comm_to_order(self, e: COMM) -> Optional[OrderRequest]:
        try:
            side_txt, qty_txt, reason, flags = parse_comm_text(e.text)
        except ValueError as exc:
            print(f"[EXEC] malformed COMM '{e.text}': {exc}")
            return None

        security = self._resolve_security()
        if not security:
            print(f"[EXEC] skip COMM '{e.text}': no security context available")
            return None

        reduce_only = reason in self._close_intent_reasons
        order_type = "next_open" if "at_open" in flags else "market"

        if qty_txt.upper() == "ALL":
            quantity = None
            quantity_type: QuantityType = "all_position" if reduce_only else "all_cash"
        else:
            quantity = float(qty_txt)
            quantity_type = "units"

        return OrderRequest(
            timestamp=e.timestamp,
            sender=e.sender,
            portfolio_id=self.default_portfolio_id,
            security=security,
            side=side_txt,
            quantity=quantity,
            quantity_type=quantity_type,
            order_type=order_type,
            reduce_only=reduce_only,
            reason=reason,
        )

    def _resolve_order_quantity(
        self,
        order: OrderRequest,
        portfolio: PortfolioState,
        old_pos: float,
        base_price: float,
    ) -> tuple[float, Optional[float], int]:
        side_sign = 1 if order.side == "BUY" else -1
        fixed_cash: Optional[float] = None
        qty = 0.0

        if order.quantity_type == "target_position":
            target = float(order.quantity or 0.0)
            delta = target - old_pos
            if abs(delta) <= 1e-12:
                return 0.0, None, side_sign
            side_sign = 1 if delta > 0 else -1
            qty = abs(delta)
        elif order.quantity_type == "all_position":
            qty = abs(old_pos)
        elif order.quantity_type == "percent_position":
            qty = abs(old_pos) * max(0.0, float(order.quantity or 0.0))
        elif order.quantity_type == "all_cash":
            fixed_cash = max(0.0, portfolio.cash)
            qty = fixed_cash / base_price if base_price > 0 else 0.0
        elif order.quantity_type == "percent_cash":
            fixed_cash = max(0.0, portfolio.cash) * max(0.0, float(order.quantity or 0.0))
            qty = fixed_cash / base_price if base_price > 0 else 0.0
        else:
            qty = abs(float(order.quantity or 0.0))

        if order.reduce_only:
            if abs(old_pos) <= 1e-12:
                return 0.0, None, side_sign
            if old_pos * side_sign >= 0:
                side_sign = -1 if old_pos > 0 else 1
            qty = min(qty, abs(old_pos))
        else:
            # Keep the current engine behavior: one order can close but not flip.
            if old_pos * side_sign < 0:
                qty = min(qty, abs(old_pos))

        return qty, fixed_cash, side_sign

    def _execute_order(self, order: OrderRequest) -> None:
        market = self.market.get(order.security)
        if market is None:
            print(f"[EXEC] skip order: no market data for {order.security}")
            return

        base_price = (
            market.last_open
            if order.order_type == "next_open" and market.last_open is not None
            else market.last_price
        )
        if not base_price:
            print(f"[EXEC] skip order: no reference price for {order.security}")
            return

        portfolio = self.portfolio_manager.get_portfolio(order.portfolio_id)
        trade = portfolio.get_trade_state(order.security)
        old_pos = portfolio.get_position(order.security)

        qty, fixed_cash, side_sign = self._resolve_order_quantity(order, portfolio, old_pos, base_price)
        if qty <= 0.0:
            return

        signed_qty_req = side_sign * qty
        adv_i = self.adv_map.get(order.timestamp) if self.adv_map else None
        if self.k > 0 and adv_i and adv_i > 0 and signed_qty_req != 0.0:
            rho_abs = abs(signed_qty_req) / adv_i
            slip_frac = self.k * rho_abs
            fill_price = base_price * (1.0 + (1.0 if signed_qty_req > 0 else -1.0) * slip_frac)
        else:
            fill_price = base_price

        if fixed_cash is not None and fill_price > 0:
            qty = fixed_cash / fill_price
            signed_qty_req = side_sign * qty

        slip_cost_fill = abs(fill_price - base_price) * abs(signed_qty_req)
        trade.slip_accum += slip_cost_fill
        trade.slip_pct_accum += abs((fill_price / base_price) - 1.0)
        trade.slip_n += 1

        trade_val = signed_qty_req * fill_price
        prev_pos = old_pos

        if trade.entry_price is None and abs(prev_pos) <= 1e-12 and signed_qty_req != 0.0:
            trade.cash_at_entry = portfolio.cash
            trade.entry_base_price = base_price
            trade.vwap_exit_num = 0.0
            trade.vwap_exit_den = 0.0
            trade.lost_qty_open_signed = 0.0
            trade.lost_qty_total_signed = 0.0
            trade.qty_effect_cost_accum = 0.0

        reduced = 0.0
        if trade.entry_price is not None and prev_pos != 0.0 and (prev_pos * signed_qty_req) < 0.0:
            reduced = min(abs(prev_pos), abs(signed_qty_req))
            if reduced > 0.0:
                trade.vwap_exit_num += reduced * fill_price
                trade.vwap_exit_den += reduced

                r = reduced / abs(prev_pos)
                dec = r * trade.lost_qty_open_signed
                sign_entry = 1.0 if trade.entry_qty > 0 else -1.0
                entry_base = trade.entry_base_price if trade.entry_base_price is not None else base_price
                unit_ret_cash_slice = sign_entry * (base_price - entry_base)
                qty_effect_slice = unit_ret_cash_slice * dec
                trade.qty_effect_cost_accum += qty_effect_slice
                trade.lost_qty_open_signed -= dec

        emit(
            TradeFill(
                timestamp=order.timestamp,
                price=fill_price,
                qty=signed_qty_req,
                value=trade_val,
                portfolio_id=order.portfolio_id,
                security=order.security,
                side=order.side,
                reason=order.reason,
                base_price=base_price,
                adv_at_fill=adv_i,
                metadata=dict(order.metadata),
            )
        )

        portfolio.cash -= trade_val
        new_pos = prev_pos + signed_qty_req
        portfolio.set_position(order.security, new_pos)

        added_abs = max(0.0, abs(new_pos) - abs(prev_pos))
        if added_abs > 0.0:
            delta_q_signed = added_abs * (fill_price / base_price - 1.0)
            if delta_q_signed != 0.0:
                trade.lost_qty_open_signed += delta_q_signed
                trade.lost_qty_total_signed += delta_q_signed

        if prev_pos * signed_qty_req < 0.0 and abs(signed_qty_req) > abs(prev_pos):
            trade.entry_price = fill_price
            trade.entry_ts = order.timestamp
            trade.entry_qty = new_pos
            trade.trade_vol = fill_price * new_pos
            trade.total_entry_qty = abs(new_pos)
            trade.cash_at_entry = portfolio.cash
            trade.entry_base_price = base_price
            trade.lost_qty_open_signed = 0.0
            trade.lost_qty_total_signed = 0.0
            trade.qty_effect_cost_accum = 0.0

            remainder = signed_qty_req + prev_pos
            trade.slip_accum = abs(fill_price - base_price) * abs(remainder)
            trade.slip_pct_accum = abs((fill_price / base_price) - 1.0)
            trade.slip_n = 1

        if reduced > 0.0 and abs(new_pos) > 1e-12:
            dir_sign = 1.0 if prev_pos > 0 else -1.0
            slice_realized = dir_sign * reduced * (fill_price - (trade.entry_price or fill_price))
            emit(
                PartialExitReport(
                    timestamp=order.timestamp,
                    entry_ts=float(trade.entry_ts or order.timestamp),
                    exit_ts=order.timestamp,
                    entry_price=float(trade.entry_price or fill_price),
                    exit_price=fill_price,
                    qty=reduced,
                    pnl=slice_realized,
                    reason=order.reason,
                    trade_type=("long" if prev_pos > 0 else "short"),
                    cash_after=portfolio.cash,
                    position_after=new_pos,
                    portfolio_id=order.portfolio_id,
                    security=order.security,
                    slippage_cost=abs(fill_price - base_price) * reduced,
                    slippage_pct=abs((fill_price / base_price) - 1.0) * 100.0,
                )
            )

        if trade.entry_price is None and signed_qty_req != 0.0 and abs(new_pos) > 1e-12:
            trade.entry_price = fill_price
            trade.entry_ts = order.timestamp
            trade.entry_qty = signed_qty_req
            trade.trade_vol = fill_price * signed_qty_req
            trade.total_entry_qty = abs(signed_qty_req)
            trade.entry_base_price = base_price

        inventory_value = new_pos * float(market.last_price or fill_price)
        emit(
            InvUpdate(
                timestamp=order.timestamp,
                position=new_pos,
                inventory_value=inventory_value,
                portfolio_id=order.portfolio_id,
                security=order.security,
                cash=portfolio.cash,
                equity=portfolio.cash + inventory_value,
            )
        )

        if trade.entry_price is not None and abs(new_pos) <= 1e-12:
            pnl = (
                portfolio.cash - float(trade.cash_at_entry or portfolio.cash)
                if trade.cash_at_entry is not None
                else 0.0
            )
            denom_notional = abs(trade.entry_price * trade.total_entry_qty)
            ret_pct = (pnl / denom_notional) * 100.0 if denom_notional else 0.0
            slippage_pct = (trade.slip_pct_accum / trade.slip_n) * 100.0 if trade.slip_n else 0.0
            trade_type = "long" if trade.entry_qty > 0 else "short"
            exit_price_last = fill_price
            exit_price_vwap = (
                trade.vwap_exit_num / trade.vwap_exit_den
                if trade.vwap_exit_den > 0
                else exit_price_last
            )
            quantity_effect_cost = trade.qty_effect_cost_accum
            total_effect_cost = trade.slip_accum + quantity_effect_cost

            emit(
                TradeReport(
                    timestamp=float(trade.entry_ts or order.timestamp),
                    entry_ts=float(trade.entry_ts or order.timestamp),
                    exit_ts=order.timestamp,
                    entry_price=float(trade.entry_price),
                    exit_price=exit_price_last,
                    qty=abs(trade.total_entry_qty),
                    pnl=pnl,
                    return_pct=ret_pct,
                    inventory_after=0.0,
                    cash_after=portfolio.cash,
                    trade_volume=trade.trade_vol,
                    slippage_cost=trade.slip_accum,
                    slippage_pct=slippage_pct,
                    portfolio_id=order.portfolio_id,
                    security=order.security,
                    exit_reason=order.reason,
                    trade_type=trade_type,
                    exit_price_vwap=exit_price_vwap,
                    quantity_effect_cost=quantity_effect_cost,
                    total_effect_cost=total_effect_cost,
                )
            )
            trade.reset()
