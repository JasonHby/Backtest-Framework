# core/calibration.py
from collections import deque
from typing       import List, Tuple, Dict
from core.fsm     import Agent
from core.events  import Event, BAR, TradeFill

class CalibrationAgent(Agent):
    """
    Pass #1: build ADV_i, mid‐price, then on each TradeFill record
    (ρ_i = qty/ADV_i,   s_i = (fill_price−mid)/mid) and stash ADV_i.
    """
    def __init__(self, bar_per_day: int = 24):
        super().__init__(name="CALIB")
        # rolling buffer of the last `bar_per_day` BAR.volume values (24 hours)
        self._vol_buffer: deque[float] = deque(maxlen=bar_per_day)
        self.last_adv:     float       = 0.0    # ADV_i at the current BAR
        self.last_mid:     float       = 0.0    # mid price at the current BAR

        # collected calibration samples
        self.samples: List[Tuple[float, float]] = []
        # mapping from fill‐timestamps to the ADV used
        self.adv_map: Dict[float, float] = {}

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, TradeFill))

    def main(self, e: Event) -> None:
        # 1) On each BAR, update rolling ADV and mid
        if isinstance(e, BAR):
            # push this bar's volume into the rolling window
            self._vol_buffer.append(getattr(e, "volume", 0.0))
            if self._vol_buffer:
                # average volume per bar * bars_per_day = ADV_i
                avg_per_bar = sum(self._vol_buffer) / len(self._vol_buffer)
                self.last_adv = avg_per_bar * self._vol_buffer.maxlen
            else:
                self.last_adv = 0.0

            # mid‐price = (O + C) / 2
            self.last_mid = 0.5 * (e.O + e.C)
            return

        # 2) On each TradeFill, record (ρ_i, s_i) and stash ADV_i
        if isinstance(e, TradeFill):
            adv_i = self.last_adv or 1.0
            mid_i = self.last_mid or e.price

            rho      = e.qty / adv_i
            slippage = (e.price - mid_i) / mid_i

            # store for regression
            self.samples.append((rho, slippage))
            # also make available to executor later
            self.adv_map[e.timestamp] = adv_i

    def compute_k(self) -> float:
        """
        Simple one‐variable least squares: k = sum(r*s) / sum(r^2)
        """
        num = sum(r * s for (r, s) in self.samples)
        den = sum(r * r for (r, _) in self.samples)
        return (num / den) if den else 0.0

    def get_adv_map(self) -> Dict[float, float]:
        """After calibration pass, returns {timestamp: ADV_i}."""
        return self.adv_map


# # core/calibration.py
#
# from typing import List, Tuple
# from core.fsm      import Agent
# from core.events   import Event, BAR, TradeFill
#
# class CalibrationAgent(Agent):
#     """
#     In pass #1, listen for BARs to track ADV & mid‐price,
#     then for each TradeFill compute (rho, slippage) and stash it.
#     """
#     def __init__(self):
#         super().__init__(name="CALIB")
#         # rolling ADV, mid‐price state
#         self.last_adv:   float = 0.0
#         self.last_mid:   float = 0.0
#         # collected samples of (participation_rate, realized_slippage)
#         self.samples: List[Tuple[float, float]] = []
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, TradeFill))
#
#     def main(self, e: Event) -> None:
#         # on each BAR, update your ADV estimate & mid price
#         if isinstance(e, BAR):
#             # example: use BAR.volume as proxy for per‐bar volume,
#             # and pretend 1,000 bars ≈ 1 trading day (you can replace
#             # with any ADV calculation you like)
#             self.last_adv = sum(getattr(e, "volume", 0.0) for _ in range(1)) * 1_000
#             self.last_mid = 0.5 * (e.O + e.C)
#             return
#
#         # on each fill, record (ρ, s)
#         if isinstance(e, TradeFill):
#             adv = self.last_adv or 1.0
#             mid = self.last_mid or e.price
#             rho      = e.qty / adv
#             slippage = (e.price - mid) / mid
#             self.samples.append((rho, slippage))
#
#     def compute_k(self) -> float:
#         """
#         Simple one‐variable least squares: k = sum(rho*slip) / sum(rho^2)
#         """
#         num = sum(r*s for (r, s) in self.samples)
#         den = sum(r*r for (r, _) in self.samples)
#         return (num/den) if den else 0.0
