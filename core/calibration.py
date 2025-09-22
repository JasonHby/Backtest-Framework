# core/calibration.py
from collections import deque
from typing import Dict, Optional
from core.fsm    import Agent
from core.events import Event, BAR, TradeFill

class CalibrationAgent(Agent):
    """
    PASS #1（校准）：
      - 维护滚动 ADV（过去 bar_per_day 根 BAR 的平均 * bar_per_day）
      - 对每个 TradeFill 抽样（与执行口径一致）：
            rho = |qty| / ADV_i                   (非负，相对流动性占比)
            s   = ((fill / base) - 1) * sign(qty) (买、卖都转为“正的成本”)
        零截距一元回归：
            k ~= sum(rho * s) / sum(rho * rho)
        并做非负与上限裁剪。
    """
    def __init__(self,
                 bar_per_day: int = 24,
                 rho_cap: float = 0.5,     # ρ 上限（去极值）
                 s_cap: float   = 0.02,    # 单笔成本上限（2%）
              ):   # k 上限（1×ADV ≤ 20% 成本）
        super().__init__(name="CALIB")
        self._vol_buf: deque[float] = deque(maxlen=bar_per_day)
        self.bar_per_day = bar_per_day

        # 最近市场状态
        self.last_adv:   float = 0.0
        self.last_open:  Optional[float] = None
        self.last_close: Optional[float] = None

        # 回归累计量
        self.sum_rho2: float = 0.0
        self.sum_rhos: float = 0.0
        self.n_used:   int   = 0

        # 给回测 PASS#2 用
        self.adv_map: Dict[float, float] = {}

        # 超参
        self.rho_cap = rho_cap
        self.s_cap   = s_cap

    def observe(self, e: Event) -> bool:
        return isinstance(e, (BAR, TradeFill))

    def main(self, e: Event) -> None:
        # 1) BAR：更新 ADV 与参考价
        if isinstance(e, BAR):
            vol = getattr(e, "V", None)
            if vol is None:
                vol = getattr(e, "volume", 0.0)
            self._vol_buf.append(float(vol))

            if self._vol_buf:
                avg_per_bar = sum(self._vol_buf) / len(self._vol_buf)
                self.last_adv = float(avg_per_bar * self._vol_buf.maxlen)
            else:
                self.last_adv = 0.0

            self.last_open  = float(e.O)
            self.last_close = float(e.C)
            self.adv_map[e.timestamp] = self.last_adv
            return

        # 2) TradeFill：抽样（与执行口径一致）
        if isinstance(e, TradeFill):
            # ADV：优先用事件自带（executor 传入），否则回退
            adv_i = getattr(e, "adv_at_fill", None)
            if adv_i is None or adv_i <= 0:
                adv_i = self.adv_map.get(e.timestamp, self.last_adv)
            if not adv_i or adv_i <= 0:
                return  # 无有效 ADV，略过

            # 以 BAR 的 mid 作为参考（与执行口径解耦，避免 k=0 时 s=0）
            if self.last_open is not None and self.last_close is not None:
                mid = 0.5 * (self.last_open + self.last_close)
            else:
                mid = float(e.price)  # 极端回退

            fill = float(e.price)
            qty = float(e.qty)

            #rho = abs(qty) / max(adv_i, 1e-12)  # ≥ 0
            rho = abs(qty) / adv_i
            s_raw = (fill / mid) - 1.0  # 可能正负
            s = s_raw if qty >= 0 else -s_raw  # 成本同向(≥0)

            # 去极值保护（保持你已有的 cap）
            if not (0.0 < rho <= self.rho_cap and 0.0 < s <= self.s_cap):
                return

            self.sum_rho2 += rho * rho
            self.sum_rhos += rho * s
            self.n_used += 1

    def compute_k(self) -> float:
        if getattr(self, "sum_rho2", 0.0) <= 1e-12:
            print("[CAL] not enough samples; k=0")
            return 0.0
        k_raw = self.sum_rhos / self.sum_rho2
        k = max(k_raw, 0.0)  # 仅下限0，**无上限**
        print(f"[CAL] k_raw={k_raw:.6g} -> k={k:.6g}")
        return k

    def get_adv_map(self) -> Dict[float, float]:
        return self.adv_map



# # core/calibration.py
# from collections import deque
# from typing       import List, Tuple, Dict
# from core.fsm     import Agent
# from core.events  import Event, BAR, TradeFill
#
# class CalibrationAgent(Agent):
#     """
#     Pass #1: build ADV_i, mid‐price, then on each TradeFill record
#     (ρ_i = qty/ADV_i,   s_i = (fill_price−mid)/mid) and stash ADV_i.
#     """
#     def __init__(self, bar_per_day: int = 24):
#         super().__init__(name="CALIB")
#         # rolling buffer of the last `bar_per_day` BAR.volume values (24 hours)
#         self._vol_buffer: deque[float] = deque(maxlen=bar_per_day)
#         self.last_adv:     float       = 0.0    # ADV_i at the current BAR
#         self.last_mid:     float       = 0.0    # mid price at the current BAR
#
#         # collected calibration samples
#         self.samples: List[Tuple[float, float]] = []
#         # mapping from fill‐timestamps to the ADV used
#         self.adv_map: Dict[float, float] = {}
#
#     def observe(self, e: Event) -> bool:
#         return isinstance(e, (BAR, TradeFill))
#
#     def main(self, e: Event) -> None:
#         # 1) On each BAR, update rolling ADV and mid
#         if isinstance(e, BAR):
#             # push this bar's volume into the rolling window
#             self._vol_buffer.append(getattr(e, "volume", 0.0))
#             if self._vol_buffer:
#                 # average volume per bar * bars_per_day = ADV_i
#                 avg_per_bar = sum(self._vol_buffer) / len(self._vol_buffer)
#                 self.last_adv = avg_per_bar * self._vol_buffer.maxlen
#             else:
#                 self.last_adv = 0.0
#
#             # mid‐price = (O + C) / 2
#             self.last_mid = 0.5 * (e.O + e.C)
#             self.adv_map[e.timestamp] = self.last_adv  # NEW: 为每根 BAR 建立 ADV 映射
#             return
#
#         # 2) On each TradeFill, record (ρ_i, s_i) and stash ADV_i
#         if isinstance(e, TradeFill):
#             adv_i = self.last_adv or 1.0
#             mid_i = self.last_mid or e.price
#
#             rho      = e.qty / adv_i
#             slippage = (e.price - mid_i) / mid_i
#
#             # store for regression
#             self.samples.append((rho, slippage))
#             # also make available to executor later
#             self.adv_map[e.timestamp] = adv_i
#
#     def compute_k(self) -> float:
#         """
#         Simple one‐variable least squares: k = sum(r*s) / sum(r^2)
#         """
#         num = sum(r * s for (r, s) in self.samples)
#         den = sum(r * r for (r, _) in self.samples)
#         return (num / den) if den else 0.0
#
#     def get_adv_map(self) -> Dict[float, float]:
#         """After calibration pass, returns {timestamp: ADV_i}."""
#         return self.adv_map



