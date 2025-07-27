# from __future__ import annotations
# from typing import List
# from .events import PRC, TickBar, TimeBar, BAR
# from .plumbing import emit, bar_from_ticks
# from .fsm import Agent
#
# class TickBarGenerator(Agent):
#     def __init__(self, mkt: str, N: int):
#         super().__init__(name=f"TBG_{mkt}_{N}")
#         self.MKT, self.N = mkt, N
#         self.buf: List[PRC] = []
#     def observe(self, e):
#         return isinstance(e, PRC) and e.security == self.MKT
#     def main(self, e: PRC):
#         self.buf.append(e)
#         if len(self.buf) == self.N:
#             o,h,l,c = bar_from_ticks(self.buf)
#             emit(TickBar(timestamp=self.buf[0].timestamp,
#                          security=self.MKT,
#                          value=(o,h,l,c),
#                          numticks=self.N))
#             self.buf.clear()
#
# class TimeBarGenerator(Agent):
#     def __init__(self, mkt: str, seconds: int):
#         super().__init__(name=f"TBG_TIME_{mkt}_{seconds}")
#         self.MKT, self.seconds = mkt, seconds
#         self.buf: List[PRC] = []
#         self.start = None
#     def observe(self,e):
#         return isinstance(e, PRC) and e.security == self.MKT
#     def main(self,e:PRC):
#         if self.start is None:
#             self.start = e.timestamp
#         self.buf.append(e)
#         if e.timestamp - self.start >= self.seconds:
#             o,h,l,c = bar_from_ticks(self.buf)
#             emit(TimeBar(timestamp=self.start,
#                          security=self.MKT,
#                          value=(o,h,l,c),
#                          seconds=self.seconds))
#             self.buf.clear(); self.start = None
#
# class VolumeBarGenerator(Agent):
#     def __init__(self, mkt: str, volume: float):
#         super().__init__(name=f"VBG_{mkt}_{volume}")
#         self.MKT, self.target = mkt, volume
#         self.buf: List[PRC] = []; self.acc = 0.0
#     def observe(self,e):
#         return isinstance(e, PRC) and e.security == self.MKT
#     def main(self,e:PRC):
#         self.buf.append(e); self.acc += e.volume
#         if self.acc >= self.target:
#             o,h,l,c = bar_from_ticks(self.buf)
#             emit(BAR(timestamp=self.buf[0].timestamp,
#                      security=self.MKT,
#                      value=(o,h,l,c)))
#             self.buf.clear(); self.acc = 0.0
