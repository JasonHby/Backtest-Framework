import math
from typing import List
from core.events import BAR

def ema(prev: float, price: float, alpha: float) -> float:
    return alpha * price + (1 - alpha) * prev

def sma(prices: List[float]) -> float:
    return sum(prices) / len(prices) if prices else 0.0

def atr(bars: List[BAR]) -> float:
    trs = [(b.H - b.L) for b in bars]
    return sum(trs) / len(trs) if trs else 0.0

def dsi(bars: List[BAR]) -> float:
    # placeholder for shadow index (accumulated overlap)
    return 0.0

def vwma(prices: List[float], volumes: List[float]) -> float:
    if not prices or not volumes: return 0.0
    num = sum(p * v for p, v in zip(prices, volumes))
    den = sum(volumes)
    return num / den if den else 0.0

def true_range(bar: BAR) -> float:
    """
    True range of a single bar, defined as H − L.
    (For a more exact ‘true range’ you could include
    previous close, but for our CTA we just use H−L.)
    """
    return bar.H - bar.L