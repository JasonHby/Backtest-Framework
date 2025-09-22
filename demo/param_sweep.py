# demo/param_sweep.py
from __future__ import annotations
import os
import itertools
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
import argparse
import pandas as pd
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# 项目内导入（与你的目录一致）
from Backtester.loader   import load_events
from core.executor       import ExecutionAgent
from core.calibration    import CalibrationAgent
from Backtester.recorder import Recorder
from core.plumbing       import run_simulation
from strategies.CTA      import CTA
from Backtester.report   import compute_metrics, export_artifacts
from itertools import product

# ====== 全局指针：给缓存/子进程用（只读） =======================================
_EVENTS_TRAIN: list | None = None   # --- add ---
_EVENTS_TEST:  list | None = None   # --- add ---

# ←← 在这里加：探针模式的 CLI/默认值（全局可见）
ARGS: argparse.Namespace = argparse.Namespace(
    probe_fixed=False,   # 关：正常回测
    probe_tp=2.0,        # 固定 TP (R)
    probe_sl=1.0,        # 固定 SL (R)
    probe_slip=0.10,     # 固定滑点（%）
    probe_tie="tp_first" # tp_first / sl_first
)

# # 想测几组就写几组：(tp_pct, sl_pct, slip_pct, tie_rule)
# PROBE_SETS: list[tuple[float, float, float, str]] = [
#     (2.0, 2.0, 0.10, "tp_first"),
#     (1.5, 1.5, 0.10, "tp_first"),
#     (2.5, 2.5, 0.10, "tp_first"),
#     (1.0, 1.0, 0.10, "tp_first"),
# ]

# 供 CTA 构造用的兜底默认值（探针模式下仅用于占位）
CTA_DEFAULTS: Dict[str, Any] = {
    "stop_atr": 2.75,
    "take_profit_r1": 1.0,
    "take_profit_frac1": 0.6,
    "take_profit_r": 2.25,
    "breakeven_r": 1.0,
    "giveback_k": 4.5,
    "prefer_giveback": False,
}

# 给静态导出用的占位，避免“可能未定义”的警告
synth_trades: list[dict] = []

@lru_cache(maxsize=8)               # --- add ---
def _build_adv_map_cached(split: str, atr_len: int) -> dict[int, float]:
    """
    计算并缓存 ADV 映射；只依赖 split(train|test) 与 atr_len。
    仅用 CalibrationAgent 即可，不做 k 计算。
    """
    evs = _EVENTS_TRAIN if split == "train" else _EVENTS_TEST
    if not evs:
        return {}
    calib = CalibrationAgent(bar_per_day=atr_len)
    run_simulation([calib], evs)
    return calib.get_adv_map()

# ====== 类型别名（允许列表里有 None） =========================================
GridValue = Union[
    List[int], List[float], List[bool],
    List[Optional[int]], List[Optional[float]]
]
Grid = Dict[str, GridValue]

# ====== 运行配置 ===============================================================
DATA_FOLDER   = "REST_api_data"
STARTING_CASH = 10_000.0
TRAIN_RATIO   = 0.70
MAKE_EXPORTS  = False
TAG           = "sweep"

# 可选: "small" | "regular" | "aggressive" | "theme_a" | "theme_b"
GRID_MODE = "theme_a"

PRESET_GRIDS: Dict[str, Grid] = {
    "small": {
        "short": [50],
        "long":  [300, 336],
        "atr_len": [50],
        "stop_atr": [2.5, 3.0],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.3, 0.5],
        "breakeven_r": [0.5, 1.0],
        "take_profit_r": [2.0, 2.5, None],
        "giveback_k": [3.0, 4.0],
        "prefer_giveback": [False, True],
        "allow_long": [True],
        "allow_short": [False, True],
    },

    "regular": {
        "short": [30, 50, 70],
        "long":  [250, 336, 420],
        "atr_len": [20, 50],
        "stop_atr": [2.0, 2.5],
        "take_profit_r1": [0.5, 1.0],
        "take_profit_frac1": [0.4, 0.5],
        "breakeven_r": [0.5, 0.8],
        "take_profit_r": [2.5, 3.0],
        "giveback_k": [3, 4, 5],
        "prefer_giveback": [True, False],
        "allow_long": [True],
        "allow_short": [False],
    },

    "aggressive": {
        "short": [20, 30, 50, 70],
        "long":  [220, 280, 336, 420],
        "atr_len": [20, 50, 80],
        "stop_atr": [2.0, 2.5, 3.0],
        "take_profit_r1": [0.5, 0.8, 1.0],
        "take_profit_frac1": [0.33, 0.5, 0.67],
        "breakeven_r": [0.4, 0.6, 0.8],
        "take_profit_r": [2.5, 3.0],
        "giveback_k": [2, 3, 4, 5],
        "prefer_giveback": [True, False],
        "allow_long": [True],
        "allow_short": [False],
    },

    # 主题 A/B（与你上轮最优簇一致）
    "theme_a": {
        "short": [50, 58, 65],
        "long":  [300, 336],
        "atr_len": [50],
        "stop_atr": [2.5, 2.75, 3.0],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.5, 0.6],
        "breakeven_r": [0.75, 1.0, 1.25],
        "take_profit_r": [1.75, 2.0, 2.25],
        "giveback_k": [4.0, 4.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "theme_b": {
        "short": [46, 50, 54],
        "long":  [300],
        "atr_len": [50],
        "stop_atr": [2.5, 2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.5, 0.6],
        "breakeven_r": [0.25, 0.5, 0.75],
        "take_profit_r": [1.75, 2.0, 2.25],
        "giveback_k": [3.0, 3.5, 4.0],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "theme_b_refine": {
        "short": [50],
        "long": [300],
        "atr_len": [50],
        "stop_atr": [2.75],
        "breakeven_r": [0.25, 0.50],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.55, 0.60, 0.65],
        "take_profit_r": [2.00, 2.25],
        "giveback_k": [3.0, 3.5, 4.0],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },

    "theme_b_refine_plus": {
        "short": [48, 50, 52],
        "long": [300],
        "atr_len": [50],
        "stop_atr": [2.75],
        "breakeven_r": [0.25, 0.40, 0.50],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.55, 0.60, 0.65],
        "take_profit_r": [1.75, 2.00, 2.25],
        "giveback_k": [3.0, 3.5, 4.0],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "regular_scout": {
        "short": [50, 58, 65],
        "long":  [280, 336, 420],
        "atr_len": [50],
        "stop_atr": [2.5, 2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.6, 0.7],
        "breakeven_r": [0.75, 1.0],
        "take_profit_r": [2.0, 2.25],
        "giveback_k": [4.0, 4.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "aggressive_scout": {
        "short": [30, 46, 52],
        "long": [250, 280, 336],
        "atr_len": [50],
        "stop_atr": [2.5, 2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.6, 0.65],
        "breakeven_r": [0.25, 0.5],
        "take_profit_r": [2.0, 2.25],
        "giveback_k": [3.5, 4.0],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "theme_a_lock": {
        "short": [58],
        "long":  [336],
        "atr_len": [50],
        "stop_atr": [2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.7],
        "breakeven_r": [1.0],
        "take_profit_r": [2.25],
        "giveback_k": [4.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },

    "theme_b_lock": {
        "short": [52],
        "long":  [300],
        "atr_len": [50],
        "stop_atr": [2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.65],
        "breakeven_r": [0.25],
        "take_profit_r": [2.00],
        "giveback_k": [3.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },

    "theme_c_lock": {
        "short": [30],
        "long":  [280],
        "atr_len": [50],
        "stop_atr": [2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.65],
        "breakeven_r": [0.25],
        "take_profit_r": [2.25],
        "giveback_k": [3.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "slow_trend_probe": {
        "short": [65, 70, 75],
        "long":  [380, 420, 460],
        "atr_len": [50],
        "stop_atr": [2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.6, 0.7],
        "breakeven_r": [1.0],           # 慢趋势常见配置；可选再加 1.25
        "take_profit_r": [2.0, 2.25],
        "giveback_k": [4.5, 5.0],       # 慢趋势通常容忍更大回吐
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [False],
    },
    "theme_a_short_flip": {
        "short": [58],
        "long":  [336],
        "atr_len": [50],
        "stop_atr": [2.75],
        "take_profit_r1": [1.0],
        "take_profit_frac1": [0.7],
        "breakeven_r": [1.0],
        "take_profit_r": [2.25],
        "giveback_k": [4.5],
        "prefer_giveback": [False],
        "allow_long": [True],
        "allow_short": [True],   # ← 只切这一项
    },
    "theme_a_probe": {
        "short": [50, 58, 65],
        "long":  [300, 336],
        "atr_len": [50],
        "allow_long": [True],
        "allow_short": [False]
    }
}
GRID: Grid = PRESET_GRIDS[GRID_MODE]

TOP_K            = 5
DO_REFINE        = False
LOCK_ALLOW_SHORT: Optional[bool] = None
K_SCALES         = [1.0]
EXPORT_TOP_N     = 0
SKIP_HEURISTICS  = True

# --- add: 并行开关 ---
PARALLEL_DEFAULT = False
WORKERS_DEFAULT  = max(1, (os.cpu_count() or 2) - 1)

# ====== 工具函数 ===============================================================
def product_dict(d: Grid) -> List[Dict[str, Any]]:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    return [{k: v for k, v in zip(keys, combo)} for combo in itertools.product(*vals)]

def split_events_by_ratio(events, train_ratio: float):
    n = len(events)
    k = max(1, min(n - 1, int(n * train_ratio)))
    return events[:k], events[k:]

# 邻域细化配置
REFINE_DELTAS: Dict[str, List[float]] = {
    "short":             [-8, 0, +8],
    "long":              [-40, 0, +40],
    "stop_atr":          [-0.25, 0, +0.25],
    "atr_len":           [0],
    "take_profit_r1":    [0],
    "take_profit_frac1": [-0.1, 0, +0.1],
    "take_profit_r":     [-0.25, 0, +0.25],
    "breakeven_r":       [-0.25, 0, +0.25],
    "giveback_k":        [-0.5, 0, +0.5],
    "prefer_giveback":   [0],
    "allow_long":        [0],
    "allow_short":       [0],
}

def _neighbor_vals(base, deltas, integer=False):
    vals = []
    for d in deltas:
        v = base + d if isinstance(d, (int, float)) else base
        vals.append(int(round(v)) if integer else v)
    return sorted({v for v in vals})

def build_refine_grid_from_top(best_row: dict) -> Grid:
    grid: Grid = {}
    for k, deltas in REFINE_DELTAS.items():
        base = best_row.get(k, None)
        if base is None or (isinstance(base, float) and pd.isna(base)):
            continue
        if isinstance(base, bool):
            grid[k] = [base]
        elif isinstance(base, int):
            grid[k] = _neighbor_vals(int(base), deltas, integer=True)
        else:
            grid[k] = _neighbor_vals(float(base), deltas, integer=False)
    return grid

def _build_bar_index(events_test) -> Tuple[List[float], List[Tuple[float,float,float,float]]]:
    """Return (ts_list, [(O,H,L,C), ...]) for quick scanning."""
    ts_list, ohlc = [], []
    for e in events_test:
        if getattr(e, "__class__", None).__name__ == "BAR":
            ts_list.append(e.timestamp)
            ohlc.append((e.O, e.H, e.L, e.C))
    return ts_list, ohlc

def _find_start_idx(ts_list: List[float], t0: float) -> int:
    """first index with ts > t0（下一根bar开始扫描）"""
    lo, hi = 0, len(ts_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_list[mid] <= t0:
            lo = mid + 1
        else:
            hi = mid
    return lo

def synth_trades_fixed_tp_sl(
    closed_trades_from_sim: List[Dict],
    events_test: List,
    starting_cash: float,
    tp_pct: float,
    sl_pct: float,
    slip_pct: float,
    tie_rule: str = "tp_first",  # or "tp_first"
) -> List[Dict]:
    """
    基于现有CTA生成的开仓点（entry_ts, entry_price, trade_type），
    在测试集bars上用“固定对称 TP/SL + 固定滑点”合成交易。
      - tp_pct / sl_pct: 百分比（如 2.0 表示 ±2%）
      - slip_pct: 每一侧的滑点百分比（买+、卖−；空单反向）
    规则：
      - 顺序处理，不允许重叠持仓：每次找到出口后才看下一笔 entry。
      - 同一bar同时命中TP与SL时，按 tie_rule 决定先后。
      - 若到测试集末尾仍未触发，按最后一根C价强制平仓。
    返回的 trade dict 与 compute_metrics 兼容（pnl 基于 starting_cash）。
    """
    if not closed_trades_from_sim:
        return []

    ts_list, ohlc = _build_bar_index(events_test)
    out: List[Dict] = []
    i_bar_min = 0  # 下一个entry的最早扫描位置

    # 只取 entry 信息（不依赖CTA的exit）
    entries = []
    for t in sorted(closed_trades_from_sim, key=lambda x: x["entry_ts"]):
        entries.append({
            "entry_ts":    t["entry_ts"],
            "entry_price": t["entry_price"],
            "side":        t.get("trade_type", "long")  # "long" / "short"
        })

    for ent in entries:
        entry_ts  = float(ent["entry_ts"])
        entry_px  = float(ent["entry_price"])
        is_long   = (str(ent["side"]).lower() == "long")

        # 应用固定滑点后的入场成交价
        if is_long:
            entry_fill = entry_px * (1.0 + slip_pct/100.0)
            tp_level   = entry_px * (1.0 + tp_pct/100.0)
            sl_level   = entry_px * (1.0 - sl_pct/100.0)
        else:
            entry_fill = entry_px * (1.0 - slip_pct/100.0)
            tp_level   = entry_px * (1.0 - tp_pct/100.0)
            sl_level   = entry_px * (1.0 + sl_pct/100.0)

        j = max(i_bar_min, _find_start_idx(ts_list, entry_ts))
        exit_ts   = None
        exit_fill = None
        exit_reason = "tp"  # default

        while j < len(ts_list):
            _, H, L, C = ohlc[j]
            hit_tp = (H >= tp_level) if is_long else (L <= tp_level)
            hit_sl = (L <= sl_level) if is_long else (H >= sl_level)
            if hit_tp and hit_sl:
                first = tie_rule
            elif hit_tp:
                first = "tp_first"
            elif hit_sl:
                first = "stop_first"
            else:
                j += 1
                continue

            exit_ts = ts_list[j]
            if (first == "tp_first"):
                raw_exit = tp_level
                exit_reason = "take_profit"
            else:
                raw_exit = sl_level
                exit_reason = "stop_loss"

            # 应用固定滑点后的出场成交价
            if is_long:
                exit_fill = raw_exit * (1.0 - slip_pct/100.0)
            else:
                exit_fill = raw_exit * (1.0 + slip_pct/100.0)

            i_bar_min = j + 1  # 下一笔从这里之后扫描
            break

        # 未命中则用最后一根C价强平
        if exit_ts is None:
            if not ts_list:
                continue  # 无bar
            exit_ts = ts_list[-1]
            raw_exit = ohlc[-1][3]  # C
            if is_long:
                exit_fill = raw_exit * (1.0 - slip_pct/100.0)
            else:
                exit_fill = raw_exit * (1.0 + slip_pct/100.0)
            exit_reason = "forced_close"

        # 以“每笔全仓”名义计算 pnl（与 starting_cash 对齐，便于横向比较信号质量）
        ret_pct = (exit_fill - entry_fill) / entry_fill * (100.0 if is_long else -100.0)
        pnl_cash = (ret_pct / 100.0) * starting_cash

        out.append({
            "entry_ts": entry_ts,
            "exit_ts":  exit_ts,
            "entry_price": entry_fill,
            "exit_price":  exit_fill,
            "pnl": pnl_cash,
            "return_pct": ret_pct,
            "holding_time_s": max(0.0, exit_ts - entry_ts),
            # 仅诊断口径：把两侧名义滑点和起来写进 slippage_pct；slippage_cost 用0
            "slippage_cost": 0.0,
            "slippage_pct":  2.0 * slip_pct,
            "trade_type": "long" if is_long else "short",
            "exit_reason": exit_reason,
            "qty": 1.0,  # 名义数量
        })

    return out

def run_one(params: Dict[str, Any],
            events_train: list,
            events_test: list,
            starting_cash: float,
            outdir: Optional[Path] = None,
            k_scale: float = 1.0,
            adv_map_precomputed: Optional[dict[int, float]] = None,
            probe_override: tuple[float, float, float, str] | None = None   # ← 新增
            ) -> Dict[str, Any]:
    """
    1) 用 train 做一次 calibration pass 得到 k
    2) ADV：优先使用外部预计算；否则本地计算 train-ADV + 缓存 test-ADV，再合并
    3) 用 test 做真实回测；如开启探针模式(--probe-fixed)，将出场改为固定 TP/SL + 固定滑点
    """
    global ARGS
    p = {**CTA_DEFAULTS, **params}
    # ==== PASS #1: 计算 k（探针模式下仍计算，仅用于记录） ====
    exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
    calib    = CalibrationAgent(bar_per_day=params["atr_len"])

    cta_cal = CTA(
        symbol="BTCUSDT",
        short=p["short"], long=p["long"], qty=None,
        stop_atr=p["stop_atr"], atr_len=p["atr_len"],
        allow_long=p["allow_long"], allow_short=p["allow_short"],
        take_profit_r1=p["take_profit_r1"],
        take_profit_frac1=p["take_profit_frac1"],
        take_profit_r=p["take_profit_r"],
        breakeven_r=p["breakeven_r"],
        giveback_k=p["giveback_k"],
        prefer_giveback=p["prefer_giveback"],
    )
    run_simulation([cta_cal, exec_cal, calib], events_train)
    k = max(0.0, calib.compute_k())

    # ==== ADV ====
    if adv_map_precomputed is not None:
        adv_map = adv_map_precomputed
    else:
        adv_train = _build_adv_map_cached("train", params["atr_len"])
        adv_test  = _build_adv_map_cached("test",  params["atr_len"])
        adv_map   = {**adv_train, **adv_test}

    # ==== PASS #2: 样本外 ====
    cta_real = CTA(
        symbol="BTCUSDT",
        short=p["short"], long=p["long"], qty=None,
        stop_atr=p["stop_atr"], atr_len=p["atr_len"],
        allow_long=p["allow_long"], allow_short=p["allow_short"],
        take_profit_r1=p["take_profit_r1"],
        take_profit_frac1=p["take_profit_frac1"],
        take_profit_r=p["take_profit_r"],
        breakeven_r=p["breakeven_r"],
        giveback_k=p["giveback_k"],
        prefer_giveback=p["prefer_giveback"],
    )

    # 探针模式：忽略冲击成本与 ADV，仅为了拿“入场序列”
    if ARGS.probe_fixed:
        exec_real = ExecutionAgent(
            starting_cash=starting_cash,
            shortfall_coeff=0.0,   # 关闭冲击成本
            adv_map=None,          # 不用 ADV
        )
    else:
        exec_real = ExecutionAgent(
            starting_cash=starting_cash,
            shortfall_coeff=k * k_scale,
            adv_map=adv_map,
        )

    rec = Recorder()
    run_simulation([cta_real, exec_real, rec], events_test)

    # ==== 指标 ====
    if ARGS.probe_fixed:
        # 优先使用任务级覆盖；否则退回全局 ARGS
        if probe_override is not None:
            tp, sl, sp, tie = probe_override  # (2.0, 1.0, 0.10, "tp_first")
        else:
            tp = float(ARGS.probe_tp)
            sl = float(ARGS.probe_sl)
            sp = float(ARGS.probe_slip)
            tie = str(ARGS.probe_tie)
        # 用固定 TP/SL 合成出场，覆盖 rec.closed_trades
        synth_trades = synth_trades_fixed_tp_sl(
            closed_trades_from_sim=rec.closed_trades,  # 只取入场信息
            events_test=events_test,
            starting_cash=starting_cash,
            tp_pct=tp,
            sl_pct=sl,
            slip_pct=sp,
            tie_rule=tie,
        )
        metrics = compute_metrics(synth_trades, starting_cash)

        # 记录探针元数据
        metrics.update({
            "probe_fixed": True,
            "probe_tp_pct": tp,
            "probe_sl_pct": sl,
            "probe_slip_pct": sp,
            "probe_tie": tie,
        })
    else:
        metrics = compute_metrics(rec.closed_trades, starting_cash)

    metrics.update({
        "k": k,
        "k_scale": k_scale,
        "train_bars": len(events_train),
        "test_bars": len(events_test),
        "train_start_dt": datetime.utcfromtimestamp(events_train[0].timestamp).strftime(
            "%Y-%m-%d %H:%M:%S") if events_train else "",
        "train_end_dt": datetime.utcfromtimestamp(events_train[-1].timestamp).strftime(
            "%Y-%m-%d %H:%M:%S") if events_train else "",
        "test_start_dt": datetime.utcfromtimestamp(events_test[0].timestamp).strftime(
            "%Y-%m-%d %H:%M:%S") if events_test else "",
        "test_end_dt": datetime.utcfromtimestamp(events_test[-1].timestamp).strftime(
            "%Y-%m-%d %H:%M:%S") if events_test else "",
    })


    # ==== 可选导出 ====
    if MAKE_EXPORTS and outdir is not None:
        export_artifacts(
            outdir=outdir,
            underlying=rec.underlying,
            trades=(synth_trades if ARGS.probe_fixed else rec.closed_trades),
            partial_trades=None,
            starting_cash=starting_cash,
            cta_params=p,
            k=(0.0 if ARGS.probe_fixed else (k * k_scale)),
            tag=("test_probe" if ARGS.probe_fixed else "test"),
            make_zip=False,
            save_signals=True,
        )

    return metrics

# # ====== 单次 run（校准→回测） ==================================================
# def run_one(params: Dict[str, Any],
#             events_train: list,
#             events_test: list,
#             starting_cash: float,
#             outdir: Optional[Path] = None,
#             k_scale: float = 1.0,
#             adv_map_precomputed: Optional[dict[int, float]] = None   # --- add ---
#             ) -> Dict[str, Any]:
#     """
#     1) 用 train 做一次 calibration pass 得到 k
#     2) ADV：优先使用外部预计算；否则本地计算 train-ADV + 缓存 test-ADV，再合并
#     3) 用 test 做真实回测，返回指标
#     """
#     # PASS #1 计算 k（k 依赖 train，ADV 不依赖 CTA）
#     exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib    = CalibrationAgent(bar_per_day=params["atr_len"])
#     # CTA 在校准阶段不发单，放与不放都行；放上便于将来需要查 stop/breakeven 状态
#     cta_cal  = CTA(symbol="BTCUSDT",
#                    short=params["short"], long=params["long"], qty=None,
#                    stop_atr=params["stop_atr"], atr_len=params["atr_len"],
#                    allow_long=params["allow_long"], allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     run_simulation([cta_cal, exec_cal, calib], events_train)
#     k = max(0.0, calib.compute_k())
#
#     # ADV
#     if adv_map_precomputed is not None:                 # --- add ---
#         adv_map = adv_map_precomputed
#     else:
#         adv_train = _build_adv_map_cached("train", params["atr_len"])
#         adv_test  = _build_adv_map_cached("test",  params["atr_len"])
#         adv_map   = {**adv_train, **adv_test}
#
#     # PASS #2 样本外真实回测
#     cta_real = CTA(symbol="BTCUSDT",
#                    short=params["short"], long=params["long"], qty=None,
#                    stop_atr=params["stop_atr"], atr_len=params["atr_len"],
#                    allow_long=params["allow_long"], allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_real = ExecutionAgent(starting_cash=starting_cash,
#                                shortfall_coeff=k * k_scale,
#                                adv_map=adv_map)
#     rec = Recorder()
#     run_simulation([cta_real, exec_real, rec], events_test)
#
#     metrics = compute_metrics(rec.closed_trades, starting_cash)
#     metrics.update({
#         "k": k,
#         "k_scale": k_scale,
#         "train_bars": len(events_train),
#         "test_bars": len(events_test),
#         "train_start_dt": datetime.utcfromtimestamp(events_train[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "train_end_dt":   datetime.utcfromtimestamp(events_train[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "test_start_dt":  datetime.utcfromtimestamp(events_test[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#         "test_end_dt":    datetime.utcfromtimestamp(events_test[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#     })
#
#     if MAKE_EXPORTS and outdir is not None:
#         export_artifacts(outdir=outdir,
#                          underlying=rec.underlying,
#                          trades=rec.closed_trades,
#                          partial_trades=None,
#                          starting_cash=starting_cash,
#                          cta_params=params,
#                          k=k * k_scale,
#                          tag="test",
#                          make_zip=False,
#                          save_signals=True)
#     return metrics

# ====== 进程池：初始化 & 工人函数 ==============================================
def _pool_init(train, test,args_ns):                    # --- add ---
    # 把大对象放到子进程全局，避免每个任务重复传
    global _EVENTS_TRAIN, _EVENTS_TEST, ARGS
    _EVENTS_TRAIN = train
    _EVENTS_TEST  = test
    ARGS = args_ns

def _pool_worker(task):                         # --- add ---
    params, ks, starting_cash, adv_map = task
    # 直接从子进程全局拿 events
    return run_one(params, _EVENTS_TRAIN, _EVENTS_TEST,
                   starting_cash, None, k_scale=ks,
                   adv_map_precomputed=adv_map)

# ====== 子 sweep（用于二次细化） ===============================================
def run_sweep_with_grid(
    grid_local: Grid,
    events_train: list,
    events_test: list,
    root: Path,
    tag_suffix: str = "",
    k_scales: list[float] | None = None,
    parallel: bool = True,            # ← 这里开/关并行
    workers: int | None = None,       # ← 并行进程数(None=cpu_count())
    adv_by_len: dict[int, dict] | None = None,  # ← 预计算的 ADV：{atr_len: adv_map}
) -> pd.DataFrame:
    if k_scales is None:
        k_scales = [1.0]

    combos = product_dict(grid_local)
    rows: list[dict[str, Any]] = []

    # 轻量剪枝（和粗扫一致，避免无效组合）
    def _skip(p: dict) -> bool:
        if p["long"] <= p["short"]: return True
        if abs(p["long"] - p["short"]) < 30: return True
        tp_r = p.get("take_profit_r", None)
        tp_r1 = p.get("take_profit_r1", None)
        if (tp_r is not None) and (tp_r1 is not None) and (tp_r <= tp_r1):
            return True
        return False

    # 组任务：与粗扫保持同一结构，复用 _pool_worker_params
    tasks: list[tuple] = []
    for p in combos:
        if _skip(p):
            continue
        adv_map = adv_by_len[p["atr_len"]] if adv_by_len is not None else None
        for ks in k_scales:
            tasks.append((p, ks, STARTING_CASH, adv_map))

    print(f"\n[refine] combos={len(combos)}  tasks={len(tasks)}  parallel={parallel}  workers={workers}\n")

    if parallel:
        # 复用粗扫的进程初始化：把 events_* 传给子进程，避免反复序列化
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_pool_init,
                                 initargs=(events_train, events_test,ARGS)) as ex:
            futs = [ex.submit(_pool_worker_params, t) for t in tasks]
            for fut in as_completed(futs):
                params, metrics = fut.result()
                row = dict(params); row.update(metrics); rows.append(row)
    else:
        # 串行
        for params, ks, starting_cash, adv_map in tasks:
            print(f"[refine] params={params}  k_scale={ks}")
            try:
                m = run_one(params,
                            events_train, events_test,
                            starting_cash,
                            outdir=None,
                            k_scale=ks,
                            adv_map_precomputed=adv_map)
            except Exception as e:
                m = {"error": str(e), "k_scale": ks}
            row = dict(params); row.update(m); rows.append(row)

    df_ref = pd.DataFrame(rows)
    return _finalize_df(df_ref, root, filename=f"sweep_refined_results{tag_suffix}.csv")

# ====== 汇总/排序/落盘（复用） ==================================================
def _finalize_df(df: pd.DataFrame, root: Path, filename: str) -> pd.DataFrame:
    # 基本健壮化：转数值
    for col in ("n_trades", "max_dd_pct", "realized_return_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 健康过滤
    if "n_trades" in df.columns:
        df = df[df["n_trades"] >= 10]
    if "max_dd_pct" in df.columns:
        df = df[df["max_dd_pct"].between(0, 80)]
    if "realized_return_pct" in df.columns:
        df = df[df["realized_return_pct"].between(-50, 200)]

    print(f"[finalize] rows after filter = {len(df)}")

    # 排序
    if ("calmar" in df.columns) and ("cagr" in df.columns):
        df = df.sort_values(["calmar", "cagr"], ascending=[False, False])

    # 落盘 + 预览
    out_csv = root / filename
    df.to_csv(out_csv, index=False)
    print(f"✅ results saved → {out_csv}")

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\nTop 10 preview:")
        print(df.head(10))
    return df


# ====== 主流程 =================================================================
def main(parallel: bool, workers: int):
    # 0) 载入全部 BAR（一次）
    events = load_events(DATA_FOLDER)
    events_train, events_test = split_events_by_ratio(events, TRAIN_RATIO)

    # 绑定到全局，供缓存/子进程使用
    global _EVENTS_TRAIN, _EVENTS_TEST
    _EVENTS_TRAIN = events_train
    _EVENTS_TEST  = events_test

    # 1) 组合枚举 + 预过滤
    combos = product_dict(GRID)
    combos = [
        p for p in combos
        if not (p.get("take_profit_r") is None and p.get("prefer_giveback") is False)
    ]
    if LOCK_ALLOW_SHORT is not None:
        combos = [{**p, "allow_short": bool(LOCK_ALLOW_SHORT)} for p in combos]

    # 2) 输出目录
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(__file__).parent / "runs" / f"{TAG}-{GRID_MODE}-{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    # 3) 预计算 ADV（按 atr_len 合并 train+test）
    if not ARGS.probe_fixed:
        uniq_len = sorted({c["atr_len"] for c in combos})
        adv_by_len = {L: {**_build_adv_map_cached("train", L),
                          **_build_adv_map_cached("test", L)} for L in uniq_len}
    else:
        adv_by_len = {c["atr_len"]: {} for c in combos}

    # 4) 构建任务
    tasks: list[tuple] = []
    for params in combos:
        if SKIP_HEURISTICS:
            if params["long"] <= params["short"]:
                continue
            if abs(params["long"] - params["short"]) < 30:
                continue
            tp_r = params.get("take_profit_r", None)
            tp_r1 = params.get("take_profit_r1", None)
            if (tp_r is not None) and (tp_r1 is not None) and (tp_r <= tp_r1):
                continue

        adv_map = adv_by_len[params["atr_len"]]

        for ks in K_SCALES:
            if ARGS.probe_fixed:
                tp_list = getattr(ARGS, "probe_tp_list", None) or [ARGS.probe_tp]
                sl_list = getattr(ARGS, "probe_sl_list", None) or [ARGS.probe_sl]
                slip_list = getattr(ARGS, "probe_slip_list", None) or [ARGS.probe_slip]
                tie_list = getattr(ARGS, "probe_tie_list", None) or [ARGS.probe_tie]

                if getattr(ARGS, "probe_symmetric", False):
                    # 只跑对称：SL 固定等于 TP
                    for tp in tp_list:
                        for slip in slip_list:
                            for tie in tie_list:
                                probe = (float(tp), float(tp), float(slip), str(tie))
                                tasks.append((params, ks, STARTING_CASH, adv_map, probe))
                else:
                    # 原来的全组合（可能不对称）
                    for tp in tp_list:
                        for sl in sl_list:
                            for slip in slip_list:
                                for tie in tie_list:
                                    probe = (float(tp), float(sl), float(slip), str(tie))
                                    tasks.append((params, ks, STARTING_CASH, adv_map, probe))
            else:
                # 正常模式：没有探针参数，用 None 占位（仍然 5 元组，结构统一）
                tasks.append((params, ks, STARTING_CASH, adv_map, None))

    rows: List[Dict[str, Any]] = []
    print(f"[coarse] combos={len(combos)}  tasks={len(tasks)}  parallel={parallel}  workers={workers}")

    if parallel:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_pool_init,
                                 initargs=(events_train, events_test, ARGS)) as ex:
            futures = [ex.submit(_pool_worker_params, t) for t in tasks]
            for fut in as_completed(futures):
                params, metrics = fut.result()
                row = dict(params);
                row.update(metrics);
                rows.append(row)
    else:
        for params, ks, starting_cash, adv_map, probe in tasks:  # ← 5 元组解包
            print(f"[{GRID_MODE}] params={params} k_scale={ks} probe={probe}")
            try:
                metrics = run_one(params, events_train, events_test, starting_cash,
                                  outdir=None, k_scale=ks,
                                  adv_map_precomputed=adv_map,
                                  probe_override=probe)  # ← 传给 run_one
            except Exception as e:
                metrics = {"error": str(e), "k_scale": ks}
            row = dict(params);
            row.update(metrics);
            rows.append(row)

    df = pd.DataFrame(rows)
    df = _finalize_df(df, root, "sweep_results.csv")

    # 导出前 N
    if MAKE_EXPORTS and EXPORT_TOP_N > 0 and not df.empty:
        top_rows = df.head(EXPORT_TOP_N).to_dict("records")
        for i, r in enumerate(top_rows, 1):
            params = {k: r[k] for k in GRID.keys() if k in r}
            ks = float(r.get("k_scale", 1.0))
            outdir = root / f"export_top_{i:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            _ = run_one(params, events_train, events_test, STARTING_CASH, outdir,
                        k_scale=ks, adv_map_precomputed=adv_by_len[params["atr_len"]])

    # 二次细化
    if DO_REFINE and not df.empty:
        best_rows = df.head(TOP_K).to_dict("records") if TOP_K > 1 else [df.iloc[0].to_dict()]
        for i, best in enumerate(best_rows, 1):
            grid_ref = build_refine_grid_from_top(best)
            if LOCK_ALLOW_SHORT is not None:
                grid_ref["allow_short"] = [bool(LOCK_ALLOW_SHORT)]
            _ = run_sweep_with_grid(grid_ref, events_train, events_test, root,
                                    tag_suffix=f"_v2_{i:02d}",
                                    k_scales=K_SCALES,
                                    parallel=parallel, workers=workers,
                                    adv_by_len=adv_by_len)

# --- add: 一个返回 (params, metrics) 的 worker，供主流程用 ----------------------
def _pool_worker_params(task):
    # 兼容 4/5 元组（refine 子流程仍是 4 元组）
    if len(task) == 5:
        params, ks, starting_cash, adv_map, probe = task
    else:
        params, ks, starting_cash, adv_map = task
        probe = None

    m = run_one(params, _EVENTS_TRAIN, _EVENTS_TEST,
                starting_cash, None, k_scale=ks,
                adv_map_precomputed=adv_map,
                probe_override=probe)  # ← 用 probe_override
    return params, m
# --- end add ------------------------------------------------------------------

# ====== 命令行 =================================================================
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID_MODE,
                    help="small|regular|aggressive|theme_a|theme_b")
    ap.add_argument("--k", dest="k_scales", default="1.0",
                    help="comma list, e.g. 0.9,1.0,1.1")
    ap.add_argument("--no-refine", action="store_true",
                    help="disable second-pass refinement")
    ap.add_argument("--parallel", action="store_true",
                    help="run with ProcessPoolExecutor")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT,
                    help="number of worker processes")
    ap.add_argument("--refine", action="store_true",
                    help="enable second-pass refinement")
    ap.add_argument("--topk", type=int, default=TOP_K,
                    help="number of top rows used for refinement")

    # 探针参数（支持逗号列表）
    ap.add_argument("--probe-fixed", action="store_true", default=ARGS.probe_fixed)
    ap.add_argument("--probe-tp",   default=str(ARGS.probe_tp))     # e.g. "2.0,1.5,2.5"
    ap.add_argument("--probe-sl",   default=str(ARGS.probe_sl))     # e.g. "2.0"
    ap.add_argument("--probe-slip", default=str(ARGS.probe_slip))   # e.g. "0.10,0.20"
    ap.add_argument("--probe-tie",  choices=["tp_first", "sl_first"], default=ARGS.probe_tie)
    ap.add_argument("--probe-symmetric", action="store_true",
                    help="only run symmetric TP/SL pairs (tp == sl)")

    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # ① 同步 CLI 到全局（run_one 里会读 ARGS.probe_*）
    ARGS.__dict__.update(vars(args))
    def _floats(s: str) -> list[float]:
        return [float(x) for x in str(s).split(",") if x != ""]

    ARGS.probe_tp_list = _floats(args.probe_tp)
    ARGS.probe_sl_list = _floats(args.probe_sl)
    ARGS.probe_slip_list = _floats(args.probe_slip)

    GRID_MODE = args.grid
    GRID = PRESET_GRIDS[GRID_MODE]
    K_SCALES = [float(x) for x in args.k_scales.split(",")]
    if args.no_refine:
        DO_REFINE = False
    if getattr(args, "refine", False):
        DO_REFINE = True
    TOP_K = getattr(args, "topk", TOP_K)
    main(parallel=args.parallel or PARALLEL_DEFAULT, workers=args.workers)





# # demo/param_sweep.py
# from __future__ import annotations
# import itertools
# import json
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Dict, Any, List, Tuple
#
# import pandas as pd
# # === NEW: 退出理由占比统计需要 ===
# from collections import Counter   # <-- NEW
#
# from Backtester.loader   import load_events
# from core.executor       import ExecutionAgent
# from core.calibration    import CalibrationAgent
# from Backtester.recorder import Recorder
# from core.plumbing       import run_simulation
# from strategies.CTA      import CTA
# from Backtester.report   import compute_metrics, export_artifacts
#
#
# # === NEW: helper —— 统计退出理由占比（放在 GRID 之前最安全） ===
# def exit_reason_breakdown(trades: List[dict]) -> Dict[str, float]:
#     """
#     trades: TradeReport 字典列表（rec.closed_trades）
#     返回每种退出理由的百分比（%）
#     """
#     n = max(1, len(trades))
#     c = Counter(t.get("exit_reason", "") for t in trades)
#     keys = ["giveback", "take_profit", "stop_loss", "breakeven", "take_profit1"]
#     out = {f"pct_{k}": 100.0 * c.get(k, 0) / n for k in keys}
#     other_cnt = n - sum(c.get(k, 0) for k in keys)
#     out["pct_other"] = 100.0 * other_cnt / n
#     return out
# # === NEW END ===
#
#
# # ---------- 可调参数网格：先小网格探方向 ----------
# GRID: Dict[str, list] = {
#     "short": [50],
#     "long":  [300, 336],
#     "atr_len": [50],
#     "stop_atr": [2.5, 3.0],
#     "take_profit_r1": [1.0],
#     "take_profit_frac1": [0.3, 0.5],
#     "breakeven_r": [0.5, 1.0],
#     "take_profit_r": [2.0, 2.5, None],   # ← 支持无上限 TP
#     "giveback_k": [3.0, 4.0],
#     "prefer_giveback": [False, True],    # ← 允许 True
#     "allow_long": [True],
#     "allow_short": [False, True],        # 先比较只做多 vs 多空
# }
#
# DATA_FOLDER   = "REST_api_data"
# STARTING_CASH = 10_000.0
# TRAIN_RATIO   = 0.70           # k 在样本内校准，真实回测在样本外
# MAKE_EXPORTS  = False          # 是否导出 signals.html/CSV（大量组合建议先关）
# TAG           = "sweep"
#
# # --- NEW: 控制项 ---
# TOP_K            = 5            # 粗扫后取前 K 个组合做细化
# DO_REFINE        = True         # 是否自动细化
# LOCK_ALLOW_SHORT = False        # None=不锁；True/False=强制只做多/只做空
# K_SCALES         = [1.0]        # 样本外 k 的敏感性测试（如 [0.8,1.0,1.2]）
# EXPORT_TOP_N     = 0            # >0 时，仅对排名前 N 的组合导出 artifacts
# # --- NEW END ---
#
#
# # ---------- 工具函数 ----------
# def product_dict(d: Dict[str, list]) -> List[Dict[str, Any]]:
#     keys = list(d.keys())
#     vals = [d[k] for k in keys]
#     out = []
#     for combo in itertools.product(*vals):
#         out.append({k: v for k, v in zip(keys, combo)})
#     return out
#
# def split_events_by_ratio(events, train_ratio: float) -> Tuple[list, list]:
#     """按时间顺序直接按比例切；events 已经排序。"""
#     n = len(events)
#     k = max(1, min(n-1, int(n * train_ratio)))
#     return events[:k], events[k:]
#
#
# # --- NEW: 自动细化的邻域生成与二次 sweep ---
# REFINE_DELTAS = {
#     "short":             [-8, 0, +8],
#     "long":              [-40, 0, +40],
#     "stop_atr":          [-0.25, 0, +0.25],
#     "atr_len":           [0],             # 不细化就填 [0]
#     "take_profit_r1":    [0],
#     "take_profit_frac1": [-0.1, 0, +0.1],
#     "take_profit_r":     [-0.25, 0, +0.25],
#     "breakeven_r":       [-0.25, 0, +0.25],
#     "giveback_k":        [-0.5, 0, +0.5],
#     "prefer_giveback":   [0],
#     "allow_long":        [0],
#     "allow_short":       [0],             # 若要锁定只做多，下面会覆盖
# }
#
# def _neighbor_vals(base, deltas, integer=False):
#     vals = []
#     for d in deltas:
#         v = base + d if isinstance(d, (int, float)) else base
#         vals.append(int(round(v)) if integer else v)
#     return sorted({v for v in vals})
#
# def build_refine_grid_from_top(best_row: dict) -> Dict[str, list]:
#     grid = {}
#     for k, deltas in REFINE_DELTAS.items():
#         base = best_row.get(k, None)
#         # 处理 NaN（比如 take_profit_r=None 写入 CSV 后会变 NaN）
#         if base is None or (isinstance(base, float) and pd.isna(base)):
#             continue
#         if isinstance(base, bool):
#             grid[k] = [base]
#         elif isinstance(base, int):
#             grid[k] = _neighbor_vals(int(base), deltas, integer=True)
#         else:
#             grid[k] = _neighbor_vals(float(base), deltas, integer=False)
#     return grid
#
# def run_sweep_with_grid(GRID_LOCAL: Dict[str, list],
#                         events_train: list,
#                         events_test: list,
#                         root: Path,
#                         tag_suffix: str = "",
#                         k_scales: list[float] = None) -> pd.DataFrame:
#     if k_scales is None:
#         k_scales = [1.0]
#     combos = product_dict(GRID_LOCAL)
#     rows = []
#     print(f"\n[refine] combos={len(combos)}  GRID={GRID_LOCAL}\n")
#     for idx, params in enumerate(combos, 1):
#         for ks in k_scales:
#             print(f"[refine {idx}/{len(combos)}] {params}  k_scale={ks}")
#             try:
#                 metrics = run_one(params, events_train, events_test, STARTING_CASH, None, k_scale=ks)
#             except Exception as e:
#                 print(f"  !! failed: {e}")
#                 metrics = {"error": str(e), "k_scale": ks}
#             row = dict(params); row.update(metrics); rows.append(row)
#
#     df_ref = pd.DataFrame(rows)
#     # --- 健康过滤 ---
#     df_ref["n_trades"] = pd.to_numeric(df_ref["n_trades"], errors="coerce")
#     df_ref["max_dd_pct"] = pd.to_numeric(df_ref["max_dd_pct"], errors="coerce")
#     df_ref["realized_return_pct"] = pd.to_numeric(df_ref["realized_return_pct"], errors="coerce")
#     df_ref = df_ref[df_ref["n_trades"] >= 10]
#     df_ref = df_ref[df_ref["max_dd_pct"].between(0, 80)]
#     df_ref = df_ref[df_ref["realized_return_pct"].between(-50, 200)]
#     print(f"[refine] rows after filter = {len(df_ref)}")
#     # -----------------------
#     if "calmar" in df_ref.columns:
#         df_ref = df_ref.sort_values(["calmar","cagr"], ascending=[False, False])
#
#     out_csv_ref = root / f"sweep_refined_results{tag_suffix}.csv"
#     df_ref.to_csv(out_csv_ref, index=False)
#     print(f"✅ refined sweep finished → {out_csv_ref}")
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print("\nRefined Top 10:")
#         print(df_ref.head(10))
#     return df_ref
# # --- NEW END ---
#
#
# # --- EDIT: run_one 增加 k_scale，并在函数内合并“退出理由占比”到 metrics 返回 ---
# def run_one(params: Dict[str, Any],
#             events_train: list,
#             events_test: list,
#             starting_cash: float,
#             outdir: Path | None = None,
#             k_scale: float = 1.0) -> Dict[str, Any]:
#     """
#     1) 用 train 做一次 calibration pass 得到 k + adv_map
#     2) 用 test 做真实回测，返回指标（含退出理由占比）
#     3) 可选导出（signals.html / CSV / config）
#     """
#     # --- PASS #1: calibration (k=0 执行) ---
#     cta_cal  = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib    = CalibrationAgent(bar_per_day=params["atr_len"])
#     run_simulation([cta_cal, exec_cal, calib], events_train)
#
#     k = max(0.0, calib.compute_k())
#     adv_map = calib.get_adv_map()
#
#     # --- PASS #2: real backtest ---
#     cta_real = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_real = ExecutionAgent(starting_cash=starting_cash,
#                                shortfall_coeff=k * k_scale,   # 应用 k_scale
#                                adv_map=adv_map)
#     rec = Recorder()
#     run_simulation([cta_real, exec_real, rec], events_test)
#
#     # --- 指标 ---
#     metrics = compute_metrics(rec.closed_trades, starting_cash)
#
#     # === NEW: 在 run_one 内直接把“退出理由占比”并进 metrics 返回 ===
#     reason_stats = exit_reason_breakdown(rec.closed_trades)   # <-- NEW
#     metrics.update(reason_stats)                               # <-- NEW
#
#     # 附加一些上下文
#     metrics.update({
#         "k": k,
#         "k_scale": k_scale,
#         "train_bars": len(events_train),
#         "test_bars": len(events_test),
#         "train_start_dt": datetime.utcfromtimestamp(events_train[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "train_end_dt":   datetime.utcfromtimestamp(events_train[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "test_start_dt":  datetime.utcfromtimestamp(events_test[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#         "test_end_dt":    datetime.utcfromtimestamp(events_test[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#     })
#
#     # --- 可选导出（建议先关，加快 sweep） ---
#     if MAKE_EXPORTS and outdir is not None:
#         export_artifacts(
#             outdir=outdir,
#             underlying=rec.underlying,
#             trades=rec.closed_trades,
#             partial_trades=None,
#             starting_cash=starting_cash,
#             cta_params=params,
#             k=k * k_scale,
#             tag="test",
#             make_zip=False,
#             save_signals=True,
#         )
#
#     return metrics
# # --- EDIT END ---
#
#
# def main():
#     # 0) 载入全部 BAR（一次）
#     events = load_events(DATA_FOLDER)  # 已排序
#     events_train, events_test = split_events_by_ratio(events, TRAIN_RATIO)
#
#     # 1) 组合枚举
#     combos = product_dict(GRID)
#
#     # --- NEW: 预过滤 —— 无上限 TP 搭配 prefer_giveback=False 没意义 ---
#     combos = [
#         p for p in combos
#         if not (p.get("take_profit_r") is None and p.get("prefer_giveback") is False)
#     ]
#     # --- NEW: 可选强制只做多/只做空 ---
#     if LOCK_ALLOW_SHORT is not None:
#         combos = [{**p, "allow_short": bool(LOCK_ALLOW_SHORT)} for p in combos]
#         combos = [dict(s) for s in {tuple(sorted(p.items())) for p in combos}]
#     # --- NEW END ---
#
#     # 2) 输出目录
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
#     root = Path(__file__).parent / "runs" / f"{TAG}-{stamp}"
#     root.mkdir(parents=True, exist_ok=True)
#
#     # 3) 执行（粗扫）
#     rows = []
#     for idx, params in enumerate(combos, 1):
#         for ks in K_SCALES:
#             # === NEW: 跑前守卫，减少无效计算 ===
#             if params["long"] <= params["short"]:
#                 continue
#             if abs(params["long"] - params["short"]) < 30:
#                 continue
#             if (params["take_profit_r"] is not None) and (params["take_profit_r"] <= params["take_profit_r1"]):
#                 continue
#             # === NEW END ===
#
#             print(f"[{idx}/{len(combos)}] params = {params}  k_scale={ks}")
#             # 粗扫阶段不导出（如需只导出前N，见排序后的导出块）
#             outdir = None
#             try:
#                 metrics = run_one(params, events_train, events_test, STARTING_CASH, outdir, k_scale=ks)
#             except Exception as e:
#                 print(f"  !! failed: {e}")
#                 metrics = {"error": str(e), "k_scale": ks}
#
#             row = dict(params)
#             row.update(metrics)
#             rows.append(row)
#
#     # 4) 汇总表（粗扫）
#     df = pd.DataFrame(rows)
#
#     # --- NEW: 健康过滤（你之前这段已在，这里明确标一下） ---
#     df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce")
#     df["max_dd_pct"] = pd.to_numeric(df["max_dd_pct"], errors="coerce")
#     df["realized_return_pct"] = pd.to_numeric(df["realized_return_pct"], errors="coerce")
#     df = df[df["n_trades"] >= 10]
#     df = df[df["max_dd_pct"].between(0, 80)]  # 先把 >80% 回撤的扔掉
#     df = df[df["realized_return_pct"].between(-50, 200)]
#     print(f"[coarse] rows after filter = {len(df)}")
#     # --- NEW END ---
#
#     if "calmar" in df.columns:
#         df = df.sort_values(["calmar", "cagr"], ascending=[False, False])
#
#     out_csv = root / "sweep_results.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n✅ sweep finished. Results → {out_csv}")
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print("\nTop 10 by calmar:")
#         print(df.head(10))
#
#     # （可选）导出排名前 N 的 artifacts（用粗扫排序后的前 N）
#     if MAKE_EXPORTS and EXPORT_TOP_N > 0 and not df.empty:
#         top_rows = df.head(EXPORT_TOP_N).to_dict("records")
#         for i, r in enumerate(top_rows, 1):
#             print(f"[export {i}/{len(top_rows)}] {r}")
#             params = {k: r[k] for k in GRID.keys() if k in r}
#             ks = float(r.get("k_scale", 1.0))
#             outdir = root / f"export_top_{i:02d}"
#             outdir.mkdir(parents=True, exist_ok=True)
#             _ = run_one(params, events_train, events_test, STARTING_CASH, outdir, k_scale=ks)
#
#     # 5) 自动细化（围绕冠军邻域）
#     if DO_REFINE and not df.empty:
#         best = df.iloc[0].to_dict()
#         grid_ref = build_refine_grid_from_top(best)
#         # 若要强制只做多/只做空
#         if LOCK_ALLOW_SHORT is not None:
#             grid_ref["allow_short"] = [bool(LOCK_ALLOW_SHORT)]
#         _ = run_sweep_with_grid(grid_ref, events_train, events_test, root,
#                                 tag_suffix="_v2", k_scales=K_SCALES)
#
# if __name__ == "__main__":
#     main()




# demo/param_sweep.py
# from __future__ import annotations
# import itertools
# import json
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Dict, Any, List, Tuple
#
# import pandas as pd
#
# from Backtester.loader   import load_events
# from core.executor       import ExecutionAgent
# from core.calibration    import CalibrationAgent
# from Backtester.recorder import Recorder
# from core.plumbing       import run_simulation
# from strategies.CTA      import CTA
# from Backtester.report   import compute_metrics, export_artifacts
#
# # ---------- 可调参数网格：先小网格探方向 ----------
# GRID: Dict[str, list] = {
#     "short": [50],
#     "long":  [300, 336],
#     "atr_len": [50],
#     "stop_atr": [2.5, 3.0],
#     "take_profit_r1": [1.0],
#     "take_profit_frac1": [0.3, 0.5],
#     "breakeven_r": [0.5, 1.0],
#     "take_profit_r": [2.0, 2.5, None],   # ← 支持无上限 TP
#     "giveback_k": [3.0, 4.0],
#     "prefer_giveback": [False, True],    # ← 允许 True
#     "allow_long": [True],
#     "allow_short": [False, True],        # 先比较只做多 vs 多空
# }
#
# DATA_FOLDER   = "REST_api_data"
# STARTING_CASH = 10_000.0
# TRAIN_RATIO   = 0.70           # k 在样本内校准，真实回测在样本外
# MAKE_EXPORTS  = False          # 是否导出 signals.html/CSV（大量组合建议先关）
# TAG           = "sweep"
#
# # --- NEW: 控制项 ---
# TOP_K            = 5            # 粗扫后取前 K 个组合做细化
# DO_REFINE        = True         # 是否自动细化
# LOCK_ALLOW_SHORT = False         # None=不锁；True/False=强制只做多/只做空
# K_SCALES         = [1.0]        # 样本外 k 的敏感性测试（如 [0.8,1.0,1.2]）
# EXPORT_TOP_N     = 0            # >0 时，仅对排名前 N 的组合导出 artifacts
#
# # ---------- 工具函数 ----------
# def product_dict(d: Dict[str, list]) -> List[Dict[str, Any]]:
#     keys = list(d.keys())
#     vals = [d[k] for k in keys]
#     out = []
#     for combo in itertools.product(*vals):
#         out.append({k: v for k, v in zip(keys, combo)})
#     return out
#
# def split_events_by_ratio(events, train_ratio: float) -> Tuple[list, list]:
#     """按时间顺序直接按比例切；events 已经排序。"""
#     n = len(events)
#     k = max(1, min(n-1, int(n * train_ratio)))
#     return events[:k], events[k:]
#
# # --- NEW: 自动细化的邻域生成与二次 sweep ---
# REFINE_DELTAS = {
#     "short":             [-8, 0, +8],
#     "long":              [-40, 0, +40],
#     "stop_atr":          [-0.25, 0, +0.25],
#     "atr_len":           [0],             # 不细化就填 [0]
#     "take_profit_r1":    [0],
#     "take_profit_frac1": [-0.1, 0, +0.1],
#     "take_profit_r":     [-0.25, 0, +0.25],
#     "breakeven_r":       [-0.25, 0, +0.25],
#     "giveback_k":        [-0.5, 0, +0.5],
#     "prefer_giveback":   [0],
#     "allow_long":        [0],
#     "allow_short":       [0],             # 若要锁定只做多，下面会覆盖
# }
#
# def _neighbor_vals(base, deltas, integer=False):
#     vals = []
#     for d in deltas:
#         v = base + d if isinstance(d, (int, float)) else base
#         vals.append(int(round(v)) if integer else v)
#     return sorted({v for v in vals})
#
# def build_refine_grid_from_top(best_row: dict) -> Dict[str, list]:
#     grid = {}
#     for k, deltas in REFINE_DELTAS.items():
#         base = best_row.get(k, None)
#         # 处理 NaN（比如 take_profit_r=None 写入 CSV 后会变 NaN）
#         if base is None or (isinstance(base, float) and pd.isna(base)):
#             continue
#         if isinstance(base, bool):
#             grid[k] = [base]
#         elif isinstance(base, int):
#             grid[k] = _neighbor_vals(int(base), deltas, integer=True)
#         else:
#             grid[k] = _neighbor_vals(float(base), deltas, integer=False)
#     return grid
#
# def run_sweep_with_grid(GRID_LOCAL: Dict[str, list],
#                         events_train: list,
#                         events_test: list,
#                         root: Path,
#                         tag_suffix: str = "",
#                         k_scales: list[float] = None) -> pd.DataFrame:
#     if k_scales is None:
#         k_scales = [1.0]
#     combos = product_dict(GRID_LOCAL)
#     rows = []
#     print(f"\n[refine] combos={len(combos)}  GRID={GRID_LOCAL}\n")
#     for idx, params in enumerate(combos, 1):
#         for ks in k_scales:
#             print(f"[refine {idx}/{len(combos)}] {params}  k_scale={ks}")
#             try:
#                 metrics = run_one(params, events_train, events_test, STARTING_CASH, None, k_scale=ks)
#             except Exception as e:
#                 print(f"  !! failed: {e}")
#                 metrics = {"error": str(e), "k_scale": ks}
#             row = dict(params); row.update(metrics); rows.append(row)
#
#     df_ref = pd.DataFrame(rows)
#     # --- 健康过滤 ---
#     df_ref["n_trades"] = pd.to_numeric(df_ref["n_trades"], errors="coerce")
#     df_ref["max_dd_pct"] = pd.to_numeric(df_ref["max_dd_pct"], errors="coerce")
#     df_ref["realized_return_pct"] = pd.to_numeric(df_ref["realized_return_pct"], errors="coerce")
#     df_ref = df_ref[df_ref["n_trades"] >= 10]
#     df_ref = df_ref[df_ref["max_dd_pct"].between(0, 80)]
#     df_ref = df_ref[df_ref["realized_return_pct"].between(-50, 200)]
#     print(f"[refine] rows after filter = {len(df_ref)}")
#     # -----------------------
#     if "calmar" in df_ref.columns:
#         df_ref = df_ref.sort_values(["calmar","cagr"], ascending=[False, False])
#
#     out_csv_ref = root / f"sweep_refined_results{tag_suffix}.csv"
#     df_ref.to_csv(out_csv_ref, index=False)
#     print(f"✅ refined sweep finished → {out_csv_ref}")
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print("\nRefined Top 10:")
#         print(df_ref.head(10))
#     return df_ref
#
# # --- EDIT: run_one 增加 k_scale ---
# def run_one(params: Dict[str, Any],
#             events_train: list,
#             events_test: list,
#             starting_cash: float,
#             outdir: Path | None = None,
#             k_scale: float = 1.0) -> Dict[str, Any]:
#     """
#     1) 用 train 做一次 calibration pass 得到 k + adv_map
#     2) 用 test 做真实回测，返回指标
#     3) 可选导出（signals.html / CSV / config）
#     """
#     # --- PASS #1: calibration (k=0 执行) ---
#     cta_cal  = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib    = CalibrationAgent(bar_per_day=params["atr_len"])
#     run_simulation([cta_cal, exec_cal, calib], events_train)
#
#     k = max(0.0, calib.compute_k())
#     adv_map = calib.get_adv_map()
#
#     # --- PASS #2: real backtest ---
#     cta_real = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_real = ExecutionAgent(starting_cash=starting_cash,
#                                shortfall_coeff=k * k_scale,   # 应用 k_scale
#                                adv_map=adv_map)
#     rec = Recorder()
#     run_simulation([cta_real, exec_real, rec], events_test)
#
#     # --- 指标 ---
#     metrics = compute_metrics(rec.closed_trades, starting_cash)
#     # 附加一些上下文
#     metrics.update({
#         "k": k,
#         "k_scale": k_scale,
#         "train_bars": len(events_train),
#         "test_bars": len(events_test),
#         "train_start_dt": datetime.utcfromtimestamp(events_train[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "train_end_dt":   datetime.utcfromtimestamp(events_train[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "test_start_dt":  datetime.utcfromtimestamp(events_test[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#         "test_end_dt":    datetime.utcfromtimestamp(events_test[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#     })
#
#     # --- 可选导出（建议先关，加快 sweep） ---
#     if MAKE_EXPORTS and outdir is not None:
#         export_artifacts(
#             outdir=outdir,
#             underlying=rec.underlying,
#             trades=rec.closed_trades,
#             partial_trades=None,
#             starting_cash=starting_cash,
#             cta_params=params,
#             k=k * k_scale,
#             tag="test",
#             make_zip=False,
#             save_signals=True,
#         )
#
#     return metrics
#
# def main():
#     # 0) 载入全部 BAR（一次）
#     events = load_events(DATA_FOLDER)  # 已排序
#     events_train, events_test = split_events_by_ratio(events, TRAIN_RATIO)
#
#     # 1) 组合枚举
#     combos = product_dict(GRID)
#
#     # --- 新增：过滤无效组合（无上限 TP 必须 prefer_giveback=True） ---
#     combos = [
#         p for p in combos
#         if not (p.get("take_profit_r") is None and p.get("prefer_giveback") is False)
#     ]
#
#     # （可选）锁定 allow_short（结构性决策后可打开）
#     if LOCK_ALLOW_SHORT is not None:
#         combos = [{**p, "allow_short": bool(LOCK_ALLOW_SHORT)} for p in combos]
#
#     # 2) 输出目录
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
#     root = Path(__file__).parent / "runs" / f"{TAG}-{stamp}"
#     root.mkdir(parents=True, exist_ok=True)
#
#     # 3) 执行（粗扫）
#     rows = []
#     for idx, params in enumerate(combos, 1):
#         for ks in K_SCALES:
#             print(f"[{idx}/{len(combos)}] params = {params}  k_scale={ks}")
#             # 粗扫阶段不导出（如需只导出前N，见排序后的导出块）
#             outdir = None
#             try:
#                 metrics = run_one(params, events_train, events_test, STARTING_CASH, outdir, k_scale=ks)
#             except Exception as e:
#                 print(f"  !! failed: {e}")
#                 metrics = {"error": str(e), "k_scale": ks}
#             row = dict(params); row.update(metrics); rows.append(row)
#
#     # 4) 汇总表（粗扫）
#     df = pd.DataFrame(rows)
#     # --- 健康过滤（可调） ---  ← 就加在这里
#     df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce")
#     df["max_dd_pct"] = pd.to_numeric(df["max_dd_pct"], errors="coerce")
#     df["realized_return_pct"] = pd.to_numeric(df["realized_return_pct"], errors="coerce")
#     df = df[df["n_trades"] >= 10]
#     df = df[df["max_dd_pct"].between(0, 80)]  # 先把 >80% 回撤的扔掉
#     df = df[df["realized_return_pct"].between(-50, 200)]
#     print(f"[coarse] rows after filter = {len(df)}")
#     # -------------------------------------
#     if "calmar" in df.columns:
#         df = df.sort_values(["calmar", "cagr"], ascending=[False, False])
#
#     out_csv = root / "sweep_results.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n✅ sweep finished. Results → {out_csv}")
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print("\nTop 10 by calmar:")
#         print(df.head(10))
#
#     # （可选）导出排名前 N 的 artifacts（用粗扫排序后的前 N）
#     if MAKE_EXPORTS and EXPORT_TOP_N > 0 and not df.empty:
#         top_rows = df.head(EXPORT_TOP_N).to_dict("records")
#         for i, r in enumerate(top_rows, 1):
#             print(f"[export {i}/{len(top_rows)}] {r}")
#             params = {k: r[k] for k in GRID.keys() if k in r}
#             ks = float(r.get("k_scale", 1.0))
#             outdir = root / f"export_top_{i:02d}"
#             outdir.mkdir(parents=True, exist_ok=True)
#             _ = run_one(params, events_train, events_test, STARTING_CASH, outdir, k_scale=ks)
#
#     # 5) 自动细化（围绕冠军邻域）
#     if DO_REFINE and not df.empty:
#         best = df.iloc[0].to_dict()
#         grid_ref = build_refine_grid_from_top(best)
#         # 若要强制只做多/只做空
#         if LOCK_ALLOW_SHORT is not None:
#             grid_ref["allow_short"] = [bool(LOCK_ALLOW_SHORT)]
#         _ = run_sweep_with_grid(grid_ref, events_train, events_test, root,
#                                 tag_suffix="_v2", k_scales=K_SCALES)
#
# if __name__ == "__main__":
#     main()


# # demo/param_sweep.py
# from __future__ import annotations
# import itertools
# import json
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Dict, Any, List, Tuple
#
# import pandas as pd
#
# from Backtester.loader   import load_events
# from core.executor       import ExecutionAgent
# from core.calibration    import CalibrationAgent
# from Backtester.recorder import Recorder
# from core.plumbing       import run_simulation
# from strategies.CTA      import CTA
# from Backtester.report   import compute_metrics, export_artifacts
#
# # ---------- 可调参数网格：先小网格探方向 ----------
# GRID: Dict[str, list] = {
#     "short": [50],
#     "long":  [300, 336],
#     "atr_len": [50],
#     "stop_atr": [2.5, 3.0],
#     "take_profit_r1": [1.0],
#     "take_profit_frac1": [0.3, 0.5],
#     "breakeven_r": [0.5, 1.0],
#     "take_profit_r": [2.0, 2.5],
#     "giveback_k": [3.0, 4.0],
#     "prefer_giveback": [False],
#     "allow_long": [True],
#     "allow_short": [False, True],   # 先比较只做多 vs 多空
# }
#
# DATA_FOLDER   = "REST_api_data"
# STARTING_CASH = 10_000.0
# TRAIN_RATIO   = 0.70           # k 在样本内校准，真实回测在样本外
# MAKE_EXPORTS  = False          # 每个组合是否导出 signals.html/CSV（大量组合建议先关）
# TAG           = "sweep"
#
# # ---------- 工具函数 ----------
# def product_dict(d: Dict[str, list]) -> List[Dict[str, Any]]:
#     keys = list(d.keys())
#     vals = [d[k] for k in keys]
#     out = []
#     for combo in itertools.product(*vals):
#         out.append({k: v for k, v in zip(keys, combo)})
#     return out
#
# def split_events_by_ratio(events, train_ratio: float) -> Tuple[list, list]:
#     """按时间顺序直接按比例切；events 已经排序。"""
#     n = len(events)
#     k = max(1, min(n-1, int(n * train_ratio)))
#     return events[:k], events[k:]
#
# def run_one(params: Dict[str, Any],
#             events_train: list,
#             events_test: list,
#             starting_cash: float,
#             outdir: Path | None = None) -> Dict[str, Any]:
#     """
#     1) 用 train 做一次 calibration pass 得到 k + adv_map
#     2) 用 test 做真实回测，返回指标
#     3) 可选导出（signals.html / CSV / config）
#     """
#     # --- PASS #1: calibration (k=0 执行) ---
#     cta_cal  = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib    = CalibrationAgent(bar_per_day=params["atr_len"])
#     run_simulation([cta_cal, exec_cal, calib], events_train)
#
#     k = calib.compute_k()
#     adv_map = calib.get_adv_map()
#
#     # --- PASS #2: real backtest ---
#     cta_real = CTA(symbol="BTCUSDT",
#                    short=params["short"],
#                    long=params["long"],
#                    qty=None,
#                    stop_atr=params["stop_atr"],
#                    atr_len=params["atr_len"],
#                    allow_long=params["allow_long"],
#                    allow_short=params["allow_short"],
#                    take_profit_r1=params["take_profit_r1"],
#                    take_profit_frac1=params["take_profit_frac1"],
#                    take_profit_r=params["take_profit_r"],
#                    breakeven_r=params["breakeven_r"],
#                    giveback_k=params["giveback_k"],
#                    prefer_giveback=params["prefer_giveback"])
#     exec_real = ExecutionAgent(starting_cash=starting_cash,
#                                shortfall_coeff=k,
#                                adv_map=adv_map)
#     rec = Recorder()
#     run_simulation([cta_real, exec_real, rec], events_test)
#
#     # --- 指标 ---
#     metrics = compute_metrics(rec.closed_trades, starting_cash)
#     # 附加一些上下文
#     metrics.update({
#         "k": k,
#         "train_bars": len(events_train),
#         "test_bars": len(events_test),
#         "train_start_dt": datetime.utcfromtimestamp(events_train[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "train_end_dt":   datetime.utcfromtimestamp(events_train[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_train else "",
#         "test_start_dt":  datetime.utcfromtimestamp(events_test[0].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#         "test_end_dt":    datetime.utcfromtimestamp(events_test[-1].timestamp).strftime("%Y-%m-%d %H:%M:%S") if events_test else "",
#     })
#
#     # --- 可选导出（建议先关，加快 sweep） ---
#     if MAKE_EXPORTS and outdir is not None:
#         export_artifacts(
#             outdir=outdir,
#             underlying=rec.underlying,
#             trades=rec.closed_trades,
#             partial_trades=None,
#             starting_cash=starting_cash,
#             cta_params=params,
#             k=k,
#             tag="test",
#             make_zip=False,
#             save_signals=True,
#         )
#
#     return metrics
#
# def main():
#     # 0) 载入全部 BAR（一次）
#     events = load_events(DATA_FOLDER)  # 已排序
#     events_train, events_test = split_events_by_ratio(events, TRAIN_RATIO)
#
#     # 1) 组合枚举
#     combos = product_dict(GRID)
#
#     # 2) 输出目录
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
#     root = Path(__file__).parent / "runs" / f"{TAG}-{stamp}"
#     root.mkdir(parents=True, exist_ok=True)
#
#     # 3) 执行
#     rows = []
#     for idx, params in enumerate(combos, 1):
#         print(f"[{idx}/{len(combos)}] params = {params}")
#         outdir = (root / f"i_{idx:03d}") if MAKE_EXPORTS else None
#         if outdir:
#             outdir.mkdir(parents=True, exist_ok=True)
#
#         try:
#             metrics = run_one(params, events_train, events_test, STARTING_CASH, outdir)
#         except Exception as e:
#             print(f"  !! failed: {e}")
#             metrics = {"error": str(e)}
#
#         row = dict(params)
#         row.update(metrics)
#         rows.append(row)
#
#     # 4) 汇总表
#     df = pd.DataFrame(rows)
#
#     # 排序：优先看 Calmar，其次看 CAGR 与回撤
#     if "calmar" in df.columns:
#         df = df.sort_values(["calmar", "cagr"], ascending=[False, False])
#
#     out_csv = root / "sweep_results.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n✅ sweep finished. Results → {out_csv}")
#
#     # 5) 顺手输出前 10 行
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print("\nTop 10 by calmar:")
#         print(df.head(10))
#
# if __name__ == "__main__":
#     main()
