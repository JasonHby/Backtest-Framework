# backtester/report.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import csv, json, shutil
import math
from pathlib import Path
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
import statistics
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional


def build_trade_signals_figure(
    price_csv: str,
    trade_csv: str,
    partials_csv: Optional[str] = None,
    show_labels: bool = False
)-> go.Figure:
    """
    Interactive Plotly chart of price with trade signals.

    Layers:
      • Price line
      • Final exits: Long (▲ green entry / ▼ red exit), Short (▼ blue entry / ▲ orange exit)
      • Optional partial exits: grey ▼ for long slices, grey ▲ for short slices

    Parameters
    ----------
    price_csv : str
        CSV with columns ['timestamp', 'price'].
    trade_csv : str
        CSV with at least ['entry_ts','exit_ts','entry_price','exit_price','trade_type'].
    partials_csv : Optional[str]
        CSV with partial slices, at least
        ['exit_ts','exit_price','qty','pnl','reason','trade_type'].
    show_labels : bool
        If True, shows the partial-exit reason text near each marker.
    """
    # 1) Load data
    price_df = pd.read_csv(price_csv)
    trades_df = pd.read_csv(trade_csv)

    # 2) Convert timestamps
    price_df["dt"] = pd.to_datetime(price_df["timestamp"], unit="s", utc=True)
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_ts"], unit="s", utc=True)
    trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_ts"], unit="s", utc=True)
    trades_df["trade_type"] = trades_df["trade_type"].astype(str).str.lower()

    longs = trades_df[trades_df["trade_type"] == "long"]
    shorts = trades_df[trades_df["trade_type"] == "short"]

    # 3) Build figure
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=price_df["dt"], y=price_df["price"],
        mode="lines", name="Price", line=dict(color="lightblue")
    ))

    # Long Entry ▲ (green)
    fig.add_trace(go.Scatter(
        x=longs["entry_dt"], y=longs["entry_price"],
        mode="markers", name="Long Entry",
        marker=dict(symbol="triangle-up", size=10, color="green"),
        customdata=longs[["qty"]].values,
        hovertemplate=(
            "Long Entry<br>"
            "time=%{x|%Y-%m-%d %H:%M}<br>"
            "price=%{y:.2f}<br>"
            "qty=%{customdata[0]:.6f}"
            "<extra></extra>"
        ),
    ))

    # Long Exit ▼ (red)
    fig.add_trace(go.Scatter(
        x=longs["exit_dt"], y=longs["exit_price"],
        mode="markers", name="Long Exit",
        marker=dict(symbol="triangle-down", size=10, color="red"),
        customdata=longs[["qty", "pnl", "exit_reason", "slippage_pct", "slippage_cost"]].values,
        hovertemplate=(
            "Long Exit<br>"
            "time=%{x|%Y-%m-%d %H:%M}<br>"
            "price=%{y:.2f}<br>"
            "qty=%{customdata[0]:.6f}<br>"
            "pnl=%{customdata[1]:+.2f}<br>"
            "reason=%{customdata[2]}<br>"
            "slip=%{customdata[3]:.2f}% ($%{customdata[4]:.2f})"
            "<extra></extra>"
        ),
    ))

    # Short Entry ▼ (blue)
    fig.add_trace(go.Scatter(
        x=shorts["entry_dt"], y=shorts["entry_price"],
        mode="markers", name="Short Entry",
        marker=dict(symbol="triangle-down", size=10, color="blue"),
        customdata=shorts[["qty"]].values,
        hovertemplate=(
            "Short Entry<br>"
            "time=%{x|%Y-%m-%d %H:%M}<br>"
            "price=%{y:.2f}<br>"
            "qty=%{customdata[0]:.6f}"
            "<extra></extra>"
        ),
    ))

    # Short Exit ▲ (orange)
    fig.add_trace(go.Scatter(
        x=shorts["exit_dt"], y=shorts["exit_price"],
        mode="markers", name="Short Exit",
        marker=dict(symbol="triangle-up", size=10, color="orange"),
        customdata=shorts[["qty", "pnl", "exit_reason", "slippage_pct", "slippage_cost"]].values,
        hovertemplate=(
            "Short Exit<br>"
            "time=%{x|%Y-%m-%d %H:%M}<br>"
            "price=%{y:.2f}<br>"
            "qty=%{customdata[0]:.6f}<br>"
            "pnl=%{customdata[1]:+.2f}<br>"
            "reason=%{customdata[2]}<br>"
            "slip=%{customdata[3]:.2f}% ($%{customdata[4]:.2f})"
            "<extra></extra>"
        ),
    ))

    # 4) partial exits (split by side, grey markers)
    if partials_csv and os.path.exists(partials_csv):
        parts_df = pd.read_csv(partials_csv)
        if not parts_df.empty:
            parts_df["dt"] = pd.to_datetime(parts_df["exit_ts"], unit="s", utc=True)
            parts_df["trade_type"] = parts_df["trade_type"].astype(str).str.lower()
            parts_long = parts_df[parts_df["trade_type"] == "long"]
            parts_short = parts_df[parts_df["trade_type"] == "short"]

            # Long partials: grey triangle-down
            fig.add_trace(go.Scatter(
                x=parts_long["dt"],
                y=parts_long["exit_price"],
                mode="markers",  # <- was "markers+text" if show_labels else "markers"
                name="Partial Exit (Long)",
                marker=dict(symbol="triangle-down", size=8, color="gray"),
                hovertemplate=(
                    "Partial Exit (Long)<br>"
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "price=%{y:.2f}<br>"
                    "qty=%{customdata[0]:.6f}<br>"
                    "pnl=%{customdata[1]:+.2f}<br>"
                    "reason=%{customdata[2]}"
                ),
                customdata=parts_long[["qty", "pnl", "reason"]].values
            ))

            # Short partials: grey triangle-up
            fig.add_trace(go.Scatter(
                x=parts_short["dt"],
                y=parts_short["exit_price"],
                mode="markers",  # <- was "markers+text" if show_labels else "markers"
                name="Partial Exit (Short)",
                marker=dict(symbol="triangle-up", size=8, color="gray"),
                hovertemplate=(
                    "Partial Exit (Short)<br>"
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "price=%{y:.2f}<br>"
                    "qty=%{customdata[0]:.6f}<br>"
                    "pnl=%{customdata[1]:+.2f}<br>"
                    "reason=%{customdata[2]}"
                ),
                customdata=parts_short[["qty", "pnl", "reason"]].values
            ))

    # 5) Layout
    fig.update_layout(
        title="Price with Trade Entry/Exit Signals (incl. Partial Exits)",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        width=1600, height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Time (browser local)")
    return fig

def plot_trade_signals_interactive(
    price_csv: str,
    trade_csv: str,
    partials_csv: Optional[str] = None,
    show_labels: bool = False
):
    """
    Convenience wrapper: build figure and show it.
    """
    fig = build_trade_signals_figure(
        price_csv=price_csv,
        trade_csv=trade_csv,
        partials_csv=partials_csv,
        show_labels=show_labels,
    )
    fig.show()


def save_trade_signals(
    price_csv: str,
    trade_csv: str,
    partials_csv: Optional[str] = None,
    out_html: Optional[str] = None,
    width: int = 1600,
    height: int = 600,
    show_labels: bool = False
) -> None:
    """
    Build the signals figure and save to HTML only (no PNG).
    """
    fig = build_trade_signals_figure(
        price_csv=price_csv,
        trade_csv=trade_csv,
        partials_csv=partials_csv,
        show_labels=show_labels,
    )
    if out_html:
        fig.write_html(
            out_html,
            include_plotlyjs="cdn",
            full_html=True,
            default_width=width,
            default_height=height,
        )

def compute_metrics(trades: list[dict], starting_cash: float) -> dict:
    """
    Return a dict of key metrics for a list of closed trades.
    Units:
      - *_pct fields are in percent (e.g., 12.34 means 12.34%)
      - cagr is in decimal (e.g., 0.1234 means 12.34% per year)
    """
    if not trades:
        return {
            "n_trades": 0,
            "realized_return_pct": 0.0,
            "cagr": float("nan"),
            "max_dd_pct": 0.0,
            "dd_multiple": 0.0,            # 新增：原始回撤倍数（诊断）
            "calmar": float("nan"),
            "sharpe_ann": float("nan"),
            "profit_factor": float("nan"),
            "win_rate_pct": 0.0,
            "avg_hold_hours": 0.0,
            "total_slip": 0.0,
            "avg_slip_pct": 0.0,
        }

    rows = sorted(trades, key=lambda t: t["exit_ts"])
    n_trades = len(rows)

    # -------- 现金路径与回撤（稳健化）--------
    eq = starting_cash
    peak = starting_cash
    max_dd_raw = 0.0      # 未裁剪：可能 > peak（>100%），仅诊断用
    max_dd_capped = 0.0   # 裁剪：eq<0 时按 0 计算，用于排序指标

    profits, losses = 0.0, 0.0
    rets = []            # per-trade % returns (用于 Sharpe)
    total_hold_s = 0.0
    total_slip = 0.0
    avg_slip_pct_acc = 0.0

    for t in rows:
        pnl = float(t.get("pnl", 0.0))
        eq += pnl
        if eq > peak:
            peak = eq

        # 原始回撤（诊断）
        dd_raw = peak - eq
        if dd_raw > max_dd_raw:
            max_dd_raw = dd_raw

        # 裁剪后的回撤（用于 max_dd_pct / calmar）
        eq_floor = max(eq, 0.0)
        dd_capped = peak - eq_floor
        if dd_capped > max_dd_capped:
            max_dd_capped = dd_capped

        r_pct = float(t.get("return_pct", 0.0))
        rets.append(r_pct)

        if pnl > 0:
            profits += pnl
        else:
            losses += pnl  # negative

        total_hold_s += float(t.get("holding_time_s", 0.0))
        total_slip += float(t.get("slippage_cost", 0.0))
        avg_slip_pct_acc += float(t.get("slippage_pct", 0.0))

    final_cash = eq
    realized_return_pct = (final_cash - starting_cash) / starting_cash * 100.0

    # 时间跨度 → 年化
    first_day = datetime.utcfromtimestamp(rows[0]["entry_ts"]).date()
    last_day  = datetime.utcfromtimestamp(rows[-1]["exit_ts"]).date()
    total_days = (last_day - first_day).days or 1
    total_years = total_days / 365.0

    # -------- CAGR（稳健化）--------
    if final_cash <= 0 or total_years <= 0:
        cagr = float("nan")
    else:
        cagr = (final_cash / starting_cash) ** (1 / total_years) - 1

    # per-trade Sharpe → 年化（按交易频次）
    trades_per_year = n_trades / total_years if total_years else n_trades

    # 可选：年化无风险利率（按需设成常量或从外部传入，这里用 0.0 保持原行为）
    rf_annual = 0.0428

    # 将年化 Rf 折算为“每笔等效”的无风险收益（单位：百分比点）
    if trades_per_year > 0:
        rf_per_trade_pct = ((1.0 + rf_annual) ** (1.0 / trades_per_year) - 1.0) * 100.0
    else:
        rf_per_trade_pct = 0.0

    if len(rets) > 1:
        # —— 分子：超额收益的均值 E[Rp - Rf] —— #
        mean_excess = sum((r - rf_per_trade_pct) for r in rets) / len(rets)

        # —— 分母：资产收益 Rp 的标准差 σ(Rp) —— #
        # 与你原实现一致，默认总体标准差（ddof=0）
        mean_asset = sum(rets) / len(rets)
        var_asset = sum((x - mean_asset) ** 2 for x in rets) / len(rets)
        std_asset = math.sqrt(var_asset)

        sharpe = (mean_excess / std_asset) if std_asset else float("nan")
    else:
        sharpe = float("nan")

    # 年化（按交易频次年化）
    sharpe_ann = sharpe * math.sqrt(trades_per_year) if sharpe == sharpe else float("nan")

    # Profit Factor
    profit_factor = (profits / abs(losses)) if losses < 0 else float("inf")

    # 胜率
    win_rate_pct = 100.0 * sum(1 for t in rows if t.get("pnl", 0.0) > 0) / n_trades

    avg_hold_hours = (total_hold_s / n_trades) / 3600.0
    avg_slip_pct = avg_slip_pct_acc / n_trades if n_trades else 0.0

    # -------- 回撤与 Calmar（稳健化）--------
    # 用“裁剪后的回撤”换算百分比，范围 0~100%
    if peak > 0:
        max_dd_pct = (max_dd_capped / peak) * 100.0
        dd_multiple = max_dd_raw / peak
    else:
        max_dd_pct = 100.0
        dd_multiple = float("inf")

    if max_dd_pct > 0 and cagr == cagr:  # cagr==cagr 等价于 not math.isnan(cagr)
        calmar = cagr / (max_dd_pct / 100.0)
    else:
        calmar = float("nan")

    return {
        "n_trades": n_trades,
        "realized_return_pct": realized_return_pct,
        "cagr": cagr,
        "max_dd_pct": max_dd_pct,   # 用于排序（0~100%）
        "dd_multiple": dd_multiple, # 诊断：>1 代表未裁剪口径上已“穿透本金”
        "calmar": calmar,
        "sharpe_ann": sharpe_ann,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate_pct,
        "avg_hold_hours": avg_hold_hours,
        "total_slip": total_slip,
        "avg_slip_pct": avg_slip_pct,
    }

def print_summary(trades: List[Dict], starting_cash: float):
    """
    Console summary using the cash-path as the authoritative source.
    Requires each trade dict to include:
      'pnl','return_pct','qty','cash_after','entry_ts','exit_ts',
      'entry_price','exit_price','trade_type','holding_time_s',
      'slippage_cost','slippage_pct'
    Optional: 'exit_reason', 'exit_price_vwap'
    """
    if not trades:
        print("No trades to summarize.")
        return

    # —— 1) 排序，保证按退出时间重建现金路径 ——
    trades_sorted = sorted(trades, key=lambda t: t["exit_ts"])

    # —— 2) 调试：字段检查 + 三列成本闭合检查（就放这里） ——
    print("[DEBUG] first trade dict keys:", list(trades_sorted[0].keys()))
    sum_slip = sum(t.get("slippage_cost", 0.0) for t in trades_sorted)
    sum_qeff = sum(t.get("quantity_effect_cost", 0.0) for t in trades_sorted)
    sum_total = sum(t.get("total_effect_cost", 0.0) for t in trades_sorted)
    print(f"[DEBUG] Σslip={sum_slip:.2f}  Σqty_eff={sum_qeff:.2f}  "
          f"Σtotal={sum_total:.2f}  (check: slip+qty={sum_slip+sum_qeff:.2f})")

    # —— 3) 基础向量 ——
    returns   = [t.get("return_pct", 0.0) for t in trades_sorted]
    pnls      = [t.get("pnl", 0.0) for t in trades_sorted]
    qtys      = [abs(t.get("qty", 0.0)) for t in trades_sorted]
    cash_after= [t.get("cash_after") for t in trades_sorted]
    slips     = [t.get("slippage_cost", 0.0) for t in trades_sorted]
    slip_pcts = [t.get("slippage_pct", 0.0) for t in trades_sorted]

    total_trades = len(trades_sorted)
    wins = [r for r in returns if r > 0]
    loss = [r for r in returns if r <= 0]

    avg_ret = sum(returns) / total_trades
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(loss) / len(loss)) if loss else 0.0
    win_rate = len(wins) / total_trades * 100.0
    total_vol = sum(qtys)

    total_slip = sum(slips)
    avg_slip = total_slip / total_trades
    avg_slip_pct = sum(slip_pcts) / total_trades

    # —— 4) 现金路径 / 回撤 ——
    eq = starting_cash
    equity_path = []
    peak = starting_cash
    max_dd = 0.0
    for t in trades_sorted:
        eq += t.get("pnl", 0.0)
        equity_path.append(eq)
        if eq > peak:
            peak = eq
        max_dd = max(max_dd, peak - eq)

    final_cash = equity_path[-1] if equity_path else starting_cash
    realized_return = (final_cash - starting_cash) / starting_cash  # decimal
    max_dd_pct = (max_dd / peak * 100.0) if peak else 0.0

    # —— 5) 年化、频次、持有期 ——
    first_day = datetime.utcfromtimestamp(trades_sorted[0]["entry_ts"]).date()
    last_day = datetime.utcfromtimestamp(trades_sorted[-1]["exit_ts"]).date()
    total_days = (last_day - first_day).days or 1
    total_years = total_days / 365.0
    ann_ret = ((final_cash / starting_cash) ** (1 / total_years) - 1) if total_years else 0.0

    trades_per_day = total_trades / total_days if total_days else total_trades
    avg_hold_secs = (sum(t.get("holding_time_s", 0.0) for t in trades_sorted) / total_trades)
    avg_hold_days = avg_hold_secs / 86400.0

    # —— 6) Sharpe（按笔，同 compute_metrics 口径：分子=E[Rp−Rf]，分母=σ(Rp)）——
    # 交易频次（年化因子）
    trades_per_year = (total_trades / total_years) if total_years else total_trades

    # 年化无风险利率（3M T-Bill）
    rf_annual = 0.0428  # 4.28%

    # 把年化 Rf 折算为“每笔等效”的无风险收益（单位：百分比点）
    if trades_per_year > 0:
        rf_per_trade_pct = ((1.0 + rf_annual) ** (1.0 / trades_per_year) - 1.0) * 100.0
    else:
        rf_per_trade_pct = 0.0

    if len(returns) > 1:
        # 分子：超额收益均值 E[Rp − Rf]
        mean_excess = sum((r - rf_per_trade_pct) for r in returns) / len(returns)

        # 分母：资产收益的标准差 σ(Rp)（总体标准差，与 compute_metrics 默认一致）
        std_asset = statistics.pstdev(returns)

        sharpe = (mean_excess / std_asset) if std_asset else float("nan")
    else:
        sharpe = float("nan")

    # 年化（按交易频次）
    sharpe_ann = sharpe * (trades_per_year ** 0.5) if sharpe == sharpe else float("nan")

    # —— 7) 多空计数 ——
    long_count = sum(1 for t in trades_sorted if t.get("trade_type") == "long")
    short_count = sum(1 for t in trades_sorted if t.get("trade_type") == "short")

    # —— 8) 成本归因一览（用三列）——
    slip = sum_slip
    qty_eff = sum_qeff
    total_imp = sum_total
    print("[DIAG] Execution cost attribution")
    print(f"  Slippage cost (cash)       : ${slip:,.2f}")
    print(f"  Quantity effect (cash)     : ${qty_eff:,.2f}")
    print(f"  Total impact (A+B)         : ${total_imp:,.2f}\n")

    # —— 9) 汇总 ——
    print("========= TRADE SUMMARY =========")
    print(f"Total Trades                  : {total_trades}")
    print(f"Long Trades                   : {long_count}")
    print(f"Short Trades                  : {short_count}")
    print(f"Win Rate                      : {win_rate:.2f}%")
    print(f"Avg Return / Trade (unwtd)    : {avg_ret:.2f}%")
    print(f"Cumulative Return (realized)  : {realized_return*100:.2f}%")
    print(f"Annualized Return (cash path) : {ann_ret*100:.2f}%")
    print(f"Total Slippage Cost           : ${total_slip:,.2f}  (avg ${avg_slip:,.2f}/trade)")
    print(f"Avg Slippage % / Trade        : {avg_slip_pct:.2f}%")
    print("--------------------------------")
    print(f"Sharpe (per-trade)            : {sharpe:.2f}  |  Ann. Sharpe: {sharpe_ann:.2f}")
    print(f"Avg Win return (winners only) : {avg_win:.2f}%")
    print(f"Avg Loss return (losers only) : {avg_loss:.2f}%")
    print(f"Total Volume Traded           : {total_vol:.4f}")
    print(f"Max Drawdown (cash path)      : {max_dd:.2f} ({max_dd_pct:.2f}%)")
    print("=================================\n")

    # —— 10) 止损统计（如需保留）——
    stops = [t for t in trades_sorted if t.get("exit_reason") == "stop_loss"]
    n_stops = len(stops)
    if n_stops:
        times_h = [(t.get("holding_time_s", 0.0) / 3600.0) for t in stops]
        avg_h = sum(times_h) / n_stops
        stop_rate = n_stops / total_trades * 100.0
        print("=== Stop-Loss Stats ===")
        print(f"Stop-Loss Hits               : {n_stops}  ({stop_rate:.2f}% of trades)")
        print(f"Avg time to stop             : {avg_h:.2f} h\n")
    else:
        print("No stop-loss exits.\n")

    print("\n=== Trade Frequency & Holding Time ===")
    print(f"Trades per day (turnover): {trades_per_day:.2f}")
    print(f"Average holding time:      {avg_hold_days:.2f} days\n")

    # —— 11) 每笔明细（可选保留）——
    for i, t in enumerate(trades_sorted, 1):
        hours = t.get("holding_time_s", 0.0) / 3600.0
        entry_dt = datetime.utcfromtimestamp(t["entry_ts"])
        exit_dt = datetime.utcfromtimestamp(t["exit_ts"])
        trade_type = t.get("trade_type", "n/a")
        exit_reason = t.get("exit_reason", "")
        exit_price_vwap = t.get("exit_price_vwap", None)
        vwap_str = f" | VWAP={exit_price_vwap:.2f}" if exit_price_vwap is not None else ""
        print(
            f"[T{i:02d}] {entry_dt:%Y-%m-%d %H:%M}Z → {exit_dt:%Y-%m-%d %H:%M}Z "
            f"Type={trade_type.upper():6s} Reason={exit_reason or 'n/a':12s}  "
            f"E@{t['entry_price']:.2f} → X@{t['exit_price']:.2f}{vwap_str}  "
            f"Qty={t['qty']:.4f}  P/L={t['pnl']:+.2f}  Ret={t['return_pct']:+.2f}%  "
            f"Hold={hours:.2f}h  Inv={t['inventory_after']:.2f}  Cash={t['cash_after']:.2f}"
        )

# --- small helpers so export_artifacts can delegate ---

def write_trades_csv(trades: list[dict], outpath: Path, *, add_dt_cols: bool = True, sort_by_exit: bool = True) -> None:
    if not trades:
        return
    rows = list(trades)
    if sort_by_exit:
        rows = sorted(rows, key=lambda r: r.get("exit_ts", 0))
    fieldnames = list(rows[0].keys())
    if add_dt_cols:
        for extra in ("entry_dt", "exit_dt"):
            if extra not in fieldnames:
                fieldnames.append(extra)
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in rows:
            rec = dict(rec)
            if add_dt_cols:
                if "entry_ts" in rec:
                    rec["entry_dt"] = datetime.utcfromtimestamp(rec["entry_ts"]).strftime("%Y-%m-%d %H:%M:%S")
                if "exit_ts" in rec:
                    rec["exit_dt"]  = datetime.utcfromtimestamp(rec["exit_ts"]).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow(rec)

def write_partials_csv(partials: list[dict] | None, outpath: Path) -> None:
    if not partials:
        return
    rows = list(partials)
    fieldnames = list(rows[0].keys())
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# --- export_artifacts now delegates to the helpers above ---

def export_artifacts(
    outdir: Path,
    underlying: list[tuple[float, float]],
    trades: list[dict],
    partial_trades: list[dict] | None,
    *,
    starting_cash: float,
    cta_params: dict | None = None,
    k: float | None = None,
    tag: str = "run",
    make_zip: bool = True,
    save_signals: bool = True,
) -> Path | None:
    """
    Writes:
      • underlying_price.csv
      • trade_report_<tag>.csv
      • trade_partials.csv (if any)
      • config.json
      • signals.html (if save_signals=True)
    Optionally zips the folder and returns the ZIP path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) underlying
    under_csv = outdir / "underlying_price.csv"
    with open(under_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "price"])
        for ts, px in underlying:
            w.writerow([int(ts), float(px)])

    # 2) trades
    trade_csv = outdir / f"trade_report_{tag}.csv"
    write_trades_csv(trades, trade_csv, add_dt_cols=True, sort_by_exit=True)

    # 3) partial exits (optional)
    parts_csv_path = None
    if partial_trades:
        parts_csv_path = outdir / "trade_partials.csv"
        write_partials_csv(partial_trades, parts_csv_path)

    # 4) config.json
    first_ts = trades[0]["entry_ts"] if trades else (underlying[0][0] if underlying else None)
    last_ts  = trades[-1]["exit_ts"] if trades else (underlying[-1][0] if underlying else None)
    cfg = {
        "tag": tag,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "starting_cash": starting_cash,
        "n_trades": len(trades),
        "date_range": [first_ts, last_ts],
        "cta_params": cta_params or {},
        "shortfall_coeff_k": (k if k is not None else 0.0),
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 5) figures
    if save_signals:
        try:
            html_path = outdir / "signals.html"
            save_trade_signals(
                price_csv=str(under_csv),
                trade_csv=str(trade_csv),
                partials_csv=(str(parts_csv_path) if parts_csv_path else None),
                out_html=str(html_path),
                show_labels=False,
            )
        except Exception as e:
            print(f"[export] skipped signals html: {e}")

    # 6) zip (optional)
    if make_zip:
        zip_base = str(outdir)
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=outdir)
        return Path(zip_path)
    return None

def plot_comparison(
    underlying: List[Tuple[float, float]],
    closed_trades: List[Dict[str, float]],
    starting_cash: float,
    tz_display: str | None = None
) -> None:
    """
    Compare benchmark buy-and-hold vs strategy cash-path.
    Strategy line is a step function that updates only at trade exits.
    """
    if not underlying:
        print("⚠️  No underlying price data.")
        return

    if tz_display is None:
        tzinfo = None  # 本地
        tz_label = "Local"
    elif tz_display.upper() == "UTC":
        tzinfo = timezone.utc
        tz_label = "UTC"
    else:
        tzinfo = ZoneInfo(tz_display)
        tz_label = tz_display

    times_u, prices = zip(*underlying)
    dts = [datetime.fromtimestamp(ts, tzinfo) for ts in times_u]

    # 1) Benchmark (price relative to the first price)
    p0 = prices[0]
    bench_ret = [(p / p0 - 1) * 100.0 for p in prices]

    # 2) Strategy cash-path (step-by-exit)
    trades_sorted = sorted(closed_trades, key=lambda t: t["exit_ts"])
    step_times, step_ret = [], []
    eq = starting_cash
    si = 0
    for ts, _ in underlying:
        while si < len(trades_sorted) and trades_sorted[si]["exit_ts"] <= ts:
            eq += trades_sorted[si].get("pnl", 0.0)
            si += 1
        step_times.append(datetime.fromtimestamp(ts, tzinfo))
        step_ret.append((eq / starting_cash - 1) * 100.0)

    # 3) Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dts, bench_ret, label="Benchmark Return (Buy & Hold)", linewidth=2)
    ax.plot(step_times, step_ret, label="Strategy Return (Cash Path)", linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=tzinfo))
    fig.autofmt_xdate()
    ax.set_xlabel(f"Time ({tz_label})")
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Benchmark vs Strategy (Cash-Path) Returns — {tz_label}")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# def run_report(
#     underlying: List[Tuple[float, float]],
#     closed_trades: List[Dict[str, float]],
#     starting_cash: float
# ):
#     """
#     Print the console summary and show the comparison plot.
#     """
#     print_summary(closed_trades, starting_cash)
#     plot_comparison(underlying, closed_trades, starting_cash,tz_display="UTC")



# # backtester/report.py
#
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Dict
#
# def print_summary(
#     closed_trades: List[Dict[str, float]]
# ):
#     """
#     Console summary & per‐trade P/L breakdown.
#     """
#     if not closed_trades:
#         print("⚠️  No trades to summarize.")
#         return
#
#     wins    = [t for t in closed_trades if t['pnl'] > 0]
#     losses  = [t for t in closed_trades if t['pnl'] <= 0]
#     win_rt  = len(wins) / len(closed_trades) * 100
#     avg_ret = sum(t['return_pct'] for t in closed_trades) / len(closed_trades)
#     # Cumulative return: ∏(1 + r_i) − 1, where r_i is in %
#     cum = 1.0
#     for t in closed_trades:
#         cum *= 1 + (t['return_pct'] / 100)
#     cum_ret = (cum - 1) * 100
#
#     print("========== TRADE SUMMARY ==========")
#     print(f"Total Trades     : {len(closed_trades)}")
#     print(f"Win Rate         : {win_rt:.2f}%")
#     print(f"Avg Return/trade : {avg_ret:.4f}%")
#     print(f"Cumulative Return: {cum_ret:.4f}%")
#     print("===================================\n")
#
#     # for i, t in enumerate(closed_trades, 1):
#     #     print(
#     #         f"[T{i:02d}] E@{t['entry_price']:.2f} → X@{t['exit_price']:.2f}  "
#     #         f"Qty={t['qty']:.3f}  P/L={t['pnl']:+.2f}  Ret={t['return_pct']:+.2f}%  "
#     #         f"Inv={t['inventory_after']:.3f}"
#     #     )
#
#
# def plot_comparison(
#     underlying:    List[Tuple[float, float]],
#     closed_trades: List[Dict[str, float]]
# ) -> None:
#     """
#     Plot buy-&-hold % return (updated each BAR) vs.
#     strategy % return per trade (as scatter at exit times).
#     """
#     if not underlying:
#         print("⚠️  No underlying price data.")
#         return
#
#     # 1) Benchmark (buy-&-hold) returns in %
#     times_u, prices = zip(*underlying)
#     p0 = prices[0]
#     bench_ret = [(p / p0 - 1) * 100 for p in prices]  # % returns from 0
#
#     # 2) Strategy running cumulative return over time
#     strat_times = []
#     strat_cum = []
#     cum = 1.0
#
#     for t in closed_trades:
#         cum *= 1 + (t["return_pct"] / 100)
#         strat_times.append(t["exit_ts"])
#         strat_cum.append((cum - 1) * 100)
#
#     # 3) Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(times_u, bench_ret, label="Benchmark Return (Buy & Hold)", linewidth=2)
#     if strat_times:
#         plt.plot(
#             strat_times,
#             strat_cum,
#             linestyle = "-",
#             marker = "o",
#             label = "Strategy Cumulative Return"
#         )
#
#
#     plt.xlabel("Time")
#     plt.ylabel("Return (%)")
#     plt.title("Benchmark vs Strategy Trade Returns")
#     plt.legend(loc="best")
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#
#
# def run_report(
#     underlying:     List[Tuple[float, float]],
#     closed_trades:  List[Dict[str, float]]
# ):
#     """
#     Print trade summary then show comparison plot.
#     """
#     print_summary(closed_trades)
#     plot_comparison(underlying, closed_trades)
