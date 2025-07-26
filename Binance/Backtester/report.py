# backtester/report.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Tuple, Dict

def print_summary(trades: List[Dict]):
    """
    trades: list of dicts returned by your Recorder,
            each with keys including:
            'pnl', 'return_pct', 'qty', 'cash_after'
    """
    if not trades:
        print("No trades to summarize.")
        return

    total_trades = len(trades)
    returns    = [t['return_pct'] for t in trades]
    pnls       = [t['pnl']        for t in trades]
    qtys       = [abs(t['qty'])   for t in trades]
    cash_after = [t['cash_after'] for t in trades]  # equity after each exit

    # winners vs losers
    wins  = [r for r in returns if r > 0]
    loss  = [r for r in returns if r <= 0]

    win_rate = len(wins) / total_trades * 100
    avg_ret  = sum(returns) / total_trades
    cum_ret  = (cash_after[-1] / cash_after[0] - 1) * 100  # assuming cash_after[0] == starting_cash

    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss= (sum(loss)/ len(loss)) if loss else 0.0
    total_vol = sum(qtys)

    # compute max drawdown on equity curve
    # equity curve is cash_after over time (inventory goes to zero after each trade)
    peak = cash_after[0]
    max_dd = 0.0
    for eq in cash_after:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / peak * 100 if peak != 0 else 0.0

    print("========= TRADE SUMMARY =========")
    print(f"Total Trades            : {total_trades}")
    print(f"Win Rate                : {win_rate:.2f}%")
    print(f"Avg Return / Trade      : {avg_ret:.2f}%")
    print(f"Cumulative Return       : {cum_ret:.2f}%")
    print("---------------------------------")
    print(f"Avg Win return (winners only)   : {avg_win:.2f}%")
    print(f"Avg Loss return (losers only)   : {avg_loss:.2f}%")
    print(f"Total Volume Traded      : {total_vol:.4f}")
    print(f"Max Drawdown             : {max_dd:.2f} ({max_dd_pct:.2f}%)")
    print("=================================")

    for i, t in enumerate(trades, 1):
        print(
            f"[T{i:02d}] E@{t['entry_price']:.2f} → X@{t['exit_price']:.2f}  "
            f"Qty={t['qty']:.3f}  P/L={t['pnl']:+.2f}  Ret={t['return_pct']:+.2f}%  "
            f"Inv={t['inventory_after']:.2f}  Cash={t['cash_after']:.2f}"
        )


def plot_comparison(
    underlying:    List[Tuple[float, float]],
    closed_trades: List[Dict[str, float]]
) -> None:
    """
    Plot:
      • Benchmark buy‑&‑hold % return (updated every BAR).
      • Strategy cumulative % return (flat between trade exits).
    """
    if not underlying:
        print("⚠️  No underlying price data.")
        return

    # 1) Benchmark returns (every bar)
    times_u, prices = zip(*underlying)
    # convert epoch seconds → datetime for nicer axis
    dts = [datetime.utcfromtimestamp(ts) for ts in times_u]
    p0 = prices[0]
    bench_ret = [(p / p0 - 1) * 100 for p in prices]

    # 2) Build a full‐bar‐by‐bar strategy cumulative curve
    full_strat_times = []
    full_strat_ret   = []
    cum = 1.0
    si  = 0  # index into closed_trades

    for ts, _ in underlying:
        # incorporate any trades that closed at or before this bar
        while si < len(closed_trades) and closed_trades[si]["exit_ts"] <= ts:
            cum *= 1 + (closed_trades[si]["return_pct"] / 100)
            si += 1
        full_strat_times.append(datetime.utcfromtimestamp(ts))
        full_strat_ret.append((cum - 1) * 100)

    # 3) Plot both
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dts, bench_ret,
            label="Benchmark Return (Buy & Hold)",
            linewidth=2)

    ax.plot(full_strat_times, full_strat_ret,
            linestyle='-',
            marker='o',
            markersize=4,
            label="Strategy Cumulative Return")

    # 4) format x‑axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()

    ax.set_xlabel("Time")
    ax.set_ylabel("Return (%)")
    ax.set_title("Benchmark vs Strategy Cumulative Returns")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def run_report(
    underlying:     List[Tuple[float, float]],
    closed_trades:  List[Dict[str, float]]
):
    """
    Print console summary then show comparison plot.
    """
    print_summary(closed_trades)
    plot_comparison(underlying, closed_trades)





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
