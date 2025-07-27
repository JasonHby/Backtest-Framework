#demo/run_backtest.py
import csv
from pathlib import Path
from datetime import datetime
from Backtester.loader     import load_events
from core.executor         import ExecutionAgent
from core.calibration      import CalibrationAgent
from Backtester.recorder   import Recorder
from core.plumbing         import run_simulation
from Backtester.report     import print_summary, plot_comparison
from strategies.CTA        import CTA

def write_trades_csv(trades, outpath):
    if not trades:
          return

      # pick up your existing fields…
    fieldnames = list(trades[0].keys())
      # …and tack on two new “human‐readable” columns
    fieldnames += ["entry_dt", "exit_dt"]

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in trades:
              # convert the raw UNIX timestamps into ISO strings (or your preferred fmt)
              rec["entry_dt"] = datetime.fromtimestamp(rec["entry_ts"]).strftime("%Y-%m-%d %H:%M:%S")
              rec["exit_dt" ] = datetime.fromtimestamp(rec["exit_ts"]).strftime("%Y-%m-%d %H:%M:%S")
              writer.writerow(rec)


def main():
    data_folder   = "REST_api_data"
    starting_cash = 10_000.0

     # 1) load bars once
    events = load_events(data_folder)

    # ─── PASS #1: calibration ───────────────────────
    cta_cal   = CTA(symbol="BTCUSDT", short=20, long=50, qty=None, stop_atr=2.5, atr_len=24)
    exec_cal  = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
    calib     = CalibrationAgent()
    run_simulation(
        agents = [cta_cal, exec_cal, calib],
        events = events,
     )
    k = calib.compute_k()
    print(f"🔧  Calibrated shortfall_coeff k = {k:.6f}")
    # grab the per‐fill ADV lookup from the calibration agent
    adv_map = calib.get_adv_map()

     # ─── PASS #2: real backtest ─────────────────────
    cta_real  = CTA(symbol="BTCUSDT", short=20, long=50, qty=None, stop_atr=2.5, atr_len=24)
    exec_real = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=k,adv_map=adv_map, )
    recorder  = Recorder()
    run_simulation(
        agents = [cta_real, exec_real, recorder],
        events = events,
     )

     # summary & plot
    print_summary(recorder.closed_trades)
    plot_comparison(recorder.underlying, recorder.closed_trades)
    # 2) Write out CSV into your demo folder
    out_folder = Path(__file__).parent
    out_csv    = out_folder / "trade_report.csv"
    write_trades_csv(recorder.closed_trades, out_csv)
    print(f"→ Wrote {len(recorder.closed_trades)} trades to {out_csv}")

if __name__ == "__main__":
    main()


#Run 1 pass without calibration
#  #demo/run_backtest.py
# import csv
# from pathlib import Path
# from datetime import datetime
#
# from Backtester.engine import run_backtest
# from Backtester.report import print_summary, plot_comparison
# from strategies.CTA    import CTA
#
# def write_trades_csv(trades, outpath):
#     if not trades:
#         return
#
#     # pick up your existing fields…
#     fieldnames = list(trades[0].keys())
#     # …and tack on two new “human‐readable” columns
#     fieldnames += ["entry_dt", "exit_dt"]
#
#     with open(outpath, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for rec in trades:
#             # convert the raw UNIX timestamps into ISO strings (or your preferred fmt)
#             rec["entry_dt"] = datetime.fromtimestamp(rec["entry_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#             rec["exit_dt" ] = datetime.fromtimestamp(rec["exit_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#             writer.writerow(rec)
#
# def main():
#     data_folder   = "REST_api_data"
#     starting_cash = 10_000.0
#
#     cta = CTA(
#        symbol   = "BTCUSDT",
#        short    = 10,
#        long     = 30,
#        qty      = None,
#        stop_atr = 2.0,
#        atr_len  = 14,
#      )
#
#     recorder = run_backtest(
#          data_folder   = data_folder,
#          strategies    = [cta],
#          starting_cash = starting_cash,
#      )
#     #print("All closed trades:", recorder.closed_trades)
#
#      # 3) print console summary (only needs closed_trades)
#     print_summary(recorder.closed_trades)
#
#      # 4) plot benchmark vs. strategy (underlying + closed_trades)
#     plot_comparison(recorder.underlying, recorder.closed_trades)
#
# #     # 2) Write out CSV into your demo folder
#     out_folder = Path(__file__).parent
#     out_csv    = out_folder / "trade_report.csv"
#     write_trades_csv(recorder.closed_trades, out_csv)
#     print(f"→ Wrote {len(recorder.closed_trades)} trades to {out_csv}")
#
# if __name__ == "__main__":
#      main()


