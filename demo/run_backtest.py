# demo/run_backtest.py
import json
from pathlib import Path
from datetime import datetime,timezone

from Backtester.loader   import load_events
from core.executor       import ExecutionAgent
from core.calibration    import CalibrationAgent
from Backtester.recorder import Recorder
from core.plumbing       import run_simulation
from strategies.CTA      import CTA

from Backtester.report import (
    print_summary,
    plot_comparison,
    plot_trade_signals_interactive,
    export_artifacts,
)

def main():
    data_folder   = "REST_api_data"
    starting_cash = 10_000.0

    # 1) Load bars once
    events = load_events(data_folder)

    # 2) Unified CTA params (use the same for calibration and real backtest)
    cta_params = dict(
        symbol="BTCUSDT",
        short=50, long=336, qty=None,
        stop_atr=2.5, atr_len=50,
        allow_long=True, allow_short=True,
        take_profit_r1=0.3, take_profit_frac1=0.5,
        take_profit_r=3, breakeven_r=0.5,
        giveback_k=4, prefer_giveback=False,
    )

    # 3) PASS #1: calibration (k=0 execution)
    cta_cal  = CTA(**cta_params)
    exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
    calib    = CalibrationAgent()
    run_simulation([cta_cal, exec_cal, calib], events)

    k = calib.compute_k()
    adv_map = calib.get_adv_map()
    print(f"🔧  Calibrated shortfall_coeff k = {k:.6f}")

    # 4) PASS #2: real backtest (use calibrated k + ADV map)
    cta_real  = CTA(**cta_params)
    exec_real = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=k, adv_map=adv_map)
    rec       = Recorder()
    run_simulation([cta_real, exec_real, rec], events)

    # 5) Console summary + cash-path comparison plot
    print_summary(rec.closed_trades, starting_cash=starting_cash)
    plot_comparison(rec.underlying, rec.closed_trades, starting_cash, tz_display="UTC")

    # 6) Export a self-contained run folder (CSV/HTML/PNG/config + ZIP)
    tag    = "long"
    stamp  = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = Path(__file__).parent / "runs" / f"{tag}-{stamp}"

    zip_path = export_artifacts(
        outdir=outdir,
        underlying=rec.underlying,
        trades=rec.closed_trades,
        partial_trades = getattr(rec, "partial_trades", None),
        starting_cash=starting_cash,
        cta_params=cta_params,
        k=k,
        tag=tag,
        make_zip=True,
        save_signals=True,   # this writes signals.html
    )

    print(f"→ artifacts written to: {outdir}")
    if zip_path:
        print(f"→ zipped: {zip_path}")

    # 7) (Optional) Display the interactive chart now from the exported CSVs
    under_csv = outdir / "underlying_price.csv"
    trade_csv = outdir / f"trade_report_{tag}.csv"
    parts_csv = outdir / "trade_partials.csv"
    plot_trade_signals_interactive(
        price_csv=str(under_csv),
        trade_csv=str(trade_csv),
        partials_csv=str(parts_csv) if parts_csv.exists() else None,
        show_labels=False,  # set True if you want text labels next to partial markers
    )

if __name__ == "__main__":
    main()



# # demo/run_backtest.py
# import csv
# from pathlib import Path
# from datetime import datetime
# from Backtester.loader   import load_events
# from core.executor       import ExecutionAgent
# from core.calibration    import CalibrationAgent
# from Backtester.recorder import Recorder
# from core.plumbing       import run_simulation
# from Backtester.report import print_summary, plot_comparison, plot_trade_signals_interactive
# from strategies.CTA      import CTA
#
# def write_trades_csv(trades, outpath):
#     if not trades:
#         return
#     fieldnames = list(trades[0].keys()) + ["entry_dt", "exit_dt"]
#     with open(outpath, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for rec in trades:
#             rec = dict(rec)
#             rec["entry_dt"] = datetime.fromtimestamp(rec["entry_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#             rec["exit_dt"]  = datetime.fromtimestamp(rec["exit_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#             writer.writerow(rec)
#
# def write_partials_csv(partials, outpath):
#     if not partials:
#         return
#     fieldnames = list(partials[0].keys())
#     with open(outpath, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(partials)
#
# def main():
#     data_folder   = "REST_api_data"
#     starting_cash = 10_000.0
#
#     events = load_events(data_folder)
#
#     # unified CTA params (calibration == real backtest)
#     cta_params = dict(
#         symbol="BTCUSDT",
#         short=50, long=336, qty=None,
#         stop_atr=2.5, atr_len=50,
#         allow_long=True, allow_short=True,
#         take_profit_r1=1.0, take_profit_frac1=0.5,
#         take_profit_r=2.0, breakeven_r=0.5,
#         giveback_k=3.0, prefer_giveback=False
#     )
#
#     # PASS 1: calibration (k=0 exec)
#     cta_cal  = CTA(**cta_params)
#     exec_cal = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib    = CalibrationAgent()
#     run_simulation([cta_cal, exec_cal, calib], events)
#     k = calib.compute_k()
#     adv_map = calib.get_adv_map()
#     print(f"🔧  Calibrated shortfall_coeff k = {k:.6f}")
#
#     # PASS 2: real backtest
#     cta_real  = CTA(**cta_params)
#     exec_real = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=k, adv_map=adv_map)
#     rec       = Recorder()
#     run_simulation([cta_real, exec_real, rec], events)
#
#     # outputs
#     out_dir = Path(__file__).parent
#     trades_csv   = out_dir / "trade_report_mixed.csv"
#     partials_csv = out_dir / "trade_partials.csv"
#     write_trades_csv(rec.closed_trades, trades_csv)
#     write_partials_csv(rec.partial_exits, partials_csv)
#
#     print_summary(rec.closed_trades, starting_cash=starting_cash)
#     plot_comparison(rec.underlying, rec.closed_trades, starting_cash)
#     plot_trade_signals_interactive(
#         'underlying_price.csv',
#         'trade_report_mixed.csv',
#         partials_csv='trade_partials.csv',
#         show_labels=True
#     )
#
#     print(f"→ Wrote {len(rec.closed_trades)} trades to {trades_csv}")
#     print(f"→ Wrote {len(rec.partial_exits)} partial exits to {partials_csv}")
#
#     # optional: save underlying price for interactive plotting
#     underlying_csv = out_dir / "underlying_price.csv"
#     with open(underlying_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "price"])
#         for ts, price in rec.underlying:
#             writer.writerow([int(ts), price])
#     print(f"→ Wrote underlying price to {underlying_csv}")
#
# if __name__ == "__main__":
#     main()

# #demo/run_backtest.py
# import argparse
# import csv
# from pathlib import Path
# from datetime import datetime
# from Backtester.loader     import load_events
# from core.executor         import ExecutionAgent
# from core.calibration      import CalibrationAgent
# from Backtester.recorder   import Recorder
# from core.plumbing         import run_simulation
# from Backtester.report     import print_summary, plot_comparison
# from strategies.CTA        import CTA
# from Backtester.report import plot_trade_signals_interactive
# import json
#
# def write_trades_csv(trades, outpath):
#     if not trades:
#           return
#
#       # pick up your existing fields…
#     fieldnames = list(trades[0].keys())
#       # …and tack on two new “human‐readable” columns
#     fieldnames += ["entry_dt", "exit_dt"]
#
#     with open(outpath, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for rec in trades:
#               # convert the raw UNIX timestamps into ISO strings (or your preferred fmt)
#               rec["entry_dt"] = datetime.fromtimestamp(rec["entry_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#               rec["exit_dt" ] = datetime.fromtimestamp(rec["exit_ts"]).strftime("%Y-%m-%d %H:%M:%S")
#               writer.writerow(rec)
#
#
# def main():
#     data_folder   = "REST_api_data"
#     starting_cash = 10_000.0
#
#      # 1) load bars once
#     events = load_events(data_folder)
#
#     # ─── PASS #1: calibration ───────────────────────
#     cta_cal   = CTA(symbol="BTCUSDT", short=50, long=336, qty=None, stop_atr=2.5, atr_len=50,take_profit_r1=1,take_profit_frac1=0.5,
#                     take_profit_r=2,breakeven_r=0.5,giveback_k=3, prefer_giveback=False)
#     exec_cal  = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=0.0)
#     calib     = CalibrationAgent()
#     run_simulation(
#         agents = [cta_cal, exec_cal, calib],
#         events = events,
#      )
#     k = calib.compute_k()
#     print(f"🔧  Calibrated shortfall_coeff k = {k:.6f}")
#     # grab the per‐fill ADV lookup from the calibration agent
#     adv_map = calib.get_adv_map()
#
#      # ─── PASS #2: real backtest ─────────────────────
#     # cta_real  = CTA(symbol="BTCUSDT", short=20, long=50, qty=None, stop_atr=2.5, atr_len=24)
#     # exec_real = ExecutionAgent(starting_cash=starting_cash, shortfall_coeff=k,adv_map=adv_map, )
#     # recorder  = Recorder()
#     # run_simulation(
#     #     agents = [cta_real, exec_real, recorder],
#     #     events = events,
#     #  )
#     #
#     # # debug_path = Path(__file__).parent / "cta_flip_debug.json"
#     # # with open(debug_path, "w") as f:
#     # #     json.dump(cta_real._debug_flips, f, indent=2)
#     # # print(f"→ Wrote CTA debug flips to {debug_path}")
#     # cta_real.dump_debug(Path(__file__).parent / "cta_flip_debug_mixed.json")
#
#
#     # 2) helper to run a pass and dump its debug JSON
#     # ----------------------------------------------------------------
#     def run_and_dump(tag: str, *, allow_long: bool, allow_short: bool):
#         print(f"[run_and_dump] tag={tag} allow_long={allow_long} allow_short={allow_short}")
#         cta = CTA(symbol="BTCUSDT",
#                   short=50, long=336, qty=None,
#                   stop_atr=2.5, atr_len=50,
#                   allow_long=allow_long, allow_short=allow_short,
#                   take_profit_r1=1, take_profit_frac1=0.5,
#                   take_profit_r=2, breakeven_r=0.5, giveback_k=3, prefer_giveback=False)
#
#         exec_ = ExecutionAgent(starting_cash=starting_cash,
#                                shortfall_coeff=k,
#                                adv_map=adv_map)
#
#         rec = Recorder()
#
#         run_simulation(agents = [cta, exec_, rec],
#          events = events,)
#
#          # write the debug flip trace
#         cta.dump_debug(Path(__file__).parent / f"cta_flip_debug_{tag}.json")
#
#          # optional: save trade CSV only for the mixed run
#         if tag == "mixed":
#         #if tag == "long":
#         #if tag == "short":
#             out_csv = Path(__file__).parent / "trade_report_mixed.csv"
#             write_trades_csv(rec.closed_trades, out_csv)
#             print_summary(rec.closed_trades, starting_cash=starting_cash)
#             plot_comparison(rec.underlying, rec.closed_trades)
#             print(f"→ Wrote {len(rec.closed_trades)} trades to {out_csv}")
#             underlying_csv = Path(__file__).parent / "underlying_price.csv"
#             with open(underlying_csv, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["timestamp", "price"])
#                 for ts, price in rec.underlying:
#                     writer.writerow([int(ts), price])
#             print(f"→ Wrote underlying price to {underlying_csv}")
#             plot_trade_signals_interactive('underlying_price.csv', 'trade_report_mixed.csv')
#             print(f"→ Wrote {len(rec.closed_trades)} trades to {out_csv}")
#
#
#
#      #3)  Produce the three traces ───────────────────────
#     run_and_dump("mixed", allow_long=True, allow_short=True)
#     #run_and_dump(tag="long", allow_long=True, allow_short=False)
#     #run_and_dump("short", allow_long=False, allow_short=True)
#
# if __name__ == "__main__":
#     main()


#Run 1 pass without calibration, use engine.py
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


