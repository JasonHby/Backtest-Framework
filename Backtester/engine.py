# Backtester/engine.py
from typing           import Sequence
from core.plumbing    import run_simulation
from Backtester.loader  import load_events
from Backtester.recorder import Recorder
from core.executor    import ExecutionAgent
from core.calibration import CalibrationAgent
from core.fsm         import Agent

def run_backtest(
    data_folder:    str,
    strategies:     Sequence[Agent],
    starting_cash:  float = 0.0
) -> Recorder:

    events = load_events(data_folder)

      # ─── PASS #1 (calibration) ────────────────────────────────────────────
    calib_exec = ExecutionAgent(starting_cash=starting_cash,shortfall_coeff = 0.0)  # no slippage yet
    calib_agent = CalibrationAgent()
    run_simulation(
        agents = [*strategies, calib_exec, calib_agent],
        events = events,)

    k = calib_agent.compute_k()
    print(f"🔧  Calibrated shortfall_coeff k = {k:.6f}")

    # # ─── RESET YOUR STRATEGIES ───────────────────────────────────────────
    #
    # for strat in strategies:
    #     if hasattr(strat, "reset"):
    #       strat.reset()

      # ─── PASS #2 (real backtest with slippage) ───────────────────────────
    real_exec = ExecutionAgent(starting_cash=starting_cash,shortfall_coeff = k)
    recorder = Recorder()
    run_simulation(
            agents = [*strategies, real_exec, recorder],
            events = events,
        )
    return recorder






# # Backtester/engine.py
# from typing           import Sequence
# from core.plumbing    import run_simulation
# from Backtester.loader  import load_events
# from Backtester.recorder import Recorder
# from core.executor    import ExecutionAgent
# from core.fsm         import Agent
#
# def run_backtest(
#     data_folder:    str,
#     strategies:     Sequence[Agent],
#     starting_cash:  float = 0.0
# ) -> Recorder:
#
#     # 1) Load all your CSV→BAR events
#     events = load_events(data_folder)
#
#     # 2) Build our cast
#     exec_agent = ExecutionAgent(starting_cash=starting_cash)
#     recorder   = Recorder()             # <- no args here any more
#
#     # 3) Run the sim
#     run_simulation(
#         agents=[*strategies, exec_agent, recorder],
#         events=events,
#     )
#
#     # 4) Return the P/L tracker
#     return recorder





# # backtester/engine.py
#
# from typing          import Sequence
#
# from core.plumbing   import run_simulation
# from .loader         import load_events     # <-- relative import
# from .recorder       import Recorder        # <-- relative import
# from core.executor   import ExecutionAgent
# from core.fsm        import Agent
#
# def run_backtest(
#     data_folder:  str,
#     strategies:   Sequence[Agent],
#     starting_cash: float = 0.0,
# ) -> Recorder:
#     """
#     1) Load all your CSV k-line files into a list of BAR events
#     2) Instantiate and wire together:
#          - your strategy agent(s)
#          - an ExecutionAgent to turn COMM→TradeFill+InvUpdate
#          - a Recorder to track cash, inventory & equity
#     3) Run the core event loop until no events remain
#     4) Return the Recorder (so caller can print/plot results)
#     """
#     # 1. load + return all BAR events (in timestamp order)
#     events = load_events(data_folder)
#     print(f"-- Loaded {len(events)} BAR events from {data_folder!r}")
#
#     # 2. build our simulation cast
#     exec_agent = ExecutionAgent(starting_cash=starting_cash)
#     recorder = Recorder()
#
#     # 3. drive the FSM loop until the queue is empty
#     run_simulation(
#         agents=[*strategies, exec_agent, recorder],
#         events=events,
#     )
#
#     # 4. hand back the P/L tracker
#     return recorder
