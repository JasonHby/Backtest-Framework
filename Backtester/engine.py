# Backtester/engine.py
from typing           import Sequence
from core.plumbing    import run_simulation
from Backtester.loader  import load_events
from Backtester.recorder import Recorder
from core.executor    import ExecutionAgent
from core.fsm         import Agent

def run_backtest(
    data_folder:    str,
    strategies:     Sequence[Agent],
    starting_cash:  float = 0.0
) -> Recorder:

    # 1) Load all your CSV→BAR events
    events = load_events(data_folder)

    # 2) Build our cast
    exec_agent = ExecutionAgent(starting_cash=starting_cash)
    recorder   = Recorder()             # <- no args here any more

    # 3) Run the sim
    run_simulation(
        agents=[*strategies, exec_agent, recorder],
        events=events,
    )

    # 4) Return the P/L tracker
    return recorder





