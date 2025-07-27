# core/__init__.py
"""
Core backtest engine: event‐driven FSM, data adapters, indicators & executor.
Public API re-exports:

  – Events:         Event, MarketUpdate, PRC, BOOK, BAR, TickBar, TimeBar, COMM, TradeFill, InvUpdate
  – FSM:            Agent, Transition
  – Plumbing:       emit, run_simulation, EVENT_Q
  – Generators:     TickBarGenerator, TimeBarGenerator, VolumeBarGenerator
  – Loader:         CSVLoader
  – Indicators:     ema, sma, vwma, atr, dsi, true_range
  – Executor:       ExecutionAgent
"""
from core.events       import Event, MarketUpdate, PRC, BOOK, BAR, TickBar, TimeBar, COMM, TradeFill, InvUpdate
from core.fsm          import Agent, Transition
from core.plumbing     import emit, run_simulation, EVENT_Q
from core.loader       import CSVLoader
from core.indicators   import ema, sma, vwma, atr, dsi, true_range
from core.executor     import ExecutionAgent