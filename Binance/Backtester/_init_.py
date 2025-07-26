# backtester/__init__.py
"""
High-level backtest harness & reporting.
Public API:

  – run_backtest   : kick off a full simulation
  – Recorder       : P/L & equity tracker (if you want to import it directly)
  – print_summary  : console P/L summary
  – plot_comparison: return vs. buy-&-hold plot
"""
from .engine        import run_backtest
from .recorder      import Recorder
from .report        import print_summary, plot_comparison