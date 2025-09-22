# Backtester/loader.py
import os
from importlib.metadata import files
from typing   import List
from pathlib  import Path

from core.loader   import CSVLoader
from core.plumbing import EVENT_Q
from core.events   import Event

def load_events(data_folder: str) -> List[Event]:
    """
    Read all CSVs in `data_folder` (via core.loader.CSVLoader)
    and return a LIST of BAR events, sorted by timestamp.
    """
    print("▶ LOADED Backtester.loader.py from", __file__)
    # 1) clear any prior events
    EVENT_Q.clear()

    # 2) resolve `data_folder` relative to *project root*
    #    (we assume this file lives in <project>/Backtester/loader.py)
    project_root = Path(__file__).parent.parent   # …/Binance
    folder_path  = (project_root / data_folder).resolve()

    if not folder_path.is_dir():
        raise FileNotFoundError(f"Data folder not found: {folder_path!r}")
    # — sanity check — before we fire up the loader —
    csvs = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")],
        key=lambda fn: int(fn.rstrip(".csv"))
    )
    print("→ load_events will read these files in order:", csvs)

    # 3) fire up CSVLoader (give it an optional fallback symbol)
    loader = CSVLoader(str(folder_path), symbol="BTCUSDT")
    loader.run()   # emits BAR(...) into EVENT_Q


    # 4) snapshot & clear
    events = list(EVENT_Q)
    EVENT_Q.clear()

    # fix the order here:
    events.sort(key=lambda e: e.timestamp)

    return events
