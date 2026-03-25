# Backtester/loader.py
import os
from pathlib import Path
from typing import List

from core.events import Event
from core.loader import CSVLoader
from core.plumbing import EVENT_Q


def load_events(data_folder: str) -> List[Event]:
    """
    Read all CSVs in `data_folder` and return BAR events sorted by timestamp.
    """
    print("[loader] Backtester.loader.py from", __file__)
    EVENT_Q.clear()

    project_root = Path(__file__).parent.parent
    folder_path = (project_root / data_folder).resolve()

    if not folder_path.is_dir():
        raise FileNotFoundError(f"Data folder not found: {folder_path!r}")

    csvs = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")],
        key=lambda fn: int(fn.rstrip(".csv")),
    )
    print("[loader] load_events will read these files in order:", csvs)

    loader = CSVLoader(str(folder_path), symbol="BTCUSDT")
    loader.run()

    events = list(EVENT_Q)
    EVENT_Q.clear()
    events.sort(key=lambda e: e.timestamp)
    return events
