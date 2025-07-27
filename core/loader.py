import os, csv
from typing import Any
from core.fsm      import Agent
from core.plumbing import emit
from core.events   import BAR

class CSVLoader(Agent):
    """
    Reads all .csv files from `folder` (in ascending filename order).
    Within each file:
      - sorts rows by `open_time`
      - emits a BAR event for each row
      - attaches all remaining CSV columns as attributes on the BAR
    """
    def __init__(self, folder: str, symbol: str = ""):
        super().__init__(name="CSVLoader")
        self.folder = folder
        self.symbol = symbol  # e.g. "BTCUSDT"

    def run(self) -> None:
        # 1) list, filter & sort CSV filenames by their numeric basename
        files = [
            fname for fname in os.listdir(self.folder)
            if fname.lower().endswith(".csv")
        ]
        files.sort(key=lambda fn: int(os.path.splitext(fn)[0]))

        # 2) process each file in ascending time‐chunk order
        for fname in files:
            path = os.path.join(self.folder, fname)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                # 2a) sort rows by open_time (ms)
                rows = sorted(reader, key=lambda r: int(r["open_time"]))

                # 3) emit one BAR per row
                for row in rows:
                    # ms → s
                    ts = int(row["open_time"]) / 1_000.0

                    # OHLC
                    o = float(row["open"])
                    h = float(row["high"])
                    l = float(row["low"])
                    c = float(row["close"])

                    # build the BAR event
                    bar = BAR(
                        timestamp=ts,
                        security=self.symbol,
                        O=o, H=h, L=l, C=c
                    )

                    # 4) attach extra CSV fields
                    bar.volume          = float(row.get("volume",       0.0))
                    bar.close_time      = int(row.get("close_time",   0)) / 1_000.0
                    bar.quote_vol       = float(row.get("quote_vol",    0.0))
                    bar.trades          = int(row.get("trades",       0))
                    bar.taker_base_vol  = float(row.get("taker_base_vol", 0.0))
                    bar.taker_quote_vol = float(row.get("taker_quote_vol",0.0))

                    # 5) push into the global EVENT_Q
                    emit(bar)
