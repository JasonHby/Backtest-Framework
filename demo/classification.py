import json
import pandas as pd
import numpy as np


def load_entry_flips(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    entries = [d for d in data if d.get("event") == "entry_signal"]
    if not entries:
        return pd.DataFrame(columns=["timestamp", "side"])
    df = pd.DataFrame([
        {
            "timestamp": float(e["timestamp"]),
            "side": e["side"].lower(),
            **{k: v for k, v in e.items() if k not in ("timestamp", "side", "event")}
        }
        for e in entries
    ])
    return df


def nearest_match(single_df: pd.DataFrame, mixed_df: pd.DataFrame, tol: float):
    matched_idxs = []
    missing_idxs = []
    mixed_df = mixed_df.copy()
    if "side" in mixed_df.columns:
        mixed_df["side"] = mixed_df["side"].str.lower()

    for i, row in single_df.iterrows():
        side = row["side"]
        ts = row["timestamp"]
        candidates = mixed_df[mixed_df["side"] == side]
        if not candidates.empty and ((candidates["timestamp"] - ts).abs() <= tol).any():
            matched_idxs.append(i)
        else:
            missing_idxs.append(i)

    present = single_df.loc[matched_idxs].reset_index(drop=True)
    missing = single_df.loc[missing_idxs].reset_index(drop=True)
    return present, missing


def classify_missing(missing_flips: pd.DataFrame, mixed_trades: pd.DataFrame, tol: float):
    merged = []
    suppressed = []

    for _, row in missing_flips.iterrows():
        ts = row["timestamp"]
        nearby = mixed_trades[np.abs(mixed_trades["entry_ts"] - ts) <= tol]
        if not nearby.empty:
            merged.append(row)
        else:
            suppressed.append(row)

    merged_df = pd.DataFrame(merged)
    suppressed_df = pd.DataFrame(suppressed)
    return merged_df, suppressed_df


def find_nearest_trade(flip_row: pd.Series, mixed_trades: pd.DataFrame):
    ts = flip_row["timestamp"]
    diffs = np.abs(mixed_trades["entry_ts"] - ts)
    if diffs.empty or diffs.min() == np.inf:
        return None, None
    idx = diffs.idxmin()
    nearest = mixed_trades.loc[idx]
    delta = nearest["entry_ts"] - ts
    return nearest, delta


def print_merge_examples(merged_df: pd.DataFrame, mixed_trades: pd.DataFrame, label: str, n: int = 5):
    if merged_df.empty:
        print(f"\nNo {label} merged examples.")
        return
    print(f"\n{label} merged examples (showing up to {n}):")
    for i, row in merged_df.head(n).iterrows():
        nearest, delta = find_nearest_trade(row, mixed_trades)
        pdiff = row.get("prev_diff", None)
        diff = row.get("diff", None)
        print(f"{i+1}. flip ts={row['timestamp']} side={row['side']} prev_diff={pdiff} diff={diff}")
        if nearest is not None:
            print(
                f"   ↳ matched mixed trade: entry_ts={nearest['entry_ts']} exit_ts={nearest['exit_ts']} "
                f"entry_price={nearest.get('entry_price')} exit_price={nearest.get('exit_price')} delta={delta}"
            )
        else:
            print("   WARNING: classified as merged but no nearby mixed trade found.")


def main():
    TOL = 3600  # tolerance in seconds (e.g., one bar)

    long_flips = load_entry_flips("cta_flip_debug_long.json")
    short_flips = load_entry_flips("cta_flip_debug_short.json")
    mixed_flips = load_entry_flips("cta_flip_debug_mixed.json")

    mixed_trades = pd.read_csv("trade_report_mixed.csv")

    # normalize sides
    mixed_flips["side"] = mixed_flips["side"].str.lower()

    long_present, long_missing = nearest_match(long_flips, mixed_flips, TOL)
    short_present, short_missing = nearest_match(short_flips, mixed_flips, TOL)

    long_merged, long_suppressed = classify_missing(long_missing, mixed_trades, TOL)
    short_merged, short_suppressed = classify_missing(short_missing, mixed_trades, TOL)

    summary = {
        "long_only_total_flips": len(long_flips),
        "mixed_long_flips": len(mixed_flips[mixed_flips["side"] == "long"]),
        "missing_long_in_mixed": len(long_missing),
        "long_merged": len(long_merged),
        "long_suppressed": len(long_suppressed),
        "short_only_total_flips": len(short_flips),
        "mixed_short_flips": len(mixed_flips[mixed_flips["side"] == "short"]),
        "missing_short_in_mixed": len(short_missing),
        "short_merged": len(short_merged),
        "short_suppressed": len(short_suppressed),
    }

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k:30}: {v}")
    print()

    print("Long suppressed example flips:")
    if not long_suppressed.empty and "timestamp" in long_suppressed.columns:
        print(long_suppressed.sort_values("timestamp").head(5))
    else:
        print("<none>")

    print("\nShort suppressed example flips:")
    if not short_suppressed.empty and "timestamp" in short_suppressed.columns:
        print(short_suppressed.sort_values("timestamp").head(5))
    else:
        print("<none>")

    # **new**: merged example detail dumps
    print_merge_examples(long_merged, mixed_trades, "Long", n=5)
    print_merge_examples(short_merged, mixed_trades, "Short", n=5)


if __name__ == "__main__":
    main()

