from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_1H = pd.Timedelta(hours=1)
EXPECTED_4H = pd.Timedelta(hours=4)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "timestamp" if "timestamp" in df.columns else "time" if "time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "timestamp"})
    return df.sort_values("timestamp").reset_index(drop=True)


def candle_gap_report(df: pd.DataFrame, expected_delta: pd.Timedelta) -> dict:
    duplicate_count = int(df["timestamp"].duplicated().sum())
    deltas = df["timestamp"].diff().dropna()
    missing_steps = deltas[deltas != expected_delta]
    return {
        "rows": int(len(df)),
        "start": df["timestamp"].min().isoformat() if len(df) else None,
        "end": df["timestamp"].max().isoformat() if len(df) else None,
        "timezone": str(df["timestamp"].dt.tz),
        "duplicates": duplicate_count,
        "gap_count": int(len(missing_steps)),
        "largest_gap": str(missing_steps.max()) if len(missing_steps) else None,
    }


def check_directory(data_dir: Path, suffix: str, expected_delta: pd.Timedelta) -> dict:
    results = {}
    for path in sorted(data_dir.glob(f"*{suffix}.csv")):
        symbol = path.stem.replace(suffix.replace(".csv", ""), "").strip("_")
        df = load_csv(path)
        results[symbol or path.stem] = candle_gap_report(df, expected_delta)
    return results


def check_4h_consistency(data_dir: Path) -> dict:
    one_h_files = sorted(data_dir.glob("*1h*.csv"))
    four_h_files = sorted(data_dir.glob("*4h*.csv"))
    if not one_h_files or not four_h_files:
        return {"status": "not_checked", "reason": "Both 1h and 4h files are required in the same data directory."}
    return {"status": "not_implemented", "reason": "Found both timeframes, but no canonical 4h aggregation contract exists in Agent 1 code."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv-dir", type=Path, default=Path("data/ohlcv"))
    parser.add_argument("--data-1h-dir", type=Path, default=Path("data/1h"))
    parser.add_argument("--out", type=Path, default=Path("research/volatility_expansion/reports/audit_checks.json"))
    args = parser.parse_args()

    output = {
        "data_ohlcv_1h": check_directory(args.ohlcv_dir, "_1h", EXPECTED_1H) if args.ohlcv_dir.exists() else {},
        "data_1h": check_directory(args.data_1h_dir, "", EXPECTED_1H) if args.data_1h_dir.exists() else {},
        "four_hour_consistency": {
            "data_ohlcv": check_4h_consistency(args.ohlcv_dir) if args.ohlcv_dir.exists() else {"status": "missing_dir"},
            "data_1h": check_4h_consistency(args.data_1h_dir) if args.data_1h_dir.exists() else {"status": "missing_dir"},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
