from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]
TIMEFRAMES = ["15m", "1h"]
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
ATR_PERIOD = 14
VOLUME_LOOKBACK = 50
FUNDING_Z_LOOKBACK = 90
FORWARD_HORIZONS = [1, 3, 6, 12]


@dataclass(frozen=True)
class EventSpec:
    event_type: str
    direction: str
    param_name: str
    param_value: float | str
    condition: pd.Series


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, format="mixed")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="raise")
    return df[["funding_time", "funding_rate"]].dropna().drop_duplicates("funding_time").sort_values("funding_time")


def add_features(ohlcv: pd.DataFrame, funding: pd.DataFrame | None) -> pd.DataFrame:
    out = ohlcv.copy()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = true_range
    out["atr_14_prior"] = true_range.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().shift(1)
    out["atr_pct_prior"] = out["atr_14_prior"] / prev_close.replace(0, np.nan)
    out["bar_return"] = out["close"].pct_change()
    out["return_atr"] = (out["close"] - prev_close) / out["atr_14_prior"].replace(0, np.nan)
    out["range_atr"] = (out["high"] - out["low"]) / out["atr_14_prior"].replace(0, np.nan)
    vol_mean = out["volume"].rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().shift(1)
    vol_std = out["volume"].rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).std(ddof=0).shift(1)
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    atr_mean = out["atr_14_prior"].rolling(50, min_periods=50).mean().shift(1)
    out["atr_jump"] = out["atr_14_prior"] / atr_mean.replace(0, np.nan)

    if funding is not None and not funding.empty:
        f = funding.copy()
        mean = f["funding_rate"].rolling(FUNDING_Z_LOOKBACK, min_periods=FUNDING_Z_LOOKBACK).mean().shift(1)
        std = f["funding_rate"].rolling(FUNDING_Z_LOOKBACK, min_periods=FUNDING_Z_LOOKBACK).std(ddof=0).shift(1)
        f["funding_z"] = (f["funding_rate"] - mean) / std.replace(0, np.nan)
        f["funding_sign"] = np.sign(f["funding_rate"])
        f["funding_sign_prior"] = f["funding_sign"].shift(1)
        f["funding_flip_up"] = (f["funding_sign_prior"] < 0) & (f["funding_sign"] > 0)
        f["funding_flip_down"] = (f["funding_sign_prior"] > 0) & (f["funding_sign"] < 0)
        out = pd.merge_asof(out.sort_values("timestamp"), f.sort_values("funding_time"), left_on="timestamp", right_on="funding_time", direction="backward")
    else:
        out["funding_rate"] = np.nan
        out["funding_z"] = np.nan
        out["funding_flip_up"] = False
        out["funding_flip_down"] = False

    for horizon in FORWARD_HORIZONS:
        out[f"fwd_return_{horizon}"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return out


def event_specs(df: pd.DataFrame) -> list[EventSpec]:
    specs: list[EventSpec] = []
    for threshold in [1.0, 1.5, 2.0]:
        specs.append(EventSpec("large_return_atr", "up", "return_atr_abs", threshold, df["return_atr"] >= threshold))
        specs.append(EventSpec("large_return_atr", "down", "return_atr_abs", threshold, df["return_atr"] <= -threshold))
    for threshold in [0.005, 0.01, 0.02]:
        specs.append(EventSpec("large_return_pct", "up", "return_pct_abs", threshold, df["bar_return"] >= threshold))
        specs.append(EventSpec("large_return_pct", "down", "return_pct_abs", threshold, df["bar_return"] <= -threshold))
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        specs.append(EventSpec("volume_spike", "all", "volume_z", threshold, df["volume_z"] >= threshold))
    for threshold in [1.5, 2.0, 2.5]:
        specs.append(EventSpec("funding_extreme", "positive", "funding_z_abs", threshold, df["funding_z"] >= threshold))
        specs.append(EventSpec("funding_extreme", "negative", "funding_z_abs", threshold, df["funding_z"] <= -threshold))
    specs.append(EventSpec("funding_flip", "up", "flip", "neg_to_pos", df["funding_flip_up"].fillna(False).astype(bool)))
    specs.append(EventSpec("funding_flip", "down", "flip", "pos_to_neg", df["funding_flip_down"].fillna(False).astype(bool)))
    for threshold in [1.2, 1.5, 2.0]:
        specs.append(EventSpec("volatility_expansion", "all", "atr_jump", threshold, df["atr_jump"] >= threshold))
    for ret_threshold in [1.0, 1.5]:
        for vol_threshold in [1.5, 2.0]:
            specs.append(
                EventSpec(
                    "combo_volume_return",
                    "up",
                    "ret_atr_and_vol_z",
                    f"{ret_threshold:g}_{vol_threshold:g}",
                    (df["return_atr"] >= ret_threshold) & (df["volume_z"] >= vol_threshold),
                )
            )
            specs.append(
                EventSpec(
                    "combo_volume_return",
                    "down",
                    "ret_atr_and_vol_z",
                    f"{ret_threshold:g}_{vol_threshold:g}",
                    (df["return_atr"] <= -ret_threshold) & (df["volume_z"] >= vol_threshold),
                )
            )
    return specs


def distribution_stats(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    if clean.empty:
        return {
            "occurrences": 0,
            "avg_forward_return": np.nan,
            "median_forward_return": np.nan,
            "win_rate": np.nan,
            "std_forward_return": np.nan,
            "q05": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "avg_positive_return": np.nan,
            "avg_negative_return": np.nan,
            "payoff_asymmetry": np.nan,
        }
    positives = clean[clean > 0]
    negatives = clean[clean < 0]
    avg_pos = positives.mean() if not positives.empty else np.nan
    avg_neg = negatives.mean() if not negatives.empty else np.nan
    return {
        "occurrences": int(clean.shape[0]),
        "avg_forward_return": float(clean.mean()),
        "median_forward_return": float(clean.median()),
        "win_rate": float((clean > 0).mean()),
        "std_forward_return": float(clean.std(ddof=0)),
        "q05": float(clean.quantile(0.05)),
        "q10": float(clean.quantile(0.10)),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "q90": float(clean.quantile(0.90)),
        "q95": float(clean.quantile(0.95)),
        "avg_positive_return": float(avg_pos) if pd.notna(avg_pos) else np.nan,
        "avg_negative_return": float(avg_neg) if pd.notna(avg_neg) else np.nan,
        "payoff_asymmetry": float(abs(avg_pos / avg_neg)) if pd.notna(avg_pos) and pd.notna(avg_neg) and avg_neg != 0 else np.nan,
    }


def scan_events(features: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forward_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    event_summary_frames: list[pd.DataFrame] = []

    for (symbol, timeframe), df in features.items():
        for spec in event_specs(df):
            event_mask = spec.condition.fillna(False).astype(bool)
            event_df = df.loc[event_mask].copy()
            if event_df.empty:
                continue
            event_summary_frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "event_type": spec.event_type,
                        "direction": spec.direction,
                        "param_name": spec.param_name,
                        "param_value": spec.param_value,
                        "event_time": event_df["timestamp"],
                    }
                )
            )
            for horizon in FORWARD_HORIZONS:
                values = event_df[f"fwd_return_{horizon}"]
                stats = distribution_stats(values)
                base = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "event_type": spec.event_type,
                    "direction": spec.direction,
                    "param_name": spec.param_name,
                    "param_value": spec.param_value,
                    "horizon": horizon,
                }
                forward_rows.append({**base, **stats})
                dist_rows.append({**base, **stats})

    event_forward = pd.DataFrame(forward_rows)
    event_distribution = pd.DataFrame(dist_rows)
    all_events = pd.concat(event_summary_frames, ignore_index=True) if event_summary_frames else pd.DataFrame()
    if event_forward.empty:
        return pd.DataFrame(), event_forward, event_distribution

    group_cols = ["timeframe", "event_type", "direction", "param_name", "param_value", "horizon"]
    weighted = []
    for key, group in event_forward.groupby(group_cols, sort=False):
        occurrences = group["occurrences"].sum()
        weights = group["occurrences"].replace(0, np.nan)
        weighted.append(
            {
                **dict(zip(group_cols, key)),
                "symbols_with_event": int((group["occurrences"] > 0).sum()),
                "total_occurrences": int(occurrences),
                "avg_forward_return": float(np.average(group["avg_forward_return"], weights=weights)) if occurrences > 0 else np.nan,
                "median_of_symbol_medians": float(group["median_forward_return"].median()),
                "weighted_win_rate": float(np.average(group["win_rate"], weights=weights)) if occurrences > 0 else np.nan,
                "mean_std_forward_return": float(group["std_forward_return"].mean()),
                "best_symbol_avg_return": float(group["avg_forward_return"].max()),
                "worst_symbol_avg_return": float(group["avg_forward_return"].min()),
            }
        )
    event_summary = pd.DataFrame(weighted).sort_values(["horizon", "avg_forward_return"], ascending=[True, False])
    return event_summary, event_forward, event_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event discovery scan without trading logic.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--funding-dir", type=Path, default=Path("data") / "binance_um_funding")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "event_scan")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in [item.upper() for item in args.symbols]:
        funding_path = args.funding_dir / f"{symbol}.csv"
        funding = load_funding(funding_path) if funding_path.exists() else None
        if funding is not None:
            funding = funding[(funding["funding_time"] >= start) & (funding["funding_time"] <= end)]
        for timeframe in [item.lower() for item in args.timeframes]:
            path = args.data_dir / timeframe / f"{symbol}.csv"
            ohlcv = load_ohlcv(path)
            ohlcv = ohlcv[(ohlcv["timestamp"] >= start) & (ohlcv["timestamp"] < end)].reset_index(drop=True)
            features[(symbol, timeframe)] = add_features(ohlcv, funding)

    event_summary, event_forward, event_distribution = scan_events(features)
    event_summary.to_csv(args.out_dir / "event_summary.csv", index=False)
    event_forward.to_csv(args.out_dir / "event_forward_returns.csv", index=False)
    event_distribution.to_csv(args.out_dir / "event_distribution_stats.csv", index=False)

    print(f"Wrote event scan reports to {args.out_dir.resolve()}")
    if not event_summary.empty:
        print(event_summary.sort_values("avg_forward_return", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
