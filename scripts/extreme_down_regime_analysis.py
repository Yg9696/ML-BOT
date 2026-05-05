from __future__ import annotations

import argparse
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
TIMEFRAME = "15m"
EVENT_RETURN_THRESHOLD = -0.02
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_ohlcv(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def load_funding(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["funding_time", "funding_rate", "funding_change"])
    df = pd.read_csv(path)
    df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, format="mixed")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna().drop_duplicates("funding_time").sort_values("funding_time")
    df["funding_change"] = df["funding_rate"].diff()
    return df[["funding_time", "funding_rate", "funding_change"]]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    returns = out["close"].pct_change()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["event_return"] = returns
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["ema50_slope_24"] = out["ema50"] / out["ema50"].shift(24) - 1.0
    out["ema200_slope_24"] = out["ema200"] / out["ema200"].shift(24) - 1.0
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["atr14_prior"] = out["atr14"].shift(1)
    out["atr_rel_96"] = out["atr14_prior"] / out["atr14_prior"].rolling(96, min_periods=48).median()
    out["realized_vol_96"] = returns.rolling(96, min_periods=48).std()
    out["distance_ema50_atr"] = (out["close"] - out["ema50"]) / out["atr14_prior"]
    out["distance_ema200_atr"] = (out["close"] - out["ema200"]) / out["atr14_prior"]
    out["prior_move_24h"] = out["close"].shift(1) / out["close"].shift(97) - 1.0
    out["prior_move_6h"] = out["close"].shift(1) / out["close"].shift(25) - 1.0
    return out


def add_btc_context(frames: dict[str, pd.DataFrame]) -> None:
    btc = frames["BTCUSDT"][["timestamp", "close", "event_return"]].copy()
    btc["btc_return_24h"] = btc["close"] / btc["close"].shift(96) - 1.0
    btc["btc_vol_24h"] = btc["event_return"].rolling(96, min_periods=48).std()
    btc_context = btc[["timestamp", "btc_return_24h", "btc_vol_24h"]]
    for symbol, df in frames.items():
        frames[symbol] = df.merge(btc_context, on="timestamp", how="left")


def add_cross_asset_context(frames: dict[str, pd.DataFrame]) -> None:
    event_counts = []
    for symbol, df in frames.items():
        event_counts.append(df[["timestamp", "event_return"]].assign(is_extreme_down=lambda x: x["event_return"] <= EVENT_RETURN_THRESHOLD))
    counts = pd.concat(event_counts).groupby("timestamp")["is_extreme_down"].sum().rename("simultaneous_extreme_down_count").reset_index()
    for symbol, df in frames.items():
        frames[symbol] = df.merge(counts, on="timestamp", how="left")


def bucket_signed(value: float, deadband: float = 0.0) -> str:
    if pd.isna(value):
        return "missing"
    if value > deadband:
        return "positive"
    if value < -deadband:
        return "negative"
    return "flat"


def bucket_drop(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value <= -0.04:
        return "<= -4%"
    if value <= -0.03:
        return "-3% to -4%"
    if value <= -0.025:
        return "-2.5% to -3%"
    return "-2% to -2.5%"


def bucket_simultaneous(value: float) -> str:
    if pd.isna(value):
        return "missing"
    value = int(value)
    if value >= 5:
        return "5+ symbols"
    if value >= 3:
        return "3-4 symbols"
    if value == 2:
        return "2 symbols"
    return "1 symbol"


def bucket_quantile(series: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    out = pd.Series("missing", index=series.index, dtype=object)
    valid = series.dropna()
    if valid.nunique() < 3:
        out.loc[valid.index] = valid.apply(lambda x: bucket_signed(float(x)))
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid, q=3, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        out.loc[valid.index] = valid.apply(lambda x: bucket_signed(float(x)))
    return out


def enrich_trades(trades: pd.DataFrame, frames: dict[str, pd.DataFrame], funding_dir: Path) -> pd.DataFrame:
    enriched = []
    context_cols = [
        "timestamp",
        "ema50_slope_24",
        "ema200_slope_24",
        "distance_ema50_atr",
        "distance_ema200_atr",
        "atr_rel_96",
        "realized_vol_96",
        "btc_return_24h",
        "btc_vol_24h",
        "event_return",
        "prior_move_24h",
        "prior_move_6h",
        "simultaneous_extreme_down_count",
    ]
    for symbol, group in trades.groupby("symbol", sort=False):
        context = frames[symbol][context_cols].copy()
        merged = group.merge(context, left_on="signal_time", right_on="timestamp", how="left").drop(columns=["timestamp"])
        funding = load_funding(funding_dir / f"{symbol}.csv")
        if not funding.empty:
            merged = pd.merge_asof(
                merged.sort_values("signal_time"),
                funding.sort_values("funding_time"),
                left_on="signal_time",
                right_on="funding_time",
                direction="backward",
            )
        else:
            merged["funding_time"] = pd.NaT
            merged["funding_rate"] = np.nan
            merged["funding_change"] = np.nan
        enriched.append(merged)
    out = pd.concat(enriched, ignore_index=True).sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    out["signal_month"] = out["signal_time"].dt.strftime("%Y-%m")
    out["ema50_slope_bucket"] = out["ema50_slope_24"].apply(lambda x: bucket_signed(x, 0.001))
    out["ema200_slope_bucket"] = out["ema200_slope_24"].apply(lambda x: bucket_signed(x, 0.0005))
    out["event_drop_bucket"] = out["event_return"].apply(bucket_drop)
    out["prior_24h_bucket"] = out["prior_move_24h"].apply(lambda x: bucket_signed(x, 0.005))
    out["prior_6h_bucket"] = out["prior_move_6h"].apply(lambda x: bucket_signed(x, 0.0025))
    out["btc_24h_bucket"] = out["btc_return_24h"].apply(lambda x: bucket_signed(x, 0.005))
    out["funding_level_bucket"] = out["funding_rate"].apply(lambda x: bucket_signed(x, 0.00005))
    out["funding_change_bucket"] = out["funding_change"].apply(lambda x: bucket_signed(x, 0.00002))
    out["simultaneous_drop_bucket"] = out["simultaneous_extreme_down_count"].apply(bucket_simultaneous)
    out["atr_regime_bucket"] = bucket_quantile(out["atr_rel_96"], ("low_atr_rel", "mid_atr_rel", "high_atr_rel"))
    out["realized_vol_bucket"] = bucket_quantile(out["realized_vol_96"], ("low_realized_vol", "mid_realized_vol", "high_realized_vol"))
    out["btc_vol_bucket"] = bucket_quantile(out["btc_vol_24h"], ("low_btc_vol", "mid_btc_vol", "high_btc_vol"))
    out["distance_ema50_bucket"] = bucket_quantile(out["distance_ema50_atr"], ("below_ema50_far", "near_ema50", "above_ema50_far"))
    out["distance_ema200_bucket"] = bucket_quantile(out["distance_ema200_atr"], ("below_ema200_far", "near_ema200", "above_ema200_far"))
    return out


def summarize_bucket(group: pd.DataFrame, feature: str, bucket: str, total_return: float) -> dict[str, object]:
    returns = group["net_return"]
    return {
        "feature": feature,
        "bucket": bucket,
        "trade_count": int(len(group)),
        "avg_return_pct": float(returns.mean() * 100.0),
        "median_return_pct": float(returns.median() * 100.0),
        "win_rate": float((returns > 0).mean()),
        "cumulative_return_units": float(returns.sum()),
        "contribution_to_total_return": float(returns.sum() / total_return) if total_return else np.nan,
        "best_month": group.groupby("signal_month")["net_return"].sum().idxmax(),
        "worst_month": group.groupby("signal_month")["net_return"].sum().idxmin(),
    }


def build_bucket_results(trades: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    total_return = float(trades["net_return"].sum())
    rows = []
    for feature in features:
        for bucket, group in trades.groupby(feature, dropna=False, sort=False):
            rows.append(summarize_bucket(group, feature, str(bucket), total_return))
    return pd.DataFrame(rows).sort_values(["feature", "cumulative_return_units"], ascending=[True, False])


def build_importance_like(bucket_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in bucket_results.groupby("feature", sort=False):
        good = group.sort_values("avg_return_pct", ascending=False).iloc[0]
        bad = group.sort_values("avg_return_pct", ascending=True).iloc[0]
        rows.append(
            {
                "feature": feature,
                "bucket_count": int(len(group)),
                "best_bucket": good["bucket"],
                "best_bucket_trade_count": int(good["trade_count"]),
                "best_bucket_avg_return_pct": float(good["avg_return_pct"]),
                "best_bucket_cumulative_return_units": float(good["cumulative_return_units"]),
                "worst_bucket": bad["bucket"],
                "worst_bucket_trade_count": int(bad["trade_count"]),
                "worst_bucket_avg_return_pct": float(bad["avg_return_pct"]),
                "worst_bucket_cumulative_return_units": float(bad["cumulative_return_units"]),
                "avg_return_spread_pct": float(good["avg_return_pct"] - bad["avg_return_pct"]),
                "positive_bucket_count": int((group["avg_return_pct"] > 0).sum()),
                "negative_bucket_count": int((group["avg_return_pct"] <= 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_return_spread_pct", ascending=False)


def build_regime_summary(bucket_results: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    total = pd.DataFrame(
        [
            {
                "section": "overall",
                "feature": "all",
                "bucket": "all",
                "trade_count": int(len(trades)),
                "avg_return_pct": float(trades["net_return"].mean() * 100.0),
                "median_return_pct": float(trades["net_return"].median() * 100.0),
                "win_rate": float((trades["net_return"] > 0).mean()),
                "cumulative_return_units": float(trades["net_return"].sum()),
                "contribution_to_total_return": 1.0,
            }
        ]
    )
    strongest = bucket_results[bucket_results["trade_count"] >= 20].sort_values("avg_return_pct", ascending=False).head(15).assign(section="strongest_positive")
    weakest = bucket_results[bucket_results["trade_count"] >= 20].sort_values("avg_return_pct", ascending=True).head(15).assign(section="edge_killers")
    largest = bucket_results.sort_values("cumulative_return_units", ascending=False).head(15).assign(section="largest_contributors")
    return pd.concat([total, strongest, weakest, largest], ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime discovery for the extreme-down micro-edge.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--funding-dir", type=Path, default=Path("data") / "binance_um_funding")
    parser.add_argument("--trade-returns", type=Path, default=Path("reports") / "extreme_down_micro_edge" / "trade_returns.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "extreme_down_regime_analysis")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {symbol: add_context_features(load_ohlcv(args.data_dir / TIMEFRAME / f"{symbol}.csv", start, end)) for symbol in SYMBOLS}
    add_btc_context(frames)
    add_cross_asset_context(frames)

    trades = pd.read_csv(args.trade_returns)
    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True, format="mixed")
    trades = trades[
        (trades["entry_mode"] == "next_close")
        & (trades["model"] == "pure_horizon_12_stacked")
        & (trades["friction_case"] == "base")
        & (trades["allow_stacking"].astype(str).str.lower().isin(["true", "1"]))
    ].copy()
    if trades.empty:
        raise RuntimeError("No trades matched next_close + pure_horizon_12_stacked + base.")

    enriched = enrich_trades(trades, frames, args.funding_dir)
    features = [
        "ema50_slope_bucket",
        "ema200_slope_bucket",
        "distance_ema50_bucket",
        "distance_ema200_bucket",
        "atr_regime_bucket",
        "realized_vol_bucket",
        "btc_24h_bucket",
        "btc_vol_bucket",
        "funding_level_bucket",
        "funding_change_bucket",
        "event_drop_bucket",
        "prior_24h_bucket",
        "prior_6h_bucket",
        "simultaneous_drop_bucket",
        "symbol",
        "signal_month",
    ]
    bucket_results = build_bucket_results(enriched, features)
    importance = build_importance_like(bucket_results)
    summary = build_regime_summary(bucket_results, enriched)

    summary.to_csv(args.out_dir / "regime_summary.csv", index=False)
    importance.to_csv(args.out_dir / "regime_feature_importance_like.csv", index=False)
    bucket_results.to_csv(args.out_dir / "regime_bucket_results.csv", index=False)

    print(f"Wrote reports to {args.out_dir.resolve()}")
    print("Overall:")
    print(summary[summary["section"] == "overall"].to_string(index=False))
    print("\nStrongest positive buckets:")
    print(summary[summary["section"] == "strongest_positive"].head(10).to_string(index=False))
    print("\nEdge-killer buckets:")
    print(summary[summary["section"] == "edge_killers"].head(10).to_string(index=False))
    print("\nLargest feature separations:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
