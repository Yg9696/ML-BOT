from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
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
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
ATR_PERIOD = 14
VOL_LOOKBACK_BARS = 96
VOL_THRESHOLD_LOOKBACK = 96 * 30
VOL_THRESHOLD_QUANTILE = 0.67


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class FilterSet:
    name: str
    event_threshold: float
    min_simultaneous_drops: int


@dataclass(frozen=True)
class ExitModel:
    name: str
    horizon_bars: int
    scaled_exit: bool = False
    stop_atr_mult: float | None = None


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in factors:
        raise ValueError(f"Unsupported interval: {interval}")
    return amount * factors[unit]


def download_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = interval_to_ms(interval)
    rows: list[list[object]] = []
    while cursor < end_ms:
        response = requests.get(
            BINANCE_FAPI_KLINES,
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"No klines returned for {symbol} {interval}")
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(out_path, index=False)


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    out["atr14"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["atr14_prior"] = out["atr14"].shift(1)
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["ema50_slope_24"] = out["ema50"] / out["ema50"].shift(24) - 1.0
    out["ema200_slope_24"] = out["ema200"] / out["ema200"].shift(24) - 1.0
    out["realized_vol_96"] = returns.rolling(VOL_LOOKBACK_BARS, min_periods=48).std()
    out["realized_vol_threshold"] = out["realized_vol_96"].rolling(VOL_THRESHOLD_LOOKBACK, min_periods=VOL_LOOKBACK_BARS * 7).quantile(VOL_THRESHOLD_QUANTILE).shift(1)
    out["high_vol_regime"] = out["realized_vol_96"] >= out["realized_vol_threshold"]
    return out


def data_quality(symbol: str, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    expected = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(minutes=15)
    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(expected.difference(timestamps))),
        "duplicate_rows": int(df["timestamp"].duplicated().sum()),
        "large_gap_count": int((gaps > expected_delta).sum()),
        "max_gap_minutes": float(gaps.max().total_seconds() / 60.0) if not gaps.empty else 0.0,
        "zero_volume_rows": int((df["volume"] <= 0).sum()),
        "bad_ohlc_rows": int(((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))).sum()),
    }


def ensure_data(symbols: list[str], data_dir: Path, start: pd.Timestamp, end: pd.Timestamp, download: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    for symbol in symbols:
        path = data_dir / TIMEFRAME / f"{symbol}.csv"
        needs_download = download or not path.exists()
        if not needs_download:
            probe = load_ohlcv(path)
            needs_download = not (probe["timestamp"].min() <= start and probe["timestamp"].max() >= end - pd.Timedelta(minutes=15))
        if needs_download:
            print(f"Downloading {symbol} {TIMEFRAME}")
            download_klines(symbol, TIMEFRAME, start, end, path)
        raw = load_ohlcv(path)
        raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)].reset_index(drop=True)
        loaded[symbol] = add_features(raw)
        quality_rows.append(data_quality(symbol, raw, start, end))
    return loaded, pd.DataFrame(quality_rows)


def add_market_context(frames: dict[str, pd.DataFrame]) -> None:
    btc = frames["BTCUSDT"][["timestamp", "close"]].copy()
    btc["btc_return_24h"] = btc["close"] / btc["close"].shift(96) - 1.0
    btc_context = btc[["timestamp", "btc_return_24h"]]
    event_counts = pd.concat(
        [
            frame[["timestamp", "event_return"]].assign(is_down_2=lambda x: x["event_return"] <= -0.02)
            for frame in frames.values()
        ]
    ).groupby("timestamp")["is_down_2"].sum().rename("simultaneous_down_2pct_count").reset_index()
    for symbol, frame in frames.items():
        frames[symbol] = frame.merge(btc_context, on="timestamp", how="left").merge(event_counts, on="timestamp", how="left")


def apply_entry_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate)


def apply_exit_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate)


def simulate_symbol(symbol: str, df: pd.DataFrame, entry_mode: str, filters: FilterSet, model: ExitModel, friction: Friction) -> list[dict[str, object]]:
    trades: list[dict[str, object]] = []
    signal_indices = np.flatnonzero(
        (
            (df["event_return"].to_numpy(dtype=float) <= filters.event_threshold)
            & df["high_vol_regime"].fillna(False).to_numpy(dtype=bool)
            & (df["simultaneous_down_2pct_count"].fillna(0).to_numpy(dtype=float) >= filters.min_simultaneous_drops)
            & (df["btc_return_24h"].to_numpy(dtype=float) < 0)
            & (df["ema50_slope_24"].to_numpy(dtype=float) < 0)
            & (df["ema200_slope_24"].to_numpy(dtype=float) < 0)
        )
    )
    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    event_returns = df["event_return"].to_numpy(dtype=float)
    atrs = df["atr14_prior"].to_numpy(dtype=float)
    realized_vols = df["realized_vol_96"].to_numpy(dtype=float)
    realized_vol_thresholds = df["realized_vol_threshold"].to_numpy(dtype=float)
    simultaneous_counts = df["simultaneous_down_2pct_count"].to_numpy(dtype=float)
    btc_returns = df["btc_return_24h"].to_numpy(dtype=float)
    ema50_slopes = df["ema50_slope_24"].to_numpy(dtype=float)
    ema200_slopes = df["ema200_slope_24"].to_numpy(dtype=float)
    for signal_i in signal_indices:
        entry_i = signal_i + 1
        final_exit_i = entry_i + model.horizon_bars
        if entry_i >= len(df) or final_exit_i >= len(df):
            continue
        atr = float(atrs[signal_i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        raw_entry = closes[entry_i] if entry_mode == "next_close" else opens[entry_i]
        entry_price = apply_entry_slippage(raw_entry, friction)
        entry_fee = friction.taker_fee_rate
        stop = entry_price - model.stop_atr_mult * atr if model.stop_atr_mult is not None else np.nan
        exit_legs: list[tuple[int, float, str]] = []
        exit_reason = "horizon"
        if model.stop_atr_mult is not None:
            first_check_i = entry_i if entry_mode == "next_open" else entry_i + 1
            for candle_i in range(first_check_i, final_exit_i + 1):
                if lows[candle_i] <= stop:
                    exit_legs = [(candle_i, 1.0, "stop")]
                    exit_reason = "stop"
                    break
        if not exit_legs:
            if model.scaled_exit:
                exit_legs = [(entry_i + 6, 0.5, "scaled_6"), (entry_i + 12, 0.5, "scaled_12")]
                exit_reason = "scaled_horizon"
            else:
                exit_legs = [(final_exit_i, 1.0, "horizon")]
        gross_return = 0.0
        exit_fee = 0.0
        weighted_exit_price = 0.0
        leg_details = []
        for exit_i, weight, leg_reason in exit_legs:
            raw_exit = stop if leg_reason == "stop" else closes[exit_i]
            exit_price = apply_exit_slippage(raw_exit, friction)
            gross_return += weight * (exit_price / entry_price - 1.0)
            exit_fee += weight * friction.taker_fee_rate
            weighted_exit_price += weight * exit_price
            leg_details.append(f"{leg_reason}:{weight}@{pd.Timestamp(timestamps[exit_i]).isoformat()}")
        exit_i = max(leg[0] for leg in exit_legs)
        net_return = gross_return - entry_fee - exit_fee
        trades.append(
            {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "entry_mode": entry_mode,
                "filter_name": filters.name,
                "event_threshold": filters.event_threshold,
                "min_simultaneous_drops": filters.min_simultaneous_drops,
                "exit_model": model.name,
                "friction_case": friction.name,
                "signal_time": timestamps[signal_i],
                "entry_time": timestamps[entry_i],
                "exit_time": timestamps[exit_i],
                "signal_return": event_returns[signal_i],
                "entry_price": entry_price,
                "exit_price": weighted_exit_price,
                "atr14_prior": atr,
                "realized_vol_96": realized_vols[signal_i],
                "realized_vol_threshold": realized_vol_thresholds[signal_i],
                "simultaneous_down_2pct_count": simultaneous_counts[signal_i],
                "btc_return_24h": btc_returns[signal_i],
                "ema50_slope_24": ema50_slopes[signal_i],
                "ema200_slope_24": ema200_slopes[signal_i],
                "horizon_bars": model.horizon_bars,
                "exit_reason": exit_reason,
                "exit_legs": ";".join(leg_details),
                "gross_return": gross_return,
                "fee_return": entry_fee + exit_fee,
                "net_return": net_return,
                "net_return_pct": net_return * 100.0,
                "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                "slippage_bps_per_side": friction.slippage_rate * 10_000,
            }
        )
    return trades


def max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return 0.0
    equity = series.cumsum()
    return float((equity - equity.cummax()).min())


def max_losing_streak(values: Iterable[float]) -> int:
    worst = current = 0
    for value in values:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def exposure_stats(group: pd.DataFrame) -> tuple[float, int]:
    changes: list[tuple[pd.Timestamp, int]] = []
    for row in group.itertuples(index=False):
        changes.append((row.entry_time, 1))
        changes.append((row.exit_time, -1))
    active = 0
    samples = []
    max_active = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], -item[1])):
        active += delta
        max_active = max(max_active, active)
        samples.append(active)
    return (float(np.mean(samples)) if samples else 0.0, max_active)


def summarize_group(group: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_return"]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    years = max((ordered["entry_time"].max() - ordered["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    trade_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe_like = float((returns.mean() / trade_std) * np.sqrt(len(returns))) if trade_std > 0 else 0.0
    annualized_sharpe_like = float((returns.mean() / trade_std) * np.sqrt(len(returns) / years)) if trade_std > 0 else 0.0
    month_returns = ordered.assign(month=ordered["exit_time"].dt.strftime("%Y-%m")).groupby("month")["net_return"].sum()
    symbol_returns = ordered.groupby("symbol")["net_return"].sum()
    cumulative = float(returns.sum())
    avg_exposure, max_exposure = exposure_stats(ordered)
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "total_trades": int(len(ordered)),
            "trades_per_year": float(len(ordered) / years),
            "average_return_pct": float(returns.mean() * 100.0),
            "median_return_pct": float(returns.median() * 100.0),
            "cumulative_return_units": cumulative,
            "win_rate": float((returns > 0).mean()),
            "return_volatility_pct": trade_std * 100.0,
            "sharpe_like": sharpe_like,
            "annualized_trade_sharpe_like": annualized_sharpe_like,
            "max_drawdown_units": max_drawdown(returns),
            "max_losing_streak": max_losing_streak(returns),
            "average_win_pct": float(wins.mean() * 100.0) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean() * 100.0) if not losses.empty else 0.0,
            "best_month": month_returns.idxmax(),
            "worst_month": month_returns.idxmin(),
            "best_symbol": symbol_returns.idxmax(),
            "worst_symbol": symbol_returns.idxmin(),
            "best_month_contribution": float(month_returns.max() / cumulative) if cumulative > 0 else np.nan,
            "best_symbol_contribution": float(symbol_returns.max() / cumulative) if cumulative > 0 else np.nan,
            "positive_month_rate": float((month_returns > 0).mean()),
            "positive_symbol_rate": float((symbol_returns > 0).mean()),
            "average_concurrent_positions": avg_exposure,
            "max_concurrent_positions": max_exposure,
        }
    )
    return out


def build_equity_curve(trades: pd.DataFrame, scenario_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, group in trades.groupby(scenario_cols, sort=False, dropna=False):
        ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
        cumulative = 0.0
        entries = ordered["entry_time"].sort_values().reset_index(drop=True)
        exits = ordered["exit_time"].sort_values().reset_index(drop=True)
        for trade in ordered.itertuples(index=False):
            cumulative += trade.net_return
            active_after_exit = int((entries <= trade.exit_time).sum() - (exits <= trade.exit_time).sum())
            row = {col: getattr(trade, col) for col in scenario_cols}
            row.update(
                {
                    "timestamp": trade.exit_time,
                    "symbol": trade.symbol,
                    "realized_return": trade.net_return,
                    "cumulative_return": cumulative,
                    "active_positions_after_exit": active_after_exit,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_reports(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "entry_mode",
        "filter_name",
        "event_threshold",
        "min_simultaneous_drops",
        "exit_model",
        "friction_case",
        "horizon_bars",
        "taker_fee_bps_per_side",
        "slippage_bps_per_side",
    ]
    metrics = pd.DataFrame([summarize_group(group, scenario_cols) for _, group in trades.groupby(scenario_cols, sort=False, dropna=False)])
    by_symbol = pd.DataFrame([summarize_group(group, scenario_cols + ["symbol"]) for _, group in trades.groupby(scenario_cols + ["symbol"], sort=False, dropna=False)])
    by_month_trades = trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m"))
    by_month = pd.DataFrame([summarize_group(group, scenario_cols + ["month"]) for _, group in by_month_trades.groupby(scenario_cols + ["month"], sort=False, dropna=False)])
    equity = build_equity_curve(trades, scenario_cols)
    pivot_keys = ["filter_name", "event_threshold", "min_simultaneous_drops", "exit_model", "friction_case", "horizon_bars"]
    open_rows = metrics[metrics["entry_mode"] == "next_open"].set_index(pivot_keys)
    close_rows = metrics[metrics["entry_mode"] == "next_close"].set_index(pivot_keys)
    comparison = open_rows[["total_trades", "average_return_pct", "cumulative_return_units", "sharpe_like", "max_drawdown_units"]].join(
        close_rows[["total_trades", "average_return_pct", "cumulative_return_units", "sharpe_like", "max_drawdown_units"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["cum_return_delta_close_minus_open"] = comparison["cumulative_return_units_next_close"] - comparison["cumulative_return_units_next_open"]
    comparison["avg_return_delta_close_minus_open_pct"] = comparison["average_return_pct_next_close"] - comparison["average_return_pct_next_open"]
    return metrics, by_symbol, by_month, equity, comparison


def pass_fail_summary(metrics: pd.DataFrame, previous_sharpe_like: float = 2.7025104848474686) -> dict[str, object]:
    base = metrics[(metrics["friction_case"] == "base") & (metrics["entry_mode"] == "next_close")].copy()
    if base.empty:
        return {"minimum_pass": False, "reason": "no base next_close scenarios"}
    best = base.sort_values("cumulative_return_units", ascending=False).iloc[0]
    stress = metrics[
        (metrics["friction_case"] == "stress")
        & (metrics["entry_mode"] == best["entry_mode"])
        & (metrics["filter_name"] == best["filter_name"])
        & (metrics["exit_model"] == best["exit_model"])
    ]
    stress_return = float(stress["cumulative_return_units"].iloc[0]) if not stress.empty else np.nan
    checks = {
        "positive_cumulative_return": float(best["cumulative_return_units"]) > 0,
        "less_concentration": pd.notna(best["best_month_contribution"]) and float(best["best_month_contribution"]) <= 0.50,
        "trades_at_least_100": int(best["total_trades"]) >= 100,
        "stress_not_deeply_negative": pd.notna(stress_return) and stress_return > -0.25,
        "improved_sharpe_like_vs_previous": float(best["sharpe_like"]) > previous_sharpe_like,
        "robust_across_symbols": float(best["positive_symbol_rate"]) >= 0.67,
    }
    checks["minimum_pass"] = all(checks[key] for key in ["positive_cumulative_return", "less_concentration", "trades_at_least_100", "stress_not_deeply_negative"])
    checks["strong_pass"] = checks["minimum_pass"] and checks["improved_sharpe_like_vs_previous"] and checks["robust_across_symbols"]
    return {"best_base_next_close": best.to_dict(), "matched_stress_cumulative_return_units": stress_return, "checks": checks}


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable panic-reversal bot phase 1 backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "panic_reversal_bot")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames, quality = ensure_data(SYMBOLS, args.data_dir, start, end, args.download)
    add_market_context(frames)
    filter_sets = [
        FilterSet("drop3_breadth4", event_threshold=-0.03, min_simultaneous_drops=4),
        FilterSet("drop3_breadth5", event_threshold=-0.03, min_simultaneous_drops=5),
        FilterSet("drop4_breadth4", event_threshold=-0.04, min_simultaneous_drops=4),
        FilterSet("drop4_breadth5", event_threshold=-0.04, min_simultaneous_drops=5),
    ]
    models = [
        ExitModel("pure_horizon_12", horizon_bars=12),
        ExitModel("pure_horizon_6", horizon_bars=6),
        ExitModel("scaled_exit_6_12", horizon_bars=12, scaled_exit=True),
        ExitModel("wide_stop_2atr_horizon_12", horizon_bars=12, stop_atr_mult=2.0),
    ]
    frictions = [
        Friction("base", taker_fee_rate=0.0004, slippage_rate=0.0005),
        Friction("stress", taker_fee_rate=0.0008, slippage_rate=0.0010),
    ]
    records: list[dict[str, object]] = []
    for filters in filter_sets:
        for model in models:
            for friction in frictions:
                for entry_mode in ["next_close", "next_open"]:
                    for symbol, df in frames.items():
                        records.extend(simulate_symbol(symbol, df, entry_mode, filters, model, friction))
    trades = pd.DataFrame(records)
    if trades.empty:
        raise RuntimeError("No trades generated.")
    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades.sort_values(["filter_name", "exit_model", "friction_case", "entry_mode", "exit_time", "symbol"]).reset_index(drop=True)
    metrics, by_symbol, by_month, equity, comparison = build_reports(trades)
    pass_fail = pass_fail_summary(metrics)

    paths = {
        "equity_curve_csv": args.out_dir / "equity_curve.csv",
        "trade_returns_csv": args.out_dir / "trade_returns.csv",
        "cumulative_metrics_csv": args.out_dir / "cumulative_metrics.csv",
        "by_symbol_csv": args.out_dir / "by_symbol.csv",
        "by_month_csv": args.out_dir / "by_month.csv",
        "execution_comparison_csv": args.out_dir / "execution_comparison.csv",
        "data_quality_report_csv": args.out_dir / "data_quality_report.csv",
        "regime_filtered_summary_json": args.out_dir / "regime_filtered_summary.json",
    }
    equity.to_csv(paths["equity_curve_csv"], index=False)
    trades.to_csv(paths["trade_returns_csv"], index=False)
    metrics.sort_values("cumulative_return_units", ascending=False).to_csv(paths["cumulative_metrics_csv"], index=False)
    by_symbol.to_csv(paths["by_symbol_csv"], index=False)
    by_month.to_csv(paths["by_month_csv"], index=False)
    comparison.to_csv(paths["execution_comparison_csv"], index=False)
    quality.to_csv(paths["data_quality_report_csv"], index=False)
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Executable panic-reversal bot phase 1.",
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "filters": [asdict(item) for item in filter_sets],
        "volatility_filter": {
            "feature": "realized_vol_96",
            "lookback_bars": VOL_THRESHOLD_LOOKBACK,
            "rolling_quantile": VOL_THRESHOLD_QUANTILE,
            "threshold_shifted_by_one_bar": True,
        },
        "models": [asdict(item) for item in models],
        "friction": [asdict(item) for item in frictions],
        "total_trade_rows": int(len(trades)),
        "scenario_count": int(len(metrics)),
        "reports": {key: str(path) for key, path in paths.items()},
        "assumptions": [
            "Base event is 15m close-to-close downside return.",
            "Entry is market execution at next candle close or open.",
            "All models allow stacking with fixed size per signal.",
            "No ML, no limit orders, no OI/funding/liquidation inputs.",
            "High volatility threshold is rolling and shifted, so no full-sample future threshold is used.",
        ],
        "top_10_by_cumulative_return": metrics.sort_values("cumulative_return_units", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    paths["regime_filtered_summary_json"].write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(metrics.sort_values("cumulative_return_units", ascending=False).head(12).to_string(index=False))
    print(json.dumps(json_safe(pass_fail["checks"]), indent=2))


if __name__ == "__main__":
    main()
