from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
START = "2020-01-01T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
ATR_PERIOD = 14
VOL_LOOKBACK_BARS = 96
VOL_THRESHOLD_LOOKBACK = 96 * 30
VOL_THRESHOLD_QUANTILE = 0.67
LISTING_STARTS = {
    "BTCUSDT": "2020-01-01T00:00:00+00:00",
    "ETHUSDT": "2020-01-01T00:00:00+00:00",
    "XRPUSDT": "2020-01-06T08:15:00+00:00",
    "LTCUSDT": "2020-01-09T08:00:00+00:00",
    "LINKUSDT": "2020-01-17T08:00:00+00:00",
    "ADAUSDT": "2020-01-31T08:00:00+00:00",
    "BNBUSDT": "2020-02-10T08:00:00+00:00",
    "SOLUSDT": "2020-09-14T07:00:00+00:00",
    "AVAXUSDT": "2020-09-23T07:00:00+00:00",
}


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class ExitModel:
    name: str
    horizon_bars: int
    scaled_exit: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    event_threshold: float = -0.03
    min_simultaneous_drops: int = 4
    btc_24h_must_be_negative: bool = True
    ema50_slope_must_be_negative: bool = True
    ema200_slope_must_be_negative: bool = True


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

    def save_rows() -> None:
        if not rows:
            return
        df = pd.DataFrame(rows, columns=columns)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
        df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
        df.to_csv(out_path, index=False)

    while cursor < end_ms:
        for attempt in range(6):
            try:
                response = requests.get(
                    BINANCE_FAPI_KLINES,
                    params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
                    timeout=30,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 5:
                    save_rows()
                    raise
                time.sleep(2 * (attempt + 1))
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        if len(rows) % 30_000 == 0:
            save_rows()
        next_cursor = int(batch[-1][0]) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.04)
    if not rows:
        raise RuntimeError(f"No klines returned for {symbol} {interval}")
    save_rows()


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
    prior_returns = returns.shift(1)
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
    out["realized_vol_96"] = prior_returns.rolling(VOL_LOOKBACK_BARS, min_periods=48).std()
    out["realized_vol_threshold"] = out["realized_vol_96"].rolling(VOL_THRESHOLD_LOOKBACK, min_periods=VOL_LOOKBACK_BARS * 7).quantile(VOL_THRESHOLD_QUANTILE).shift(1)
    out["high_vol_regime"] = out["realized_vol_96"] >= out["realized_vol_threshold"]
    return out


def listing_start(symbol: str, requested_start: pd.Timestamp) -> pd.Timestamp:
    known = parse_utc(LISTING_STARTS.get(symbol, requested_start.isoformat()))
    return max(requested_start, known)


def data_quality(symbol: str, df: pd.DataFrame, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> dict[str, object]:
    expected = pd.date_range(start=requested_start, end=requested_end, freq="15min", inclusive="left")
    listing_aware_start = listing_start(symbol, requested_start)
    listing_expected = pd.date_range(start=listing_aware_start, end=requested_end, freq="15min", inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(minutes=15)
    first_ts = timestamps.min() if len(timestamps) else pd.NaT
    last_ts = timestamps.max() if len(timestamps) else pd.NaT
    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "listing_aware_start": listing_aware_start.isoformat(),
        "first_timestamp": first_ts.isoformat() if pd.notna(first_ts) else None,
        "last_timestamp": last_ts.isoformat() if pd.notna(last_ts) else None,
        "rows": int(len(df)),
        "expected_rows_from_requested_start": int(len(expected)),
        "missing_rows_vs_requested_start": int(len(expected.difference(timestamps))),
        "pre_listing_unavailable_rows": max(0, int((listing_aware_start - requested_start) / pd.Timedelta(minutes=15))),
        "expected_rows_from_listing_start": int(len(listing_expected)),
        "true_internal_missing_rows": int(len(listing_expected.difference(timestamps))),
        "duplicate_rows": int(df["timestamp"].duplicated().sum()),
        "large_gap_count": int((gaps > expected_delta).sum()),
        "max_gap_minutes": float(gaps.max().total_seconds() / 60.0) if not gaps.empty else 0.0,
        "zero_volume_rows": int((df["volume"] <= 0).sum()),
        "bad_ohlc_rows": int(((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))).sum()),
    }


def ensure_data(symbols: list[str], data_dir: Path, start: pd.Timestamp, end: pd.Timestamp, download: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    for symbol in symbols:
        path = data_dir / TIMEFRAME / f"{symbol}.csv"
        needs_download = download or not path.exists()
        if not needs_download:
            probe = load_ohlcv(path)
            required_start = listing_start(symbol, start)
            needs_download = (
                probe["timestamp"].max() < end - pd.Timedelta(minutes=15)
                or probe["timestamp"].min() > required_start + pd.Timedelta(minutes=15)
            )
        if needs_download:
            print(f"Downloading {symbol} {TIMEFRAME} from {start.isoformat()} to {end.isoformat()}")
            download_klines(symbol, TIMEFRAME, start, end, path)
        raw = load_ohlcv(path)
        raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)].reset_index(drop=True)
        quality_rows.append(data_quality(symbol, raw, start, end))
        frames[symbol] = add_features(raw)
    return frames, pd.DataFrame(quality_rows)


def add_market_context(frames: dict[str, pd.DataFrame]) -> None:
    btc = frames["BTCUSDT"][["timestamp", "close"]].copy()
    btc["btc_return_24h"] = btc["close"] / btc["close"].shift(96) - 1.0
    event_counts = pd.concat(
        [
            frame[["timestamp", "event_return"]].assign(is_down_2=lambda x: x["event_return"] <= -0.02)
            for frame in frames.values()
        ]
    ).groupby("timestamp")["is_down_2"].sum().rename("simultaneous_down_2pct_count").reset_index()
    btc_context = btc[["timestamp", "btc_return_24h"]]
    for symbol, frame in frames.items():
        frames[symbol] = frame.merge(btc_context, on="timestamp", how="left").merge(event_counts, on="timestamp", how="left")


def apply_entry_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate)


def apply_exit_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate)


def simulate_symbol(symbol: str, df: pd.DataFrame, entry_mode: str, model: ExitModel, friction: Friction, config: StrategyConfig) -> list[dict[str, object]]:
    signal_indices = np.flatnonzero(
        (
            (df["event_return"].to_numpy(dtype=float) <= config.event_threshold)
            & df["high_vol_regime"].fillna(False).to_numpy(dtype=bool)
            & (df["simultaneous_down_2pct_count"].fillna(0).to_numpy(dtype=float) >= config.min_simultaneous_drops)
            & (df["btc_return_24h"].to_numpy(dtype=float) < 0)
            & (df["ema50_slope_24"].to_numpy(dtype=float) < 0)
            & (df["ema200_slope_24"].to_numpy(dtype=float) < 0)
        )
    )
    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    event_returns = df["event_return"].to_numpy(dtype=float)
    realized_vols = df["realized_vol_96"].to_numpy(dtype=float)
    thresholds = df["realized_vol_threshold"].to_numpy(dtype=float)
    counts = df["simultaneous_down_2pct_count"].to_numpy(dtype=float)
    btc_returns = df["btc_return_24h"].to_numpy(dtype=float)
    ema50_slopes = df["ema50_slope_24"].to_numpy(dtype=float)
    ema200_slopes = df["ema200_slope_24"].to_numpy(dtype=float)
    trades: list[dict[str, object]] = []
    bar_delta = pd.Timedelta(minutes=15)
    for signal_i in signal_indices:
        entry_i = signal_i + 1
        final_exit_i = entry_i + model.horizon_bars
        if entry_i >= len(df) or final_exit_i >= len(df):
            continue
        raw_entry = closes[entry_i] if entry_mode == "next_close" else opens[entry_i]
        entry_price = apply_entry_slippage(float(raw_entry), friction)
        if model.scaled_exit:
            exit_legs = [(entry_i + 6, 0.5, "scaled_6"), (entry_i + 12, 0.5, "scaled_12")]
        else:
            exit_legs = [(final_exit_i, 1.0, "horizon")]
        gross_return = 0.0
        exit_fee = 0.0
        weighted_exit_price = 0.0
        leg_details: list[str] = []
        for exit_i, weight, reason in exit_legs:
            raw_exit = closes[exit_i]
            exit_price = apply_exit_slippage(float(raw_exit), friction)
            gross_return += weight * (exit_price / entry_price - 1.0)
            exit_fee += weight * friction.taker_fee_rate
            weighted_exit_price += weight * exit_price
            leg_details.append(f"{reason}:{weight}@{(pd.Timestamp(timestamps[exit_i]) + bar_delta).isoformat()}")
        exit_i = max(leg[0] for leg in exit_legs)
        net_return = gross_return - friction.taker_fee_rate - exit_fee
        signal_open_time = pd.Timestamp(timestamps[signal_i])
        entry_candle_open_time = pd.Timestamp(timestamps[entry_i])
        entry_time = entry_candle_open_time + bar_delta if entry_mode == "next_close" else entry_candle_open_time
        exit_time = pd.Timestamp(timestamps[exit_i]) + bar_delta
        trades.append(
            {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "entry_mode": entry_mode,
                "entry_time_semantic": "next_candle_close" if entry_mode == "next_close" else "next_candle_open",
                "exit_model": model.name,
                "friction_case": friction.name,
                "friction_mode": friction.name,
                "event_time": signal_open_time,
                "signal_time": signal_open_time,
                "signal_close_time": signal_open_time + bar_delta,
                "entry_candle_open_time": entry_candle_open_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "signal_return": event_returns[signal_i],
                "event_return_pct": event_returns[signal_i] * 100.0,
                "entry_price": entry_price,
                "exit_price": weighted_exit_price,
                "realized_vol_96": realized_vols[signal_i],
                "realized_vol_regime_value": realized_vols[signal_i],
                "realized_vol_threshold": thresholds[signal_i],
                "simultaneous_down_2pct_count": counts[signal_i],
                "breadth_count": counts[signal_i],
                "btc_return_24h": btc_returns[signal_i],
                "ema50_slope_24": ema50_slopes[signal_i],
                "ema50_slope": ema50_slopes[signal_i],
                "ema200_slope_24": ema200_slopes[signal_i],
                "ema200_slope": ema200_slopes[signal_i],
                "horizon_bars": model.horizon_bars,
                "exit_legs": ";".join(leg_details),
                "gross_return": gross_return,
                "fee_return": friction.taker_fee_rate + exit_fee,
                "net_return": net_return,
                "net_return_pct": net_return * 100.0,
                "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                "slippage_bps_per_side": friction.slippage_rate * 10_000,
            }
        )
    return trades


def add_concurrent_positions(trades: pd.DataFrame) -> pd.DataFrame:
    scenario_cols = ["entry_mode", "exit_model", "friction_case", "horizon_bars", "taker_fee_bps_per_side", "slippage_bps_per_side"]
    out = trades.copy()
    out["concurrent_positions"] = 0
    for _, idx in out.groupby(scenario_cols, sort=False, dropna=False).groups.items():
        group = out.loc[idx].sort_values(["entry_time", "exit_time", "symbol"]).copy()
        starts = group["entry_time"].sort_values().to_numpy()
        ends = group["exit_time"].sort_values().to_numpy()
        concurrent = np.searchsorted(starts, group["entry_time"].to_numpy(), side="right") - np.searchsorted(ends, group["entry_time"].to_numpy(), side="left")
        out.loc[group.index, "concurrent_positions"] = concurrent
    return out


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


def exposure_stats(group: pd.DataFrame) -> tuple[float, int, float]:
    changes: list[tuple[pd.Timestamp, int]] = []
    for row in group.itertuples(index=False):
        changes.append((row.entry_time, 1))
        changes.append((row.exit_time, -1))
    active = 0
    samples: list[int] = []
    for _, delta in sorted(changes, key=lambda item: (item[0], -item[1])):
        active += delta
        samples.append(active)
    if not samples:
        return 0.0, 0, 0.0
    return float(np.mean(samples)), int(np.max(samples)), float(np.quantile(samples, 0.95))


def summarize_group(group: pd.DataFrame, group_cols: list[str], window_start: pd.Timestamp | None = None, window_end: pd.Timestamp | None = None) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_return"]
    if window_start is not None and window_end is not None:
        years = max((window_end - window_start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    else:
        years = max((ordered["entry_time"].max() - ordered["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    cumulative = float(returns.sum())
    month_returns = ordered.assign(month=ordered["exit_time"].dt.strftime("%Y-%m")).groupby("month")["net_return"].sum()
    symbol_returns = ordered.groupby("symbol")["net_return"].sum()
    avg_exposure, max_exposure, p95_exposure = exposure_stats(ordered)
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "total_trades": int(len(ordered)),
            "trades_per_year": float(len(ordered) / years),
            "average_return_pct": float(returns.mean() * 100.0),
            "median_return_pct": float(returns.median() * 100.0),
            "win_rate": float((returns > 0).mean()),
            "cumulative_return_units": cumulative,
            "max_drawdown_units": max_drawdown(returns),
            "sharpe_like": float((returns.mean() / std) * np.sqrt(len(returns))) if std > 0 else 0.0,
            "return_volatility_pct": std * 100.0,
            "max_losing_streak": max_losing_streak(returns),
            "average_win_pct": float(wins.mean() * 100.0) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean() * 100.0) if not losses.empty else 0.0,
            "best_month": month_returns.idxmax(),
            "worst_month": month_returns.idxmin(),
            "best_month_contribution": float(month_returns.max() / cumulative) if cumulative > 0 else np.nan,
            "best_symbol": symbol_returns.idxmax(),
            "worst_symbol": symbol_returns.idxmin(),
            "best_symbol_contribution": float(symbol_returns.max() / cumulative) if cumulative > 0 else np.nan,
            "positive_month_rate": float((month_returns > 0).mean()),
            "positive_symbol_rate": float((symbol_returns > 0).mean()),
            "average_concurrent_positions": avg_exposure,
            "max_concurrent_positions": max_exposure,
            "p95_concurrent_positions": p95_exposure,
        }
    )
    return out


def windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    last3_start = max(start, end - pd.DateOffset(years=3))
    items = [("full_available", start, end), ("last_3_years", last3_start, end)]
    for year in [2023, 2024, 2025, 2026]:
        y_start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
        y_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
        items.append((str(year) if year < 2026 else "2026_ytd", max(start, y_start), min(end, y_end)))
    return [(name, w_start, w_end) for name, w_start, w_end in items if w_start < w_end]


def build_windowed_metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = ["entry_mode", "exit_model", "friction_case", "horizon_bars", "taker_fee_bps_per_side", "slippage_bps_per_side"]
    overall_rows = []
    symbol_rows = []
    month_rows = []
    year_rows = []
    for window_name, w_start, w_end in windows(start, end):
        sample = trades[(trades["signal_time"] >= w_start) & (trades["signal_time"] < w_end)].copy()
        if sample.empty:
            continue
        for _, group in sample.groupby(scenario_cols, sort=False, dropna=False):
            row = summarize_group(group, scenario_cols, w_start, w_end)
            row.update({"window": window_name, "window_start": w_start.isoformat(), "window_end": w_end.isoformat()})
            overall_rows.append(row)
        for _, group in sample.groupby(scenario_cols + ["symbol"], sort=False, dropna=False):
            row = summarize_group(group, scenario_cols + ["symbol"], w_start, w_end)
            row.update({"window": window_name, "window_start": w_start.isoformat(), "window_end": w_end.isoformat()})
            symbol_rows.append(row)
        by_month_sample = sample.assign(month=sample["exit_time"].dt.strftime("%Y-%m"))
        for _, group in by_month_sample.groupby(scenario_cols + ["month"], sort=False, dropna=False):
            row = summarize_group(group, scenario_cols + ["month"], w_start, w_end)
            row.update({"window": window_name, "window_start": w_start.isoformat(), "window_end": w_end.isoformat()})
            month_rows.append(row)
        if window_name in ["2023", "2024", "2025", "2026_ytd"]:
            for _, group in sample.groupby(scenario_cols, sort=False, dropna=False):
                row = summarize_group(group, scenario_cols, w_start, w_end)
                row.update({"year": window_name, "window_start": w_start.isoformat(), "window_end": w_end.isoformat()})
                year_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(year_rows), pd.DataFrame(month_rows), pd.DataFrame(symbol_rows)


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    scenario_cols = ["entry_mode", "exit_model", "friction_case", "horizon_bars", "taker_fee_bps_per_side", "slippage_bps_per_side"]
    rows = []
    for _, group in trades.groupby(scenario_cols, sort=False, dropna=False):
        ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
        cumulative = 0.0
        for trade in ordered.itertuples(index=False):
            cumulative += trade.net_return
            row = {col: getattr(trade, col) for col in scenario_cols}
            row.update({"timestamp": trade.exit_time, "symbol": trade.symbol, "realized_return": trade.net_return, "cumulative_return": cumulative})
            rows.append(row)
    return pd.DataFrame(rows)


def build_execution_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    keys = ["window", "exit_model", "friction_case", "horizon_bars"]
    open_rows = overall[overall["entry_mode"] == "next_open"].set_index(keys)
    close_rows = overall[overall["entry_mode"] == "next_close"].set_index(keys)
    comparison = open_rows[["total_trades", "average_return_pct", "cumulative_return_units", "sharpe_like", "max_drawdown_units"]].join(
        close_rows[["total_trades", "average_return_pct", "cumulative_return_units", "sharpe_like", "max_drawdown_units"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["cum_return_delta_close_minus_open"] = comparison["cumulative_return_units_next_close"] - comparison["cumulative_return_units_next_open"]
    comparison["avg_return_delta_close_minus_open_pct"] = comparison["average_return_pct_next_close"] - comparison["average_return_pct_next_open"]
    return comparison


def pass_fail(overall: pd.DataFrame) -> dict[str, object]:
    full = overall[
        (overall["window"] == "full_available")
        & (overall["entry_mode"] == "next_close")
        & (overall["exit_model"] == "pure_horizon_12")
        & (overall["friction_case"] == "base")
    ]
    if full.empty:
        return {"minimum_pass": False, "reason": "no full next_close pure_horizon_12 base row"}
    row = full.iloc[0]
    stress = overall[
        (overall["window"] == "full_available")
        & (overall["entry_mode"] == "next_close")
        & (overall["exit_model"] == "pure_horizon_12")
        & (overall["friction_case"] == "stress")
    ]
    stress_return = float(stress["cumulative_return_units"].iloc[0]) if not stress.empty else np.nan
    checks = {
        "positive_full_period": float(row["cumulative_return_units"]) > 0,
        "trades_at_least_150": int(row["total_trades"]) >= 150,
        "stress_positive_or_not_deeply_negative": pd.notna(stress_return) and stress_return > -0.25,
        "month_concentration_ok": pd.notna(row["best_month_contribution"]) and float(row["best_month_contribution"]) <= 0.50,
        "symbol_concentration_ok": pd.notna(row["best_symbol_contribution"]) and float(row["best_symbol_contribution"]) <= 0.40,
    }
    checks["minimum_pass"] = all(checks.values())
    return {"full_primary_base": row.to_dict(), "matched_stress_cumulative_return_units": stress_return, "checks": checks}


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


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended-history validation for the panic-reversal bot.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "panic_reversal_extended")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = StrategyConfig()
    frames, quality = ensure_data(SYMBOLS, args.data_dir, start, end, args.download)
    add_market_context(frames)
    models = [
        ExitModel("pure_horizon_12", 12),
        ExitModel("pure_horizon_6", 6),
        ExitModel("scaled_exit_6_12", 12, scaled_exit=True),
    ]
    frictions = [
        Friction("base", 0.0004, 0.0005),
        Friction("stress", 0.0008, 0.0010),
    ]
    records: list[dict[str, object]] = []
    for model in models:
        for friction in frictions:
            for entry_mode in ["next_close", "next_open"]:
                for symbol, df in frames.items():
                    records.extend(simulate_symbol(symbol, df, entry_mode, model, friction, config))
    trades = pd.DataFrame(records)
    if trades.empty:
        raise RuntimeError("No trades generated.")
    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)
    for col in ["event_time", "signal_close_time", "entry_candle_open_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = add_concurrent_positions(trades)
    trades = trades.sort_values(["exit_model", "friction_case", "entry_mode", "exit_time", "symbol"]).reset_index(drop=True)
    overall, by_year, by_month, by_symbol = build_windowed_metrics(trades, start, end)
    execution_comparison = build_execution_comparison(overall)
    equity = build_equity_curve(trades)
    validation = pass_fail(overall)

    paths = {
        "overall_results_csv": args.out_dir / "overall_results.csv",
        "by_year_csv": args.out_dir / "by_year.csv",
        "by_month_csv": args.out_dir / "by_month.csv",
        "by_symbol_csv": args.out_dir / "by_symbol.csv",
        "execution_comparison_csv": args.out_dir / "execution_comparison.csv",
        "equity_curve_csv": args.out_dir / "equity_curve.csv",
        "trade_returns_csv": args.out_dir / "trade_returns.csv",
        "trade_returns_sample_csv": args.out_dir / "trade_returns_sample.csv",
        "data_quality_report_csv": args.out_dir / "data_quality_report.csv",
        "run_config_json": args.out_dir / "run_config.json",
        "extended_validation_summary_json": args.out_dir / "extended_validation_summary.json",
    }
    overall.to_csv(paths["overall_results_csv"], index=False)
    by_year.to_csv(paths["by_year_csv"], index=False)
    by_month.to_csv(paths["by_month_csv"], index=False)
    by_symbol.to_csv(paths["by_symbol_csv"], index=False)
    execution_comparison.to_csv(paths["execution_comparison_csv"], index=False)
    equity.to_csv(paths["equity_curve_csv"], index=False)
    trades.to_csv(paths["trade_returns_csv"], index=False)
    trades.head(5000).to_csv(paths["trade_returns_sample_csv"], index=False)
    quality.to_csv(paths["data_quality_report_csv"], index=False)
    run_config = {
        "command": " ".join(sys.argv),
        "download_used": bool(args.download),
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "git_commit_hash_at_run_start": git_commit_hash(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "strategy_config": asdict(config),
        "entry_modes": ["next_close", "next_open"],
        "primary_entry_mode": "next_close",
        "exit_models": [asdict(model) for model in models],
        "primary_exit_model": "pure_horizon_12",
        "friction": [asdict(friction) for friction in frictions],
        "volatility_filter": {
            "feature": "realized_vol_96",
            "prior_only": True,
            "lookback_bars": VOL_LOOKBACK_BARS,
            "threshold_lookback_bars": VOL_THRESHOLD_LOOKBACK,
            "rolling_quantile": VOL_THRESHOLD_QUANTILE,
            "threshold_shifted_by_one_bar": True,
        },
    }
    paths["run_config_json"].write_text(json.dumps(json_safe(run_config), indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Extended-history validation for exact Panic-Reversal Bot Phase 1 configuration.",
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "strategy_config": asdict(config),
        "run_config": run_config,
        "git_commit_hash_at_run_start": run_config["git_commit_hash_at_run_start"],
        "download_used": bool(args.download),
        "command": run_config["command"],
        "volatility_filter": {
            "feature": "realized_vol_96",
            "prior_only": True,
            "lookback_bars": VOL_LOOKBACK_BARS,
            "threshold_lookback_bars": VOL_THRESHOLD_LOOKBACK,
            "rolling_quantile": VOL_THRESHOLD_QUANTILE,
            "threshold_shifted_by_one_bar": True,
        },
        "models": [asdict(model) for model in models],
        "friction": [asdict(friction) for friction in frictions],
        "total_trade_rows": int(len(trades)),
        "reports": {key: str(path) for key, path in paths.items()},
        "data_coverage": quality.to_dict(orient="records"),
        "pass_fail": validation,
        "top_overall_rows": overall.sort_values("cumulative_return_units", ascending=False).head(12).to_dict(orient="records"),
    }
    paths["extended_validation_summary_json"].write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    primary = overall[
        (overall["entry_mode"] == "next_close")
        & (overall["exit_model"] == "pure_horizon_12")
        & (overall["friction_case"] == "base")
    ].sort_values("window")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(primary.to_string(index=False))
    print(json.dumps(json_safe(validation["checks"]), indent=2))


if __name__ == "__main__":
    main()
