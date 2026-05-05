from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
TIMEFRAMES = ["1h", "4h"]
LOOKBACKS = [20, 50, 100]
ATR_MULTS = [1.2, 1.5, 2.0]
VOL_Z_MINS = [1.0, 1.5, 2.0]
CLOSE_LOC_LONG_MINS = [0.70, 0.75, 0.80]
CLOSE_LOC_SHORT_MAXS = [0.30, 0.25, 0.20]
ENTRY_MODES = ["next_open", "next_close"]
DEFAULT_ROOM_LOOKBACK = 50
DEFAULT_MIN_ROOM_R = 1.5
DEFAULT_EMA_PERIOD = 200
DEFAULT_EMA_SLOPE_LOOKBACK = 10
DEFAULT_MAX_EMA_DISTANCE_ATR = 3.0


@dataclass(frozen=True)
class ParamSet:
    n: int
    atr_period: int
    atr_mult: float
    vol_z_min: float
    close_loc_long_min: float
    close_loc_short_max: float
    stop_atr_mult: float = 1.5
    target_r: float = 2.0

    @property
    def param_id(self) -> str:
        return (
            f"n{self.n}_atr{self.atr_period}_m{self.atr_mult:g}_"
            f"vz{self.vol_z_min:g}_cl{self.close_loc_long_min:g}_cs{self.close_loc_short_max:g}"
        )


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class SignalFilters:
    use_room_filter: bool = True
    room_lookback: int = DEFAULT_ROOM_LOOKBACK
    min_room_r: float = DEFAULT_MIN_ROOM_R
    use_trend_filter: bool = True
    ema_period: int = DEFAULT_EMA_PERIOD
    ema_slope_lookback: int = DEFAULT_EMA_SLOPE_LOOKBACK
    use_overextension_filter: bool = True
    max_ema_distance_atr: float = DEFAULT_MAX_EMA_DISTANCE_ATR


def parse_utc(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in factors:
        raise ValueError(f"Unsupported Binance interval: {interval}")
    return amount * factors[unit]


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported interval for timestamp semantics: {interval}")


def download_klines(symbol: str, interval: str, start: datetime, end: datetime, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = interval_to_ms(interval)
    rows: list[list[object]] = []

    cursor = start_ms
    while cursor < end_ms:
        response = requests.get(
            BINANCE_FAPI_KLINES,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
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
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(out_path, index=False)


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.rename(columns={timestamp_col: "timestamp"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[required].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def validate_ohlcv_quality(
    symbol: str,
    timeframe: str,
    path: Path,
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    raw = pd.read_csv(path, usecols=[0])
    raw_timestamps = pd.to_datetime(raw.iloc[:, 0], utc=True, errors="coerce")
    expected_delta = interval_to_timedelta(timeframe)
    diffs = df["timestamp"].diff().dropna()
    unexpected_gaps = diffs[diffs != expected_delta]
    expected_rows = int(((pd.Timestamp(end) - pd.Timestamp(start)) / expected_delta)) + 1
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "path": str(path),
        "rows": int(df.shape[0]),
        "expected_rows": expected_rows,
        "sufficient_data": bool(df.shape[0] >= max(DEFAULT_EMA_PERIOD + DEFAULT_EMA_SLOPE_LOOKBACK, DEFAULT_ROOM_LOOKBACK)),
        "duplicate_timestamps": int(raw_timestamps.duplicated().sum()),
        "missing_timestamps": int(max(expected_rows - df.shape[0], 0)),
        "unexpected_gaps": int(unexpected_gaps.shape[0]),
        "max_gap": str(unexpected_gaps.max()) if not unexpected_gaps.empty else "",
        "first_timestamp": df["timestamp"].min().isoformat(),
        "last_timestamp": df["timestamp"].max().isoformat(),
        "quality_ok": bool(
            raw_timestamps.duplicated().sum() == 0
            and max(expected_rows - df.shape[0], 0) == 0
            and unexpected_gaps.empty
            and df.shape[0] >= max(DEFAULT_EMA_PERIOD + DEFAULT_EMA_SLOPE_LOOKBACK, DEFAULT_ROOM_LOOKBACK)
        ),
    }


def ensure_data(
    symbols: list[str],
    timeframes: list[str],
    data_dir: Path,
    start: datetime,
    end: datetime,
    download: bool,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    for timeframe in timeframes:
        for symbol in symbols:
            path = data_dir / timeframe / f"{symbol}.csv"
            legacy_path = Path("data") / timeframe / f"{symbol}.csv"
            source_path = path if path.exists() else legacy_path

            needs_download = download or not source_path.exists()
            if not needs_download:
                probe = load_ohlcv(source_path)
                has_start = probe["timestamp"].min().to_pydatetime() <= start + timedelta(days=3)
                has_end = probe["timestamp"].max().to_pydatetime() >= end - timedelta(days=3)
                needs_download = not (has_start and has_end)

            if needs_download:
                print(f"Downloading {symbol} {timeframe} to {path}")
                download_klines(symbol, timeframe, start, end, path)
                source_path = path

            df = load_ohlcv(source_path)
            mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"No rows in requested date range for {symbol} {timeframe}")
            quality_rows.append(validate_ohlcv_quality(symbol, timeframe, source_path, df, start, end))
            loaded[(symbol, timeframe)] = df
    return loaded, pd.DataFrame(quality_rows)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    out["true_range"] = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["candle_range"] = out["high"] - out["low"]
    out["close_location"] = (out["close"] - out["low"]) / out["candle_range"].replace(0, np.nan)
    out["atr_14_prior"] = out["true_range"].rolling(14, min_periods=14).mean().shift(1)
    out["ema_200"] = out["close"].ewm(span=DEFAULT_EMA_PERIOD, adjust=False, min_periods=DEFAULT_EMA_PERIOD).mean()
    out["ema_200_slope"] = out["ema_200"] - out["ema_200"].shift(DEFAULT_EMA_SLOPE_LOOKBACK)
    out["room_swing_high"] = out["high"].rolling(DEFAULT_ROOM_LOOKBACK, min_periods=DEFAULT_ROOM_LOOKBACK).max().shift(1)
    out["room_swing_low"] = out["low"].rolling(DEFAULT_ROOM_LOOKBACK, min_periods=DEFAULT_ROOM_LOOKBACK).min().shift(1)
    return out


def add_param_features(df: pd.DataFrame, p: ParamSet, filters: SignalFilters) -> pd.DataFrame:
    out = df.copy()
    if filters.ema_period != DEFAULT_EMA_PERIOD:
        out["ema_200"] = out["close"].ewm(span=filters.ema_period, adjust=False, min_periods=filters.ema_period).mean()
    if filters.ema_slope_lookback != DEFAULT_EMA_SLOPE_LOOKBACK or filters.ema_period != DEFAULT_EMA_PERIOD:
        out["ema_200_slope"] = out["ema_200"] - out["ema_200"].shift(filters.ema_slope_lookback)
    if filters.room_lookback != DEFAULT_ROOM_LOOKBACK:
        out["room_swing_high"] = out["high"].rolling(filters.room_lookback, min_periods=filters.room_lookback).max().shift(1)
        out["room_swing_low"] = out["low"].rolling(filters.room_lookback, min_periods=filters.room_lookback).min().shift(1)

    out["prior_high"] = out["high"].rolling(p.n, min_periods=p.n).max().shift(1)
    out["prior_low"] = out["low"].rolling(p.n, min_periods=p.n).min().shift(1)
    vol_mean = out["volume"].rolling(p.n, min_periods=p.n).mean().shift(1)
    vol_std = out["volume"].rolling(p.n, min_periods=p.n).std(ddof=0).shift(1)
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    required = ["prior_high", "prior_low", "atr_14_prior", "volume_z", "close_location"]
    if filters.use_trend_filter or filters.use_overextension_filter:
        required.extend(["ema_200", "ema_200_slope"])
    if filters.use_room_filter:
        required.extend(["room_swing_high", "room_swing_low"])
    valid = out[required].notna().all(axis=1)
    expansion = out["candle_range"] > p.atr_mult * out["atr_14_prior"]
    volume_ok = out["volume_z"] > p.vol_z_min
    trend_long = (out["close"] > out["ema_200"]) & (out["ema_200_slope"] > 0)
    trend_short = (out["close"] < out["ema_200"]) & (out["ema_200_slope"] < 0)
    if not filters.use_trend_filter:
        trend_long = pd.Series(True, index=out.index)
        trend_short = pd.Series(True, index=out.index)
    ema_distance_atr = (out["close"] - out["ema_200"]).abs() / out["atr_14_prior"].replace(0, np.nan)
    not_overextended = ema_distance_atr <= filters.max_ema_distance_atr
    if not filters.use_overextension_filter:
        not_overextended = pd.Series(True, index=out.index)
    out["long_signal"] = (
        valid
        & (out["close"] > out["prior_high"])
        & expansion
        & volume_ok
        & (out["close_location"] >= p.close_loc_long_min)
        & trend_long
        & not_overextended
    )
    out["short_signal"] = (
        valid
        & (out["close"] < out["prior_low"])
        & expansion
        & volume_ok
        & (out["close_location"] <= p.close_loc_short_max)
        & trend_short
        & not_overextended
    )
    out["ema_distance_atr"] = ema_distance_atr
    return out


def apply_entry_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate) if side == "long" else raw_price * (1.0 - friction.slippage_rate)


def apply_exit_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate) if side == "long" else raw_price * (1.0 + friction.slippage_rate)


def r_multiple(entry: float, exit_price: float, side: str, risk: float) -> float:
    if side == "long":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk


def simulate_symbol(
    symbol: str,
    timeframe: str,
    data: pd.DataFrame,
    p: ParamSet,
    filters: SignalFilters,
    friction: Friction,
    entry_mode: str,
) -> list[dict[str, object]]:
    rows = add_param_features(data, p, filters).reset_index(drop=True)
    max_hold = 12 if timeframe == "1h" else 8
    trades: list[dict[str, object]] = []
    next_signal_index = 0

    timestamps = rows["timestamp"].to_numpy()
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    atrs = rows["atr_14_prior"].to_numpy(dtype=float)
    room_swing_highs = rows["room_swing_high"].to_numpy(dtype=float)
    room_swing_lows = rows["room_swing_low"].to_numpy(dtype=float)
    ema_values = rows["ema_200"].to_numpy(dtype=float)
    ema_slopes = rows["ema_200_slope"].to_numpy(dtype=float)
    ema_distances = rows["ema_distance_atr"].to_numpy(dtype=float)
    long_signals = rows["long_signal"].to_numpy(dtype=bool)
    short_signals = rows["short_signal"].to_numpy(dtype=bool)
    signal_indices = np.flatnonzero(long_signals | short_signals)
    timeframe_delta = interval_to_timedelta(timeframe)

    for signal_i in signal_indices:
        if signal_i < next_signal_index:
            continue
        side = "long" if long_signals[signal_i] else "short"

        atr = float(atrs[signal_i])
        risk = p.stop_atr_mult * atr
        if not math.isfinite(risk) or risk <= 0:
            continue

        signal_close = float(closes[signal_i])
        if side == "long":
            structure = float(room_swing_highs[signal_i])
            available_room = structure - signal_close if math.isfinite(structure) and structure > signal_close else math.nan
        else:
            structure = float(room_swing_lows[signal_i])
            available_room = signal_close - structure if math.isfinite(structure) and structure < signal_close else math.nan
        room_to_target_r = available_room / risk if math.isfinite(available_room) else math.nan
        if filters.use_room_filter and (not math.isfinite(room_to_target_r) or room_to_target_r < filters.min_room_r):
            continue

        entry_i = signal_i + 1
        if entry_i >= len(rows):
            break
        raw_entry = float(opens[entry_i] if entry_mode == "next_open" else closes[entry_i])
        entry = apply_entry_slippage(raw_entry, side, friction)

        if side == "long":
            stop = entry - risk
            target = entry + p.target_r * risk
        else:
            stop = entry + risk
            target = entry - p.target_r * risk

        first_exit_i = entry_i if entry_mode == "next_open" else entry_i + 1
        last_exit_i = min(entry_i + max_hold, len(rows) - 1)
        if first_exit_i > last_exit_i:
            continue

        raw_exit = float(closes[last_exit_i])
        exit_i = last_exit_i
        exit_reason = "time_stop"

        for j in range(first_exit_i, last_exit_i + 1):
            high = float(highs[j])
            low = float(lows[j])
            if side == "long":
                hit_stop = low <= stop
                hit_target = high >= target
            else:
                hit_stop = high >= stop
                hit_target = low <= target
            if hit_stop:
                raw_exit = stop
                exit_i = j
                exit_reason = "stop"
                break
            if hit_target:
                raw_exit = target
                exit_i = j
                exit_reason = "target"
                break

        exit_price = apply_exit_slippage(raw_exit, side, friction)
        gross_r = r_multiple(entry, exit_price, side, risk)
        fee_r = friction.taker_fee_rate * (entry + exit_price) / risk
        net_r = gross_r - fee_r
        trades.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "param_id": p.param_id,
                "friction_case": friction.name,
                "entry_mode": entry_mode,
                "signal_time": timestamps[signal_i],
                "entry_time": (
                    pd.Timestamp(timestamps[entry_i])
                    if entry_mode == "next_open"
                    else pd.Timestamp(timestamps[entry_i]) + timeframe_delta
                ),
                "entry_time_semantic": "next_candle_open" if entry_mode == "next_open" else "next_candle_close",
                "exit_time": timestamps[exit_i],
                "side": side,
                "raw_entry": raw_entry,
                "entry": entry,
                "raw_exit": raw_exit,
                "exit": exit_price,
                "stop": stop,
                "target": target,
                "atr_at_signal": atr,
                "risk": risk,
                "room_structure": structure,
                "room_reference_price": signal_close,
                "room_to_target_r": room_to_target_r,
                "available_room_r": room_to_target_r,
                "ema_200_at_signal": float(ema_values[signal_i]),
                "ema_200_slope_at_signal": float(ema_slopes[signal_i]),
                "ema_distance_atr_at_signal": float(ema_distances[signal_i]),
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "exit_reason": exit_reason,
                "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                "slippage_bps_per_side": friction.slippage_rate * 10_000,
                "n": p.n,
                "atr_period": p.atr_period,
                "atr_mult": p.atr_mult,
                "vol_z_min": p.vol_z_min,
                "close_loc_long_min": p.close_loc_long_min,
                "close_loc_short_max": p.close_loc_short_max,
                "stop_atr_mult": p.stop_atr_mult,
                "target_r": p.target_r,
                "max_hold_bars": max_hold,
                "use_room_filter": filters.use_room_filter,
                "room_lookback": filters.room_lookback,
                "min_room_r": filters.min_room_r,
                "use_trend_filter": filters.use_trend_filter,
                "ema_period": filters.ema_period,
                "ema_slope_lookback": filters.ema_slope_lookback,
                "use_overextension_filter": filters.use_overextension_filter,
                "max_ema_distance_atr": filters.max_ema_distance_atr,
            }
        )
        next_signal_index = exit_i + 1
    return trades


def max_drawdown_from_returns(values: Iterable[float]) -> float:
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


def profit_factor(values: pd.Series) -> float:
    wins = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    if losses == 0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def summarize_group(group: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_r"]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    start = ordered["entry_time"].min()
    end = ordered["exit_time"].max()
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "total_trades": int(len(ordered)),
            "trades_per_year": float(len(ordered) / years),
            "win_rate": float((returns > 0).mean()),
            "ev_per_trade_r": float(returns.mean()),
            "total_ev_r": float(returns.sum()),
            "max_drawdown_r": max_drawdown_from_returns(returns),
            "max_losing_streak": max_losing_streak(returns),
            "profit_factor": profit_factor(returns),
            "average_win_r": float(wins.mean()) if not wins.empty else 0.0,
            "average_loss_r": float(losses.mean()) if not losses.empty else 0.0,
        }
    )
    return out


def summarize_empty(group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "total_trades",
        "trades_per_year",
        "win_rate",
        "ev_per_trade_r",
        "total_ev_r",
        "max_drawdown_r",
        "max_losing_streak",
        "profit_factor",
        "average_win_r",
        "average_loss_r",
    ]
    return pd.DataFrame(columns=group_cols + metric_cols)


def fast_group_metrics(df: pd.DataFrame, group_cols: list[str], include_path_metrics: bool) -> pd.DataFrame:
    if df.empty:
        return summarize_empty(group_cols)

    work = df.copy()
    work["wins_r"] = work["net_r"].where(work["net_r"] > 0, 0.0)
    work["losses_r"] = work["net_r"].where(work["net_r"] < 0, 0.0)
    work["win_count"] = (work["net_r"] > 0).astype(int)
    work["loss_count"] = (work["net_r"] < 0).astype(int)
    grouped = work.groupby(group_cols, sort=False)
    out = grouped.agg(
        total_trades=("net_r", "size"),
        trades_start=("entry_time", "min"),
        trades_end=("exit_time", "max"),
        win_count=("win_count", "sum"),
        loss_count=("loss_count", "sum"),
        ev_per_trade_r=("net_r", "mean"),
        total_ev_r=("net_r", "sum"),
        gross_wins_r=("wins_r", "sum"),
        gross_losses_r=("losses_r", "sum"),
    ).reset_index()
    years = (out["trades_end"] - out["trades_start"]).dt.total_seconds() / (365.25 * 24 * 3600)
    years = years.clip(lower=1 / 365.25)
    out["trades_per_year"] = out["total_trades"] / years
    out["win_rate"] = out["win_count"] / out["total_trades"]
    out["profit_factor"] = np.where(
        out["gross_losses_r"] < 0,
        out["gross_wins_r"] / out["gross_losses_r"].abs(),
        np.where(out["gross_wins_r"] > 0, np.inf, 0.0),
    )
    out["average_win_r"] = np.where(out["win_count"] > 0, out["gross_wins_r"] / out["win_count"], 0.0)
    out["average_loss_r"] = np.where(out["loss_count"] > 0, out["gross_losses_r"] / out["loss_count"], 0.0)
    if include_path_metrics:
        path_rows = []
        sorted_work = work.sort_values(group_cols + ["exit_time", "symbol"])
        for key, group in sorted_work.groupby(group_cols, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            values = group["net_r"].to_numpy(dtype=float)
            path_rows.append(
                {
                    **dict(zip(group_cols, key_tuple)),
                    "max_drawdown_r": max_drawdown_from_returns(values),
                    "max_losing_streak": max_losing_streak(values),
                }
            )
        path = pd.DataFrame(path_rows)
        out = out.merge(path, on=group_cols, how="left")
    else:
        out["max_drawdown_r"] = np.nan
        out["max_losing_streak"] = np.nan

    metric_order = [
        "total_trades",
        "trades_per_year",
        "win_rate",
        "ev_per_trade_r",
        "total_ev_r",
        "max_drawdown_r",
        "max_losing_streak",
        "profit_factor",
        "average_win_r",
        "average_loss_r",
    ]
    return out[group_cols + metric_order]


def build_summaries(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "timeframe",
        "param_id",
        "friction_case",
        "entry_mode",
        "n",
        "atr_period",
        "atr_mult",
        "vol_z_min",
        "close_loc_long_min",
        "close_loc_short_max",
        "taker_fee_bps_per_side",
        "slippage_bps_per_side",
        "max_hold_bars",
        "use_room_filter",
        "room_lookback",
        "min_room_r",
        "use_trend_filter",
        "ema_period",
        "ema_slope_lookback",
        "use_overextension_filter",
        "max_ema_distance_atr",
    ]
    if trades.empty:
        empty = summarize_empty(scenario_cols)
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall = fast_group_metrics(trades, scenario_cols, include_path_metrics=True).sort_values(
        ["timeframe", "friction_case", "entry_mode", "ev_per_trade_r"],
        ascending=[True, True, True, False],
    )

    by_symbol_cols = scenario_cols + ["symbol"]
    by_symbol = fast_group_metrics(trades, by_symbol_cols, include_path_metrics=False).sort_values(
        ["timeframe", "friction_case", "entry_mode", "param_id", "symbol"]
    )

    by_month_source = trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str))
    by_month_cols = scenario_cols + ["month"]
    by_month = fast_group_metrics(by_month_source, by_month_cols, include_path_metrics=False).sort_values(
        ["timeframe", "friction_case", "entry_mode", "param_id", "month"]
    )

    pivot_keys = [
        "timeframe",
        "param_id",
        "friction_case",
        "n",
        "atr_period",
        "atr_mult",
        "vol_z_min",
        "close_loc_long_min",
        "close_loc_short_max",
        "use_room_filter",
        "room_lookback",
        "min_room_r",
        "use_trend_filter",
        "ema_period",
        "ema_slope_lookback",
        "use_overextension_filter",
        "max_ema_distance_atr",
    ]
    open_rows = overall[overall["entry_mode"] == "next_open"].set_index(pivot_keys)
    close_rows = overall[overall["entry_mode"] == "next_close"].set_index(pivot_keys)
    comparison = open_rows[["total_trades", "ev_per_trade_r", "total_ev_r", "max_drawdown_r", "profit_factor"]].join(
        close_rows[["total_trades", "ev_per_trade_r", "total_ev_r", "max_drawdown_r", "profit_factor"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["ev_delta_close_minus_open_r"] = (
        comparison["ev_per_trade_r_next_close"] - comparison["ev_per_trade_r_next_open"]
    )
    comparison["trade_delta_close_minus_open"] = (
        comparison["total_trades_next_close"] - comparison["total_trades_next_open"]
    )
    comparison = comparison.sort_values(["timeframe", "friction_case", "ev_delta_close_minus_open_r"], ascending=[True, True, False])
    return overall, by_symbol, by_month, comparison


def param_grid() -> list[ParamSet]:
    return [
        ParamSet(n=n, atr_period=14, atr_mult=atr_mult, vol_z_min=vol_z_min, close_loc_long_min=cl_long, close_loc_short_max=cl_short)
        for n in LOOKBACKS
        for atr_mult in ATR_MULTS
        for vol_z_min in VOL_Z_MINS
        for cl_long, cl_short in zip(CLOSE_LOC_LONG_MINS, CLOSE_LOC_SHORT_MAXS)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1-2 OHLCV-only volatility expansion backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "phase_1_2")
    parser.add_argument("--start", required=True, help="Explicit UTC start timestamp/date, e.g. 2025-05-05T00:00:00Z.")
    parser.add_argument("--end", required=True, help="Explicit UTC end timestamp/date, e.g. 2026-05-05T00:00:00Z.")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--download", action="store_true", help="Force fresh Binance USD-M futures OHLCV download.")
    parser.add_argument("--disable-room-filter", action="store_true")
    parser.add_argument("--room-lookback", type=int, default=DEFAULT_ROOM_LOOKBACK)
    parser.add_argument("--min-room-r", type=float, default=DEFAULT_MIN_ROOM_R)
    parser.add_argument("--disable-trend-filter", action="store_true")
    parser.add_argument("--ema-period", type=int, default=DEFAULT_EMA_PERIOD)
    parser.add_argument("--ema-slope-lookback", type=int, default=DEFAULT_EMA_SLOPE_LOOKBACK)
    parser.add_argument("--disable-overextension-filter", action="store_true")
    parser.add_argument("--max-ema-distance-atr", type=float, default=DEFAULT_MAX_EMA_DISTANCE_ATR)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [symbol.upper() for symbol in args.symbols]
    timeframes = [timeframe.lower() for timeframe in args.timeframes]
    frictions = [
        Friction(name="base", taker_fee_rate=0.0004, slippage_rate=0.0005),
        Friction(name="stress", taker_fee_rate=0.0008, slippage_rate=0.0010),
    ]
    filters = SignalFilters(
        use_room_filter=not args.disable_room_filter,
        room_lookback=args.room_lookback,
        min_room_r=args.min_room_r,
        use_trend_filter=not args.disable_trend_filter,
        ema_period=args.ema_period,
        ema_slope_lookback=args.ema_slope_lookback,
        use_overextension_filter=not args.disable_overextension_filter,
        max_ema_distance_atr=args.max_ema_distance_atr,
    )

    loaded, data_quality = ensure_data(symbols, timeframes, args.data_dir, start, end, args.download)
    featured = {key: add_base_features(df) for key, df in loaded.items()}
    params = param_grid()

    records: list[dict[str, object]] = []
    for timeframe in timeframes:
        for p in params:
            for friction in frictions:
                for entry_mode in ENTRY_MODES:
                    for symbol in symbols:
                        records.extend(
                            simulate_symbol(
                                symbol=symbol,
                                timeframe=timeframe,
                                data=featured[(symbol, timeframe)],
                                p=p,
                                filters=filters,
                                friction=friction,
                                entry_mode=entry_mode,
                            )
                        )

    trades = pd.DataFrame(records)
    if not trades.empty:
        for col in ["signal_time", "entry_time", "exit_time"]:
            trades[col] = pd.to_datetime(trades[col], utc=True)
        trades = trades.sort_values(
            ["timeframe", "friction_case", "entry_mode", "param_id", "exit_time", "symbol"]
        ).reset_index(drop=True)

    overall, by_symbol, by_month, execution_comparison = build_summaries(trades)

    trades_path = args.out_dir / "trades.csv"
    trades_sample_path = args.out_dir / "trades_sample.csv"
    overall_path = args.out_dir / "overall_results.csv"
    by_symbol_path = args.out_dir / "by_symbol.csv"
    by_month_path = args.out_dir / "by_month.csv"
    execution_path = args.out_dir / "execution_comparison.csv"
    data_quality_path = args.out_dir / "data_quality_report.csv"
    summary_path = args.out_dir / "phase_1_2_summary.json"

    trades.to_csv(trades_path, index=False)
    trades.head(1000).to_csv(trades_sample_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_month.to_csv(by_month_path, index=False)
    execution_comparison.to_csv(execution_path, index=False)
    data_quality.to_csv(data_quality_path, index=False)

    top_overall = overall.sort_values("ev_per_trade_r", ascending=False).head(10).to_dict(orient="records")
    baseline_mask = (
        (overall["friction_case"] == "base")
        & (overall["entry_mode"] == "next_open")
        & (overall["n"] == 20)
        & (overall["atr_mult"] == 1.2)
        & (overall["vol_z_min"] == 1.0)
        & (overall["close_loc_long_min"] == 0.70)
        & (overall["close_loc_short_max"] == 0.30)
    )
    baseline = overall.loc[baseline_mask].sort_values("timeframe").to_dict(orient="records")
    data_ranges = {
        f"{symbol}_{timeframe}": {
            "rows": int(df.shape[0]),
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
        }
        for (symbol, timeframe), df in loaded.items()
    }
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "OHLCV-only Phase 1-2 backtest for Executable Volatility Expansion Bot with Crowd-Avoidance Filters.",
        "symbols": symbols,
        "timeframes": timeframes,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "parameter_count": len(params),
        "scenario_count": int(overall.shape[0]),
        "total_trade_rows": int(trades.shape[0]),
        "friction": [asdict(friction) for friction in frictions],
        "signal_filters": asdict(filters),
        "execution_assumptions": [
            "Signals are evaluated only after the signal candle close.",
            "Breakout, ATR, and volume z-score reference previous candles only.",
            "Volume z-score uses the same prior N-window as the breakout lookback.",
            "Room filter eligibility is decided at signal time using signal close and prior-data structure only.",
            "Signals with no measurable prior-data structure ahead are rejected by the room filter.",
            "Trend filter requires close/EMA200 alignment and EMA200 slope alignment.",
            "Overextension filter rejects signals beyond the configured ATR distance from EMA.",
            "next_open entries execute at the next candle open timestamp; next_close entries execute at the next candle close and entry_time is shifted to close semantics.",
            "SL/TP are evaluated with OHLC after entry; if both are touched in one candle, SL wins.",
            "SL/TP exits are modeled at the trigger level with adverse slippage and taker fees.",
            "Only one trade per symbol/scenario is active at a time.",
        ],
        "reports": {
            "trades_csv": str(trades_path),
            "trades_sample_csv": str(trades_sample_path),
            "overall_results_csv": str(overall_path),
            "by_symbol_csv": str(by_symbol_path),
            "by_month_csv": str(by_month_path),
            "execution_comparison_csv": str(execution_path),
            "data_quality_report_csv": str(data_quality_path),
            "phase_1_2_summary_json": str(summary_path),
        },
        "data_ranges": data_ranges,
        "data_quality": data_quality.to_dict(orient="records"),
        "baseline_base_next_open": baseline,
        "top_10_by_ev_per_trade": top_overall,
    }
    write_json(summary_path, summary)

    print(f"Wrote Phase 1-2 reports to {args.out_dir.resolve()}")
    if not data_quality["quality_ok"].all():
        print("WARNING: Data quality report contains failures. Review data_quality_report.csv before trusting results.")
    if not overall.empty:
        print(overall.sort_values("ev_per_trade_r", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
