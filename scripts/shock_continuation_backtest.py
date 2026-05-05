from __future__ import annotations

import argparse
import json
import math
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
TIMEFRAMES = ["15m", "1h"]
RANGE_MULTS = [1.5, 2.0, 2.5, 3.0]
VOL_Z_MINS = [1.5, 2.0, 2.5]
CLOSE_LOC_PAIRS = [(0.70, 0.30), (0.80, 0.20)]
BODY_MINS = [0.50, 0.65]
ENTRY_MODES = ["next_open", "next_close"]
ATR_PERIOD = 14
VOLUME_Z_LOOKBACK = 50
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"


@dataclass(frozen=True)
class ShockParams:
    range_mult: float
    vol_z_min: float
    close_loc_long_min: float
    close_loc_short_max: float
    body_min: float
    ret_mult: float
    atr_period: int = ATR_PERIOD
    volume_z_lookback: int = VOLUME_Z_LOOKBACK
    require_direction: bool = True

    @property
    def param_id(self) -> str:
        return (
            f"range{self.range_mult:g}_ret{self.ret_mult:g}_vz{self.vol_z_min:g}_"
            f"cl{self.close_loc_long_min:g}_cs{self.close_loc_short_max:g}_body{self.body_min:g}"
        )


@dataclass(frozen=True)
class ExitModel:
    name: str
    stop_atr_mult: float
    target_r: float
    max_hold_15m: int
    max_hold_1h: int

    def max_hold(self, timeframe: str) -> int:
        if timeframe == "15m":
            return self.max_hold_15m
        if timeframe == "1h":
            return self.max_hold_1h
        raise ValueError(f"Unsupported timeframe for exit model: {timeframe}")


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


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


def interval_to_pandas_freq(interval: str) -> str:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return f"{amount}min"
    if unit == "h":
        return f"{amount}h"
    if unit == "d":
        return f"{amount}D"
    raise ValueError(f"Unsupported interval: {interval}")


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
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
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


def data_quality(symbol: str, timeframe: str, df: pd.DataFrame, start: datetime, end: datetime) -> dict[str, object]:
    expected = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=interval_to_pandas_freq(timeframe), inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    missing = expected.difference(timestamps)
    duplicates = int(df["timestamp"].duplicated().sum())
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(milliseconds=interval_to_ms(timeframe))
    large_gaps = gaps[gaps > expected_delta]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
        "duplicate_rows": duplicates,
        "first_timestamp": df["timestamp"].min().isoformat() if not df.empty else None,
        "last_timestamp": df["timestamp"].max().isoformat() if not df.empty else None,
        "large_gap_count": int(len(large_gaps)),
        "max_gap_minutes": float(gaps.max().total_seconds() / 60.0) if not gaps.empty else 0.0,
        "zero_volume_rows": int((df["volume"] <= 0).sum()),
        "bad_ohlc_rows": int(((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))).sum()),
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
            needs_download = download or not path.exists()
            if not needs_download:
                probe = load_ohlcv(path)
                has_start = probe["timestamp"].min().to_pydatetime() <= start
                has_end = probe["timestamp"].max().to_pydatetime() >= end - pd.Timedelta(milliseconds=interval_to_ms(timeframe))
                needs_download = not (has_start and has_end)
            if needs_download:
                print(f"Downloading {symbol} {timeframe} to {path}")
                download_klines(symbol, timeframe, start, end, path)
            df = load_ohlcv(path)
            mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end))
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"No rows in requested range for {symbol} {timeframe}")
            loaded[(symbol, timeframe)] = df
            quality_rows.append(data_quality(symbol, timeframe, df, start, end))
    return loaded, pd.DataFrame(quality_rows)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = tr
    out["atr_14_prior"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().shift(1)
    out["atr_pct_prior"] = out["atr_14_prior"] / prev_close.replace(0, np.nan)
    out["candle_range"] = out["high"] - out["low"]
    out["close_location"] = (out["close"] - out["low"]) / out["candle_range"].replace(0, np.nan)
    out["body_pct"] = (out["close"] - out["open"]).abs() / out["candle_range"].replace(0, np.nan)
    out["candle_return"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    vol_mean = out["volume"].rolling(VOLUME_Z_LOOKBACK, min_periods=VOLUME_Z_LOOKBACK).mean().shift(1)
    vol_std = out["volume"].rolling(VOLUME_Z_LOOKBACK, min_periods=VOLUME_Z_LOOKBACK).std(ddof=0).shift(1)
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return out


def add_signals(df: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
    out = df.copy()
    valid = out[["atr_14_prior", "atr_pct_prior", "volume_z", "close_location", "body_pct", "candle_return"]].notna().all(axis=1)
    range_shock = out["candle_range"] > params.range_mult * out["atr_14_prior"]
    long_return_shock = out["candle_return"] > params.ret_mult * out["atr_pct_prior"]
    short_return_shock = out["candle_return"] < -params.ret_mult * out["atr_pct_prior"]
    volume_ok = out["volume_z"] > params.vol_z_min
    body_ok = out["body_pct"] >= params.body_min
    bullish_ok = out["close"] > out["open"] if params.require_direction else pd.Series(True, index=out.index)
    bearish_ok = out["close"] < out["open"] if params.require_direction else pd.Series(True, index=out.index)
    out["long_signal"] = (
        valid
        & (long_return_shock | range_shock)
        & volume_ok
        & (out["close_location"] >= params.close_loc_long_min)
        & body_ok
        & bullish_ok
    )
    out["short_signal"] = (
        valid
        & (short_return_shock | range_shock)
        & volume_ok
        & (out["close_location"] <= params.close_loc_short_max)
        & body_ok
        & bearish_ok
    )
    return out


def apply_entry_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate) if side == "long" else raw_price * (1.0 - friction.slippage_rate)


def apply_exit_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate) if side == "long" else raw_price * (1.0 + friction.slippage_rate)


def r_multiple(entry: float, exit_price: float, side: str, risk: float) -> float:
    return (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk


def simulate_symbol(
    symbol: str,
    timeframe: str,
    featured: pd.DataFrame,
    params: ShockParams,
    exit_model: ExitModel,
    friction: Friction,
    entry_mode: str,
) -> list[dict[str, object]]:
    rows = add_signals(featured, params).reset_index(drop=True)
    max_hold = exit_model.max_hold(timeframe)
    timestamps = rows["timestamp"].to_numpy()
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    atrs = rows["atr_14_prior"].to_numpy(dtype=float)
    volume_z = rows["volume_z"].to_numpy(dtype=float)
    candle_return = rows["candle_return"].to_numpy(dtype=float)
    body_pct = rows["body_pct"].to_numpy(dtype=float)
    close_location = rows["close_location"].to_numpy(dtype=float)
    long_signals = rows["long_signal"].to_numpy(dtype=bool)
    short_signals = rows["short_signal"].to_numpy(dtype=bool)
    signal_indices = np.flatnonzero(long_signals | short_signals)
    trades: list[dict[str, object]] = []
    next_signal_index = 0

    for signal_i in signal_indices:
        if signal_i < next_signal_index:
            continue
        side = "long" if long_signals[signal_i] else "short"
        entry_i = signal_i + 1
        if entry_i >= len(rows):
            break
        raw_entry = float(opens[entry_i] if entry_mode == "next_open" else closes[entry_i])
        entry = apply_entry_slippage(raw_entry, side, friction)
        risk = exit_model.stop_atr_mult * float(atrs[signal_i])
        if not math.isfinite(risk) or risk <= 0:
            continue
        if side == "long":
            stop = entry - risk
            target = entry + exit_model.target_r * risk
        else:
            stop = entry + risk
            target = entry - exit_model.target_r * risk

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
                "param_id": params.param_id,
                "exit_model": exit_model.name,
                "friction_case": friction.name,
                "entry_mode": entry_mode,
                "signal_time": timestamps[signal_i],
                "entry_time": timestamps[entry_i],
                "exit_time": timestamps[exit_i],
                "side": side,
                "raw_entry": raw_entry,
                "entry": entry,
                "raw_exit": raw_exit,
                "exit": exit_price,
                "stop": stop,
                "target": target,
                "atr_at_signal": float(atrs[signal_i]),
                "risk": risk,
                "volume_z": float(volume_z[signal_i]),
                "candle_return": float(candle_return[signal_i]),
                "body_pct": float(body_pct[signal_i]),
                "close_location": float(close_location[signal_i]),
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "exit_reason": exit_reason,
                "range_mult": params.range_mult,
                "ret_mult": params.ret_mult,
                "vol_z_min": params.vol_z_min,
                "close_loc_long_min": params.close_loc_long_min,
                "close_loc_short_max": params.close_loc_short_max,
                "body_min": params.body_min,
                "atr_period": params.atr_period,
                "volume_z_lookback": params.volume_z_lookback,
                "require_direction": params.require_direction,
                "stop_atr_mult": exit_model.stop_atr_mult,
                "target_r": exit_model.target_r,
                "max_hold_bars": max_hold,
                "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                "slippage_bps_per_side": friction.slippage_rate * 10_000,
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


def fast_group_metrics(df: pd.DataFrame, group_cols: list[str], include_path_metrics: bool) -> pd.DataFrame:
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
    if df.empty:
        return pd.DataFrame(columns=group_cols + metric_cols)
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
        out = out.merge(pd.DataFrame(path_rows), on=group_cols, how="left")
    else:
        out["max_drawdown_r"] = np.nan
        out["max_losing_streak"] = np.nan
    return out[group_cols + metric_cols]


def contribution_shares(trades: pd.DataFrame, scenario_cols: list[str], dimension: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=scenario_cols + [f"best_{dimension}_ev_share", f"best_{dimension}_total_ev_r"])
    dim_ev = trades.groupby(scenario_cols + [dimension], sort=False)["net_r"].sum().reset_index()
    total_ev = trades.groupby(scenario_cols, sort=False)["net_r"].sum().reset_index(name="scenario_total_ev_r")
    dim_ev = dim_ev.merge(total_ev, on=scenario_cols, how="left")
    dim_ev["positive_dim_ev_r"] = dim_ev["net_r"].clip(lower=0.0)
    best = dim_ev.sort_values("positive_dim_ev_r", ascending=False).groupby(scenario_cols, sort=False).head(1)
    best = best.rename(columns={"net_r": f"best_{dimension}_total_ev_r"})
    best[f"best_{dimension}_ev_share"] = np.where(
        best["scenario_total_ev_r"] > 0,
        best["positive_dim_ev_r"] / best["scenario_total_ev_r"],
        np.nan,
    )
    return best[scenario_cols + [f"best_{dimension}_ev_share", f"best_{dimension}_total_ev_r"]]


def build_summaries(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "timeframe",
        "param_id",
        "exit_model",
        "friction_case",
        "entry_mode",
        "range_mult",
        "ret_mult",
        "vol_z_min",
        "close_loc_long_min",
        "close_loc_short_max",
        "body_min",
        "require_direction",
        "stop_atr_mult",
        "target_r",
        "max_hold_bars",
        "taker_fee_bps_per_side",
        "slippage_bps_per_side",
    ]
    if trades.empty:
        empty = fast_group_metrics(trades, scenario_cols, include_path_metrics=True)
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall = fast_group_metrics(trades, scenario_cols, include_path_metrics=True)
    symbol_contrib = contribution_shares(trades, scenario_cols, "symbol")
    month_trades = trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str))
    month_contrib = contribution_shares(month_trades, scenario_cols, "month")
    overall = overall.merge(symbol_contrib, on=scenario_cols, how="left").merge(month_contrib, on=scenario_cols, how="left")
    overall = overall.sort_values(["timeframe", "exit_model", "friction_case", "entry_mode", "ev_per_trade_r"], ascending=[True, True, True, True, False])

    by_symbol = fast_group_metrics(trades, scenario_cols + ["symbol"], include_path_metrics=False).sort_values(
        ["timeframe", "exit_model", "friction_case", "entry_mode", "param_id", "symbol"]
    )
    by_month = fast_group_metrics(month_trades, scenario_cols + ["month"], include_path_metrics=False).sort_values(
        ["timeframe", "exit_model", "friction_case", "entry_mode", "param_id", "month"]
    )

    pivot_keys = [
        "timeframe",
        "param_id",
        "exit_model",
        "friction_case",
        "range_mult",
        "ret_mult",
        "vol_z_min",
        "close_loc_long_min",
        "close_loc_short_max",
        "body_min",
        "require_direction",
        "stop_atr_mult",
        "target_r",
    ]
    open_rows = overall[overall["entry_mode"] == "next_open"].set_index(pivot_keys)
    close_rows = overall[overall["entry_mode"] == "next_close"].set_index(pivot_keys)
    comparison = open_rows[["total_trades", "ev_per_trade_r", "total_ev_r", "max_drawdown_r", "profit_factor"]].join(
        close_rows[["total_trades", "ev_per_trade_r", "total_ev_r", "max_drawdown_r", "profit_factor"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["ev_delta_close_minus_open_r"] = comparison["ev_per_trade_r_next_close"] - comparison["ev_per_trade_r_next_open"]
    comparison["trade_delta_close_minus_open"] = comparison["total_trades_next_close"] - comparison["total_trades_next_open"]
    return overall, by_symbol, by_month, comparison


def param_grid(require_direction: bool) -> list[ShockParams]:
    params = []
    for range_mult in RANGE_MULTS:
        for vol_z_min in VOL_Z_MINS:
            for close_long, close_short in CLOSE_LOC_PAIRS:
                for body_min in BODY_MINS:
                    params.append(
                        ShockParams(
                            range_mult=range_mult,
                            ret_mult=range_mult,
                            vol_z_min=vol_z_min,
                            close_loc_long_min=close_long,
                            close_loc_short_max=close_short,
                            body_min=body_min,
                            require_direction=require_direction,
                        )
                    )
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shock continuation OHLCV-only Phase 1-2 backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "shock_continuation")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-opposite-candle-direction", action="store_true")
    parser.add_argument("--trades-sample-size", type=int, default=5000)
    return parser.parse_args()


def pass_fail_summary(overall: pd.DataFrame) -> dict[str, object]:
    if overall.empty:
        return {"overall_pass": False, "reason": "no trades"}
    base = overall[overall["friction_case"] == "base"].copy()
    stress = overall[overall["friction_case"] == "stress"].copy()
    if base.empty:
        return {"overall_pass": False, "reason": "no base scenarios"}
    best = base.sort_values("ev_per_trade_r", ascending=False).iloc[0]
    key_cols = ["timeframe", "param_id", "exit_model", "entry_mode"]
    key = {col: best[col] for col in key_cols}
    stress_match = stress
    for col, value in key.items():
        stress_match = stress_match[stress_match[col] == value]
    stress_ev = float(stress_match["ev_per_trade_r"].iloc[0]) if not stress_match.empty else np.nan
    if best["entry_mode"] == "next_open":
        close_match = base[
            (base["timeframe"] == best["timeframe"])
            & (base["param_id"] == best["param_id"])
            & (base["exit_model"] == best["exit_model"])
            & (base["entry_mode"] == "next_close")
        ]
        next_close_ev = float(close_match["ev_per_trade_r"].iloc[0]) if not close_match.empty else np.nan
    else:
        next_close_ev = float(best["ev_per_trade_r"])
    symbol_share = float(best["best_symbol_ev_share"]) if pd.notna(best["best_symbol_ev_share"]) else np.nan
    month_share = float(best["best_month_ev_share"]) if pd.notna(best["best_month_ev_share"]) else np.nan
    checks = {
        "ev_trade_ge_0_3r_base": float(best["ev_per_trade_r"]) >= 0.3,
        "trades_year_ge_100": float(best["trades_per_year"]) >= 100,
        "stress_ev_not_below_0": pd.notna(stress_ev) and stress_ev >= 0,
        "next_close_not_destroyed": pd.notna(next_close_ev) and next_close_ev >= 0,
        "symbol_concentration_ok": pd.notna(symbol_share) and symbol_share <= 0.40,
        "month_concentration_ok": pd.notna(month_share) and month_share <= 0.40,
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "best_base_scenario": best.to_dict(),
        "matched_stress_ev_per_trade_r": stress_ev,
        "matched_next_close_ev_per_trade_r": next_close_ev,
        "checks": checks,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    symbols = [symbol.upper() for symbol in args.symbols]
    timeframes = [timeframe.lower() for timeframe in args.timeframes]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded, quality = ensure_data(symbols, timeframes, args.data_dir, start, end, args.download)
    featured = {key: add_features(df) for key, df in loaded.items()}
    params = param_grid(require_direction=not args.allow_opposite_candle_direction)
    exit_models = [
        ExitModel("model_a", stop_atr_mult=1.0, target_r=1.5, max_hold_15m=8, max_hold_1h=6),
        ExitModel("model_b", stop_atr_mult=1.2, target_r=2.0, max_hold_15m=12, max_hold_1h=8),
    ]
    frictions = [
        Friction("base", taker_fee_rate=0.0004, slippage_rate=0.0005),
        Friction("stress", taker_fee_rate=0.0008, slippage_rate=0.0010),
    ]

    records: list[dict[str, object]] = []
    for timeframe in timeframes:
        for params_item in params:
            for exit_model in exit_models:
                for friction in frictions:
                    for entry_mode in ENTRY_MODES:
                        for symbol in symbols:
                            records.extend(
                                simulate_symbol(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    featured=featured[(symbol, timeframe)],
                                    params=params_item,
                                    exit_model=exit_model,
                                    friction=friction,
                                    entry_mode=entry_mode,
                                )
                            )

    trades = pd.DataFrame(records)
    if not trades.empty:
        for col in ["signal_time", "entry_time", "exit_time"]:
            trades[col] = pd.to_datetime(trades[col], utc=True)
        trades = trades.sort_values(["timeframe", "exit_model", "friction_case", "entry_mode", "param_id", "exit_time", "symbol"]).reset_index(drop=True)

    overall, by_symbol, by_month, execution_comparison = build_summaries(trades)
    pass_fail = pass_fail_summary(overall)

    trades_path = args.out_dir / "trades.csv"
    trades_sample_path = args.out_dir / "trades_sample.csv"
    overall_path = args.out_dir / "overall_results.csv"
    by_symbol_path = args.out_dir / "by_symbol.csv"
    by_month_path = args.out_dir / "by_month.csv"
    execution_path = args.out_dir / "execution_comparison.csv"
    quality_path = args.out_dir / "data_quality_report.csv"
    summary_path = args.out_dir / "phase_1_2_summary.json"

    trades.to_csv(trades_path, index=False)
    trades.head(args.trades_sample_size).to_csv(trades_sample_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_month.to_csv(by_month_path, index=False)
    execution_comparison.to_csv(execution_path, index=False)
    quality.to_csv(quality_path, index=False)

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
        "scope": "Shock / liquidation-style continuation OHLCV-only Phase 1-2 feasibility test.",
        "symbols": symbols,
        "timeframes": timeframes,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "parameter_count": len(params),
        "scenario_count": int(overall.shape[0]),
        "total_trade_rows": int(trades.shape[0]),
        "exit_models": [asdict(exit_model) for exit_model in exit_models],
        "friction": [asdict(friction) for friction in frictions],
        "assumptions": [
            "No ML, open interest, funding, liquidation feed, limit orders, or intrabar entries.",
            "RET_MULT is tied to RANGE_MULT to keep the requested parameter grid unchanged.",
            "Close-location long/short thresholds are tested as paired values: 0.70/0.30 and 0.80/0.20.",
            "Candle direction is required by default and can be disabled with --allow-opposite-candle-direction.",
            "Signals are known only after shock candle close; entries occur at next candle open or next candle close.",
            "If SL and TP are both touched in the same candle, SL is assumed first.",
        ],
        "reports": {
            "trades_csv": str(trades_path),
            "trades_sample_csv": str(trades_sample_path),
            "overall_results_csv": str(overall_path),
            "by_symbol_csv": str(by_symbol_path),
            "by_month_csv": str(by_month_path),
            "execution_comparison_csv": str(execution_path),
            "data_quality_report_csv": str(quality_path),
            "phase_1_2_summary_json": str(summary_path),
        },
        "data_ranges": data_ranges,
        "top_10_by_ev_per_trade": overall.sort_values("ev_per_trade_r", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    write_json(summary_path, summary)

    print(f"Wrote shock continuation reports to {args.out_dir.resolve()}")
    if not overall.empty:
        print(overall.sort_values("ev_per_trade_r", ascending=False).head(12).to_string(index=False))
    print(json.dumps(pass_fail.get("checks", pass_fail), indent=2, default=str))


if __name__ == "__main__":
    main()
