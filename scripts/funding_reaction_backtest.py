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
BINANCE_FAPI_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
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
ENTRY_MODES = ["next_open", "next_close"]
ATR_PERIOD = 14
FUNDING_Z_LOOKBACK = 90
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
FUNDING_Z_MINS = [1.5, 2.0, 2.5]
REACTION_ATR_MINS = [0.0, 0.25]
BODY_MINS = [0.25, 0.50]
CLOSE_LOC_PAIRS = [(0.60, 0.40), (0.70, 0.30)]


@dataclass(frozen=True)
class FundingParams:
    funding_z_min: float
    reaction_atr_min: float
    body_min: float
    close_loc_long_min: float
    close_loc_short_max: float
    atr_period: int = ATR_PERIOD
    funding_z_lookback: int = FUNDING_Z_LOOKBACK

    @property
    def param_id(self) -> str:
        return (
            f"fz{self.funding_z_min:g}_react{self.reaction_atr_min:g}_"
            f"body{self.body_min:g}_cl{self.close_loc_long_min:g}_cs{self.close_loc_short_max:g}"
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
        raise ValueError(f"Unsupported timeframe: {timeframe}")


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
        raise ValueError(f"Unsupported interval: {interval}")
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


def download_funding(symbol: str, start: datetime, end: datetime, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict[str, object]] = []
    while cursor < end_ms:
        response = requests.get(
            BINANCE_FAPI_FUNDING,
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"No funding rows returned for {symbol}")
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="raise")
    df = df[["funding_time", "funding_rate"]].dropna().drop_duplicates("funding_time").sort_values("funding_time")
    df.to_csv(out_path, index=False)


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, format="mixed")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="raise")
    return df.dropna().drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)


def data_quality(symbol: str, timeframe: str, ohlcv: pd.DataFrame, funding: pd.DataFrame, start: datetime, end: datetime) -> dict[str, object]:
    expected = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=interval_to_pandas_freq(timeframe), inclusive="left")
    timestamps = pd.DatetimeIndex(ohlcv["timestamp"])
    missing = expected.difference(timestamps)
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(milliseconds=interval_to_ms(timeframe))
    funding_gaps = funding["funding_time"].diff().dropna()
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ohlcv_rows": int(len(ohlcv)),
        "expected_ohlcv_rows": int(len(expected)),
        "missing_ohlcv_rows": int(len(missing)),
        "duplicate_ohlcv_rows": int(ohlcv["timestamp"].duplicated().sum()),
        "large_ohlcv_gap_count": int((gaps > expected_delta).sum()),
        "max_ohlcv_gap_minutes": float(gaps.max().total_seconds() / 60.0) if not gaps.empty else 0.0,
        "zero_volume_rows": int((ohlcv["volume"] <= 0).sum()),
        "bad_ohlc_rows": int(((ohlcv["high"] < ohlcv[["open", "close", "low"]].max(axis=1)) | (ohlcv["low"] > ohlcv[["open", "close", "high"]].min(axis=1))).sum()),
        "funding_rows": int(len(funding)),
        "first_funding_time": funding["funding_time"].min().isoformat() if not funding.empty else None,
        "last_funding_time": funding["funding_time"].max().isoformat() if not funding.empty else None,
        "large_funding_gap_count": int((funding_gaps > pd.Timedelta(hours=8, minutes=1)).sum()),
        "max_funding_gap_hours": float(funding_gaps.max().total_seconds() / 3600.0) if not funding_gaps.empty else 0.0,
    }


def ensure_data(
    symbols: list[str],
    timeframes: list[str],
    data_dir: Path,
    funding_dir: Path,
    start: datetime,
    end: datetime,
    download: bool,
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    loaded_ohlcv: dict[tuple[str, str], pd.DataFrame] = {}
    loaded_funding: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    for symbol in symbols:
        funding_path = funding_dir / f"{symbol}.csv"
        if download or not funding_path.exists():
            print(f"Downloading funding {symbol}")
            download_funding(symbol, start, end, funding_path)
        funding = load_funding(funding_path)
        funding = funding[(funding["funding_time"] >= pd.Timestamp(start)) & (funding["funding_time"] <= pd.Timestamp(end))].reset_index(drop=True)
        if funding.empty:
            raise RuntimeError(f"No funding rows for {symbol}")
        loaded_funding[symbol] = funding
        for timeframe in timeframes:
            path = data_dir / timeframe / f"{symbol}.csv"
            needs_download = download or not path.exists()
            if not needs_download:
                probe = load_ohlcv(path)
                has_start = probe["timestamp"].min().to_pydatetime() <= start
                has_end = probe["timestamp"].max().to_pydatetime() >= end - pd.Timedelta(milliseconds=interval_to_ms(timeframe))
                needs_download = not (has_start and has_end)
            if needs_download:
                print(f"Downloading OHLCV {symbol} {timeframe}")
                download_klines(symbol, timeframe, start, end, path)
            ohlcv = load_ohlcv(path)
            ohlcv = ohlcv[(ohlcv["timestamp"] >= pd.Timestamp(start)) & (ohlcv["timestamp"] < pd.Timestamp(end))].reset_index(drop=True)
            if ohlcv.empty:
                raise RuntimeError(f"No OHLCV rows for {symbol} {timeframe}")
            loaded_ohlcv[(symbol, timeframe)] = ohlcv
            quality_rows.append(data_quality(symbol, timeframe, ohlcv, funding, start, end))
    return loaded_ohlcv, loaded_funding, pd.DataFrame(quality_rows)


def add_features(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv.copy()
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
    out["candle_range"] = out["high"] - out["low"]
    out["body_pct"] = (out["close"] - out["open"]).abs() / out["candle_range"].replace(0, np.nan)
    out["close_location"] = (out["close"] - out["low"]) / out["candle_range"].replace(0, np.nan)
    out["reaction_r"] = (out["close"] - out["open"]) / out["atr_14_prior"].replace(0, np.nan)
    f = funding.copy()
    mean = f["funding_rate"].rolling(FUNDING_Z_LOOKBACK, min_periods=FUNDING_Z_LOOKBACK).mean().shift(1)
    std = f["funding_rate"].rolling(FUNDING_Z_LOOKBACK, min_periods=FUNDING_Z_LOOKBACK).std(ddof=0).shift(1)
    f["funding_z"] = (f["funding_rate"] - mean) / std.replace(0, np.nan)
    out = pd.merge_asof(out.sort_values("timestamp"), f.sort_values("funding_time"), left_on="timestamp", right_on="funding_time", direction="backward")
    return out


def add_signals(df: pd.DataFrame, params: FundingParams) -> pd.DataFrame:
    out = df.copy()
    required = ["atr_14_prior", "body_pct", "close_location", "reaction_r", "funding_rate", "funding_z"]
    valid = out[required].notna().all(axis=1)
    funding_short = out["funding_z"] >= params.funding_z_min
    funding_long = out["funding_z"] <= -params.funding_z_min
    short_reaction = (
        (out["close"] < out["open"])
        & ((-out["reaction_r"]) >= params.reaction_atr_min)
        & (out["close_location"] <= params.close_loc_short_max)
        & (out["body_pct"] >= params.body_min)
    )
    long_reaction = (
        (out["close"] > out["open"])
        & (out["reaction_r"] >= params.reaction_atr_min)
        & (out["close_location"] >= params.close_loc_long_min)
        & (out["body_pct"] >= params.body_min)
    )
    out["short_signal"] = valid & funding_short & short_reaction
    out["long_signal"] = valid & funding_long & long_reaction
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
    params: FundingParams,
    exit_model: ExitModel,
    friction: Friction,
    entry_mode: str,
) -> list[dict[str, object]]:
    rows = add_signals(featured, params).reset_index(drop=True)
    max_hold = exit_model.max_hold(timeframe)
    timestamps = rows["timestamp"].to_numpy()
    funding_times = rows["funding_time"].to_numpy()
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    atrs = rows["atr_14_prior"].to_numpy(dtype=float)
    funding_rates = rows["funding_rate"].to_numpy(dtype=float)
    funding_z = rows["funding_z"].to_numpy(dtype=float)
    reaction_r = rows["reaction_r"].to_numpy(dtype=float)
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
            if side == "long":
                hit_stop = float(lows[j]) <= stop
                hit_target = float(highs[j]) >= target
            else:
                hit_stop = float(highs[j]) >= stop
                hit_target = float(lows[j]) <= target
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
                "funding_time": funding_times[signal_i],
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
                "funding_rate": float(funding_rates[signal_i]),
                "funding_z": float(funding_z[signal_i]),
                "reaction_r": float(reaction_r[signal_i]),
                "body_pct": float(body_pct[signal_i]),
                "close_location": float(close_location[signal_i]),
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "exit_reason": exit_reason,
                "funding_z_min": params.funding_z_min,
                "reaction_atr_min": params.reaction_atr_min,
                "body_min": params.body_min,
                "close_loc_long_min": params.close_loc_long_min,
                "close_loc_short_max": params.close_loc_short_max,
                "funding_z_lookback": params.funding_z_lookback,
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
            path_rows.append({**dict(zip(group_cols, key_tuple)), "max_drawdown_r": max_drawdown_from_returns(values), "max_losing_streak": max_losing_streak(values)})
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
    best[f"best_{dimension}_ev_share"] = np.where(best["scenario_total_ev_r"] > 0, best["positive_dim_ev_r"] / best["scenario_total_ev_r"], np.nan)
    return best[scenario_cols + [f"best_{dimension}_ev_share", f"best_{dimension}_total_ev_r"]]


def build_summaries(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "timeframe",
        "param_id",
        "exit_model",
        "friction_case",
        "entry_mode",
        "funding_z_min",
        "reaction_atr_min",
        "body_min",
        "close_loc_long_min",
        "close_loc_short_max",
        "funding_z_lookback",
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
    by_symbol = fast_group_metrics(trades, scenario_cols + ["symbol"], include_path_metrics=False).sort_values(["timeframe", "exit_model", "friction_case", "entry_mode", "param_id", "symbol"])
    by_month = fast_group_metrics(month_trades, scenario_cols + ["month"], include_path_metrics=False).sort_values(["timeframe", "exit_model", "friction_case", "entry_mode", "param_id", "month"])
    pivot_keys = [
        "timeframe",
        "param_id",
        "exit_model",
        "friction_case",
        "funding_z_min",
        "reaction_atr_min",
        "body_min",
        "close_loc_long_min",
        "close_loc_short_max",
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


def param_grid() -> list[FundingParams]:
    return [
        FundingParams(funding_z_min=fz, reaction_atr_min=react, body_min=body, close_loc_long_min=cl, close_loc_short_max=cs)
        for fz in FUNDING_Z_MINS
        for react in REACTION_ATR_MINS
        for body in BODY_MINS
        for cl, cs in CLOSE_LOC_PAIRS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Funding reaction Phase 1 backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--funding-dir", type=Path, default=Path("data") / "binance_um_funding")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "funding_reaction")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--trades-sample-size", type=int, default=5000)
    return parser.parse_args()


def pass_fail_summary(overall: pd.DataFrame) -> dict[str, object]:
    if overall.empty:
        return {"overall_pass": False, "reason": "no trades"}
    base = overall[overall["friction_case"] == "base"].copy()
    stress = overall[overall["friction_case"] == "stress"].copy()
    best = base.sort_values("ev_per_trade_r", ascending=False).iloc[0]
    keys = ["timeframe", "param_id", "exit_model", "entry_mode"]
    stress_match = stress
    for col in keys:
        stress_match = stress_match[stress_match[col] == best[col]]
    stress_ev = float(stress_match["ev_per_trade_r"].iloc[0]) if not stress_match.empty else np.nan
    if best["entry_mode"] == "next_open":
        close_match = base[(base["timeframe"] == best["timeframe"]) & (base["param_id"] == best["param_id"]) & (base["exit_model"] == best["exit_model"]) & (base["entry_mode"] == "next_close")]
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
    return {"best_base_scenario": best.to_dict(), "matched_stress_ev_per_trade_r": stress_ev, "matched_next_close_ev_per_trade_r": next_close_ev, "checks": checks}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    symbols = [symbol.upper() for symbol in args.symbols]
    timeframes = [timeframe.lower() for timeframe in args.timeframes]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv, funding, quality = ensure_data(symbols, timeframes, args.data_dir, args.funding_dir, start, end, args.download)
    featured = {(symbol, timeframe): add_features(ohlcv[(symbol, timeframe)], funding[symbol]) for symbol in symbols for timeframe in timeframes}
    params = param_grid()
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
                            records.extend(simulate_symbol(symbol, timeframe, featured[(symbol, timeframe)], params_item, exit_model, friction, entry_mode))
    trades = pd.DataFrame(records)
    if not trades.empty:
        for col in ["signal_time", "funding_time", "entry_time", "exit_time"]:
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
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Funding-based reaction Phase 1 backtest without open interest.",
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
            "No ML, open interest, funding prediction, liquidation feed, limit orders, or intrabar entries.",
            "Funding z-score uses prior funding observations only via rolling mean/std shifted by one funding event.",
            "Latest known funding at or before the signal candle timestamp is aligned to OHLCV with merge_asof backward.",
            "Shorts require extremely positive funding plus bearish reaction; longs require extremely negative funding plus bullish reaction.",
            "Signals are known only after candle close; entries occur at next candle open or next candle close.",
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
        "top_10_by_ev_per_trade": overall.sort_values("ev_per_trade_r", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    write_json(summary_path, summary)
    print(f"Wrote funding reaction reports to {args.out_dir.resolve()}")
    if not overall.empty:
        print(overall.sort_values("ev_per_trade_r", ascending=False).head(12).to_string(index=False))
    print(json.dumps(pass_fail.get("checks", pass_fail), indent=2, default=str))


if __name__ == "__main__":
    main()
