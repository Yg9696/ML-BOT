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


BINANCE_FAPI = "https://fapi.binance.com"
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]


@dataclass(frozen=True)
class StrategyParams:
    timeframe: str = "1h"
    breakout_lookback: int = 24
    atr_window: int = 24
    expansion_atr_mult: float = 1.25
    stop_atr_mult: float = 1.5
    target_r: float = 2.0
    max_hold_bars: int = 48
    max_signal_range_atr: float = 2.75
    max_close_extension_atr: float = 1.25
    max_volume_multiple: float = 3.0
    volume_window: int = 24


@dataclass(frozen=True)
class Friction:
    name: str
    fee_bps_per_side: float
    slippage_bps_per_side: float

    @property
    def bps_per_side(self) -> float:
        return self.fee_bps_per_side + self.slippage_bps_per_side


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in factors:
        raise ValueError(f"Unsupported interval: {interval}")
    return amount * factors[unit]


def download_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    data_dir: Path,
) -> Path:
    out_dir = data_dir / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.csv"

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = interval_to_ms(interval)
    rows: list[list[object]] = []

    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        response = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No klines returned for {symbol} {interval}")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(out_path, index=False)
    return out_path


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def add_signal_columns(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr_prev"] = true_range.rolling(params.atr_window).mean().shift(1)
    out["range_high_prev"] = out["high"].rolling(params.breakout_lookback).max().shift(1)
    out["range_low_prev"] = out["low"].rolling(params.breakout_lookback).min().shift(1)
    out["volume_mean_prev"] = out["volume"].rolling(params.volume_window).mean().shift(1)
    out["true_range"] = true_range

    valid = out["atr_prev"].notna() & (out["atr_prev"] > 0)
    expanded = out["true_range"] >= params.expansion_atr_mult * out["atr_prev"]
    not_chase = out["true_range"] <= params.max_signal_range_atr * out["atr_prev"]
    not_volume_climax = out["volume"] <= params.max_volume_multiple * out["volume_mean_prev"]

    long_break = out["close"] > out["range_high_prev"]
    short_break = out["close"] < out["range_low_prev"]
    long_extension = (out["close"] - out["range_high_prev"]) <= params.max_close_extension_atr * out["atr_prev"]
    short_extension = (out["range_low_prev"] - out["close"]) <= params.max_close_extension_atr * out["atr_prev"]

    out["long_signal"] = valid & expanded & not_chase & not_volume_climax & long_break & long_extension
    out["short_signal"] = valid & expanded & not_chase & not_volume_climax & short_break & short_extension
    return out


def apply_entry_friction(price: float, side: str, friction: Friction) -> float:
    multiplier = friction.bps_per_side / 10_000
    return price * (1 + multiplier) if side == "long" else price * (1 - multiplier)


def apply_exit_friction(price: float, side: str, friction: Friction) -> float:
    multiplier = friction.bps_per_side / 10_000
    return price * (1 - multiplier) if side == "long" else price * (1 + multiplier)


def r_multiple(entry: float, exit_price: float, side: str, risk: float) -> float:
    if side == "long":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    params: StrategyParams,
    friction: Friction,
    entry_mode: str,
) -> pd.DataFrame:
    sig = add_signal_columns(df, params).reset_index(drop=True)
    trades: list[dict[str, object]] = []
    i = 0

    while i < len(sig) - 2:
        row = sig.iloc[i]
        side = "long" if row["long_signal"] else "short" if row["short_signal"] else None
        if side is None:
            i += 1
            continue

        entry_idx = i + 1
        if entry_mode == "next_open":
            raw_entry = float(sig.at[entry_idx, "open"])
        elif entry_mode == "next_close":
            raw_entry = float(sig.at[entry_idx, "close"])
        else:
            raise ValueError(f"Unsupported entry mode: {entry_mode}")

        entry = apply_entry_friction(raw_entry, side, friction)
        atr = float(row["atr_prev"])
        risk = params.stop_atr_mult * atr
        if not math.isfinite(risk) or risk <= 0:
            i += 1
            continue

        if side == "long":
            stop = entry - risk
            target = entry + params.target_r * risk
        else:
            stop = entry + risk
            target = entry - params.target_r * risk

        exit_idx = None
        raw_exit = None
        exit_reason = "time"
        first_check = entry_idx + 1
        last_check = min(entry_idx + params.max_hold_bars, len(sig) - 2)

        for j in range(first_check, last_check + 1):
            close_j = float(sig.at[j, "close"])
            hit_stop = close_j <= stop if side == "long" else close_j >= stop
            hit_target = close_j >= target if side == "long" else close_j <= target
            if hit_stop or hit_target:
                exit_idx = j + 1
                raw_exit = float(sig.at[exit_idx, "open"])
                exit_reason = "stop" if hit_stop else "target"
                break

        if exit_idx is None:
            exit_idx = last_check + 1
            raw_exit = float(sig.at[exit_idx, "open"])

        exit_price = apply_exit_friction(raw_exit, side, friction)
        result_r = r_multiple(entry, exit_price, side, risk)
        trades.append(
            {
                "symbol": symbol,
                "timeframe": params.timeframe,
                "entry_mode": entry_mode,
                "friction": friction.name,
                "fee_bps_per_side": friction.fee_bps_per_side,
                "slippage_bps_per_side": friction.slippage_bps_per_side,
                "signal_time": row["timestamp"],
                "entry_time": sig.at[entry_idx, "timestamp"],
                "exit_time": sig.at[exit_idx, "timestamp"],
                "side": side,
                "raw_entry": raw_entry,
                "entry": entry,
                "raw_exit": raw_exit,
                "exit": exit_price,
                "risk": risk,
                "result_r": result_r,
                "exit_reason": exit_reason,
                **asdict(params),
            }
        )
        i = exit_idx + 1

    return pd.DataFrame(trades)


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    cumulative = values.cumsum()
    peak = cumulative.cummax()
    return float((cumulative - peak).min())


def max_losing_streak(values: Iterable[float]) -> int:
    current = 0
    worst = 0
    for value in values:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def summarize(trades: pd.DataFrame, start: datetime, end: datetime) -> dict[str, object]:
    if trades.empty:
        return {
            "trade_count": 0,
            "trades_per_year": 0.0,
            "ev_r_per_trade": 0.0,
            "total_r": 0.0,
            "max_drawdown_r": 0.0,
            "max_losing_streak": 0,
            "symbol_ev_max_share": None,
            "month_ev_max_share": None,
        }

    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    symbol_ev = trades.groupby("symbol")["result_r"].sum().sort_values(ascending=False)
    month_ev = trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str)).groupby("month")[
        "result_r"
    ].sum().sort_values(ascending=False)
    positive_total = max(float(trades["result_r"].sum()), 0.0)

    def max_share(series: pd.Series) -> float | None:
        if positive_total <= 0 or series.empty:
            return None
        return float(max(series.max(), 0.0) / positive_total)

    return {
        "trade_count": int(len(trades)),
        "trades_per_year": float(len(trades) / years),
        "ev_r_per_trade": float(trades["result_r"].mean()),
        "median_r_per_trade": float(trades["result_r"].median()),
        "win_rate": float((trades["result_r"] > 0).mean()),
        "total_r": float(trades["result_r"].sum()),
        "max_drawdown_r": max_drawdown(trades["result_r"]),
        "max_losing_streak": max_losing_streak(trades["result_r"]),
        "symbol_ev_max_share": max_share(symbol_ev),
        "month_ev_max_share": max_share(month_ev),
    }


def write_outputs(trades: pd.DataFrame, out_dir: Path, label: str, start: datetime, end: datetime) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"trades_{label}.csv"
    symbols_path = out_dir / f"symbol_distribution_{label}.csv"
    months_path = out_dir / f"monthly_distribution_{label}.csv"

    trades.to_csv(trades_path, index=False)

    if trades.empty:
        pd.DataFrame().to_csv(symbols_path, index=False)
        pd.DataFrame().to_csv(months_path, index=False)
    else:
        trades.groupby("symbol").agg(
            trades=("result_r", "size"),
            total_r=("result_r", "sum"),
            ev_r=("result_r", "mean"),
            max_drawdown_r=("result_r", max_drawdown),
        ).sort_values("total_r", ascending=False).to_csv(symbols_path)

        trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str)).groupby("month").agg(
            trades=("result_r", "size"),
            total_r=("result_r", "sum"),
            ev_r=("result_r", "mean"),
            max_drawdown_r=("result_r", max_drawdown),
        ).sort_values("month").to_csv(months_path)

    summary = summarize(trades, start, end)
    summary.update(
        {
            "trades_csv": str(trades_path),
            "symbol_distribution_csv": str(symbols_path),
            "monthly_distribution_csv": str(months_path),
        }
    )
    return summary


def pass_fail(base_open: dict[str, object], base_close: dict[str, object], stress_open: dict[str, object]) -> dict[str, object]:
    close_ev = float(base_close["ev_r_per_trade"])
    open_ev = float(base_open["ev_r_per_trade"])
    next_close_not_dramatically_worse = close_ev >= open_ev * 0.65 if open_ev > 0 else close_ev >= open_ev
    checks = {
        "base_ev_ge_0_3r": open_ev >= 0.3 and close_ev >= 0.3,
        "trades_per_year_ge_100": min(float(base_open["trades_per_year"]), float(base_close["trades_per_year"])) >= 100,
        "stress_does_not_collapse": float(stress_open["ev_r_per_trade"]) > 0,
        "next_close_not_dramatically_worse": next_close_not_dramatically_worse,
        "symbol_concentration_ok": all(
            value is not None and value <= 0.40
            for value in [base_open["symbol_ev_max_share"], base_close["symbol_ev_max_share"]]
        ),
        "month_concentration_ok": all(
            value is not None and value <= 0.40
            for value in [base_open["month_ev_max_share"], base_close["month_ev_max_share"]]
        ),
    }
    checks["overall_pass"] = all(checks.values())
    return checks


def run(args: argparse.Namespace) -> None:
    start = parse_dt(args.start)
    end = parse_dt(args.end) if args.end else datetime.now(timezone.utc)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    params = StrategyParams(timeframe=args.timeframe)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.download:
        for symbol in symbols:
            print(f"Downloading {symbol} {args.timeframe}")
            download_klines(symbol, args.timeframe, start, end, data_dir)

    base = Friction("base", fee_bps_per_side=4.0, slippage_bps_per_side=3.0)
    stress = Friction("stress", fee_bps_per_side=6.0, slippage_bps_per_side=10.0)
    summaries: dict[str, object] = {
        "params": asdict(params),
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "execution_note": "Signals use only closed candle t. Entries are at candle t+1 open or close. Exits are close-triggered and filled at next open.",
    }

    loaded: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = data_dir / args.timeframe / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run with --download first.")
        loaded[symbol] = load_ohlcv(path)

    for friction in [base, stress]:
        for entry_mode in ["next_open", "next_close"]:
            all_trades = [
                backtest_symbol(symbol, df, params, friction, entry_mode)
                for symbol, df in loaded.items()
            ]
            trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
            if not trades.empty:
                trades = trades.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
            label = f"{friction.name}_{entry_mode}_{args.timeframe}"
            summaries[label] = write_outputs(trades, out_dir, label, start, end)

    summaries["pass_fail"] = pass_fail(
        summaries[f"base_next_open_{args.timeframe}"],
        summaries[f"base_next_close_{args.timeframe}"],
        summaries[f"stress_next_open_{args.timeframe}"],
    )

    summary_path = out_dir / f"summary_{args.timeframe}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
