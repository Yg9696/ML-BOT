import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


@dataclass(frozen=True)
class Params:
    timeframe: str
    lookback: int
    atr_period: int
    expansion_mult: float
    volume_max_mult: float
    range_max_mult: float
    max_same_dir: int
    stop_atr_mult: float
    target_r: float
    max_hold_bars: int
    fee_bps: float
    slippage_bps: float
    stress_fee_bps: float
    stress_slippage_bps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crude executable OHLCV volatility expansion validation."
    )
    parser.add_argument("--data-dir", default="data/ohlcv")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "LINKUSDT",
        ],
    )
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--atr-period", type=int, default=48)
    parser.add_argument("--expansion-mult", type=float, default=1.5)
    parser.add_argument("--volume-max-mult", type=float, default=4.0)
    parser.add_argument("--range-max-mult", type=float, default=3.0)
    parser.add_argument("--max-same-dir", type=int, default=3)
    parser.add_argument("--stop-atr-mult", type=float, default=1.0)
    parser.add_argument("--target-r", type=float, default=1.5)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--stress-fee-bps", type=float, default=6.0)
    parser.add_argument("--stress-slippage-bps", type=float, default=8.0)
    return parser.parse_args()


def ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def download_symbol(symbol: str, interval: str, start: str, end: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{interval}.csv"
    start_ms = ms(start)
    end_ms = ms(end)
    rows = []

    while start_ms < end_ms:
        response = requests.get(
            BINANCE_FAPI,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
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
        next_start = int(batch[-1][0]) + 1
        if next_start <= start_ms:
            break
        start_ms = next_start
        time.sleep(0.08)

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {interval}")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df.to_csv(out_path, index=False)
    return out_path


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.rename(columns={timestamp_col: "timestamp"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[required].sort_values("timestamp").drop_duplicates("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr_components = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["tr"] = tr_components.max(axis=1)
    out["atr_prior"] = out["tr"].rolling(p.atr_period).mean().shift(1)
    out["vol_mean_prior"] = out["volume"].rolling(p.atr_period).mean().shift(1)
    out["prior_high"] = out["high"].rolling(p.lookback).max().shift(1)
    out["prior_low"] = out["low"].rolling(p.lookback).min().shift(1)
    direction = np.sign(out["close"] - out["open"]).replace(0, np.nan).ffill().fillna(0)
    same_dir = []
    streak = 0
    prev = 0
    for value in direction:
        if value == prev and value != 0:
            streak += 1
        else:
            streak = 1 if value != 0 else 0
        same_dir.append(streak)
        prev = value
    out["same_dir_count"] = same_dir
    out["same_dir_prior"] = out["same_dir_count"].shift(1)
    return out


def build_signals(df: pd.DataFrame, p: Params) -> pd.Series:
    valid = df["atr_prior"].notna() & (df["atr_prior"] > 0) & (df["vol_mean_prior"] > 0)
    expansion = df["tr"] >= p.expansion_mult * df["atr_prior"]
    not_crowded_volume = df["volume"] <= p.volume_max_mult * df["vol_mean_prior"]
    not_crowded_range = df["tr"] <= p.range_max_mult * df["atr_prior"]
    not_extended = df["same_dir_prior"].fillna(0) <= p.max_same_dir
    long_signal = df["close"] > df["prior_high"]
    short_signal = df["close"] < df["prior_low"]
    raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    allowed = valid & expansion & not_crowded_volume & not_crowded_range & not_extended
    return pd.Series(np.where(allowed, raw, 0), index=df.index)


def friction_r(entry: float, risk_per_unit: float, fee_bps: float, slippage_bps: float) -> float:
    round_trip_bps = 2.0 * (fee_bps + slippage_bps)
    return (entry * round_trip_bps / 10000.0) / risk_per_unit


def simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    p: Params,
    entry_mode: str,
    fee_bps: float,
    slippage_bps: float,
) -> list[dict]:
    data = add_indicators(df, p)
    data["signal"] = build_signals(data, p)
    trades = []
    active_until = -1

    for signal_idx, row in data.iterrows():
        direction = int(row["signal"])
        if direction == 0 or signal_idx <= active_until:
            continue

        entry_idx = signal_idx + 1
        if entry_mode == "next_close":
            entry_idx = signal_idx + 1
        if entry_idx >= len(data):
            break

        entry_row = data.iloc[entry_idx]
        entry_price = (
            float(entry_row["open"])
            if entry_mode == "next_open"
            else float(entry_row["close"])
        )
        atr = float(row["atr_prior"])
        risk_per_unit = p.stop_atr_mult * atr
        if not math.isfinite(risk_per_unit) or risk_per_unit <= 0:
            continue

        stop = entry_price - direction * risk_per_unit
        target = entry_price + direction * risk_per_unit * p.target_r
        exit_idx = min(entry_idx + p.max_hold_bars, len(data) - 1)
        exit_price = float(data.iloc[exit_idx]["close"])
        exit_reason = "time"

        for j in range(entry_idx + 1, exit_idx + 1):
            bar = data.iloc[j]
            high = float(bar["high"])
            low = float(bar["low"])
            if direction == 1:
                stop_hit = low <= stop
                target_hit = high >= target
            else:
                stop_hit = high >= stop
                target_hit = low <= target

            if stop_hit:
                exit_idx = j
                exit_price = stop
                exit_reason = "stop"
                break
            if target_hit:
                exit_idx = j
                exit_price = target
                exit_reason = "target"
                break

        gross_r = direction * (exit_price - entry_price) / risk_per_unit
        net_r = gross_r - friction_r(entry_price, risk_per_unit, fee_bps, slippage_bps)
        active_until = exit_idx
        trades.append(
            {
                "symbol": symbol,
                "timeframe": p.timeframe,
                "entry_mode": entry_mode,
                "signal_time": row["timestamp"],
                "entry_time": entry_row["timestamp"],
                "exit_time": data.iloc[exit_idx]["timestamp"],
                "direction": "long" if direction == 1 else "short",
                "entry": entry_price,
                "exit": exit_price,
                "stop": stop,
                "target": target,
                "exit_reason": exit_reason,
                "gross_r": gross_r,
                "net_r": net_r,
                "fees_bps": fee_bps,
                "slippage_bps": slippage_bps,
                **asdict(p),
            }
        )

    return trades


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity - peak).min())


def max_losing_streak(values: Iterable[float]) -> int:
    worst = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {
            "scenario": label,
            "trades": 0,
            "years": 0,
            "trades_per_year": 0,
            "ev_r": 0,
            "total_r": 0,
            "max_drawdown_r": 0,
            "max_losing_streak": 0,
            "top_symbol_ev_share": 0,
            "top_month_ev_share": 0,
        }

    ordered = trades.sort_values("exit_time").reset_index(drop=True)
    equity = ordered["net_r"].cumsum()
    start = ordered["entry_time"].min()
    end = ordered["exit_time"].max()
    years = max((end - start).days / 365.25, 1 / 365.25)
    symbol_ev = ordered.groupby("symbol")["net_r"].sum()
    month_ev = ordered.groupby(ordered["exit_time"].dt.to_period("M"))["net_r"].sum()
    total_positive_ev = max(float(ordered["net_r"].sum()), 0.0)

    def top_share(series: pd.Series) -> float:
        if total_positive_ev <= 0:
            return 0.0
        return float(series.max() / total_positive_ev)

    return {
        "scenario": label,
        "trades": int(len(ordered)),
        "years": years,
        "trades_per_year": len(ordered) / years,
        "ev_r": float(ordered["net_r"].mean()),
        "total_r": float(ordered["net_r"].sum()),
        "max_drawdown_r": max_drawdown(equity),
        "max_losing_streak": max_losing_streak(ordered["net_r"]),
        "top_symbol_ev_share": top_share(symbol_ev),
        "top_month_ev_share": top_share(month_ev),
    }


def verdict(base_open: dict, base_close: dict, stress_open: dict, stress_close: dict) -> str:
    checks = {
        "base_next_open_ev": base_open["ev_r"] >= 0.3,
        "base_next_close_ev": base_close["ev_r"] >= 0.3,
        "trades_per_year": min(base_open["trades_per_year"], base_close["trades_per_year"]) >= 100,
        "stress_ev_positive": min(stress_open["ev_r"], stress_close["ev_r"]) > 0,
        "next_close_not_collapsed": base_close["ev_r"] >= 0.5 * max(base_open["ev_r"], 0.000001),
        "symbol_concentration": max(base_open["top_symbol_ev_share"], base_close["top_symbol_ev_share"]) <= 0.4,
        "month_concentration": max(base_open["top_month_ev_share"], base_close["top_month_ev_share"]) <= 0.4,
        "losing_streak": max(base_open["max_losing_streak"], base_close["max_losing_streak"]) <= 12,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return "PASS" if not failed else "FAIL: " + ", ".join(failed)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    params = Params(
        timeframe=args.timeframe,
        lookback=args.lookback,
        atr_period=args.atr_period,
        expansion_mult=args.expansion_mult,
        volume_max_mult=args.volume_max_mult,
        range_max_mult=args.range_max_mult,
        max_same_dir=args.max_same_dir,
        stop_atr_mult=args.stop_atr_mult,
        target_r=args.target_r,
        max_hold_bars=args.max_hold_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        stress_fee_bps=args.stress_fee_bps,
        stress_slippage_bps=args.stress_slippage_bps,
    )

    if args.download:
        for symbol in args.symbols:
            print(f"Downloading {symbol} {args.timeframe}")
            download_symbol(symbol, args.timeframe, args.start, args.end, data_dir)

    all_trades = []
    files = []
    for symbol in args.symbols:
        path = data_dir / f"{symbol}_{args.timeframe}.csv"
        if path.exists():
            files.append((symbol, path))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir} for {args.timeframe}")

    for symbol, path in files:
        df = load_ohlcv(path)
        for entry_mode in ["next_open", "next_close"]:
            all_trades.extend(
                simulate_symbol(
                    symbol,
                    df,
                    params,
                    entry_mode,
                    args.fee_bps,
                    args.slippage_bps,
                )
            )
            all_trades.extend(
                simulate_symbol(
                    symbol,
                    df,
                    params,
                    entry_mode,
                    args.stress_fee_bps,
                    args.stress_slippage_bps,
                )
            )

    trades = pd.DataFrame(all_trades)
    if trades.empty:
        raise SystemExit("No trades produced.")
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["friction_case"] = np.where(
        (trades["fees_bps"] == args.fee_bps)
        & (trades["slippage_bps"] == args.slippage_bps),
        "base",
        "stress",
    )

    summaries = []
    for friction_case in ["base", "stress"]:
        for entry_mode in ["next_open", "next_close"]:
            subset = trades[
                (trades["friction_case"] == friction_case)
                & (trades["entry_mode"] == entry_mode)
            ].copy()
            summaries.append(summarize(subset, f"{friction_case}_{entry_mode}"))

    summary = pd.DataFrame(summaries)
    base_open = summary.loc[summary["scenario"] == "base_next_open"].iloc[0].to_dict()
    base_close = summary.loc[summary["scenario"] == "base_next_close"].iloc[0].to_dict()
    stress_open = summary.loc[summary["scenario"] == "stress_next_open"].iloc[0].to_dict()
    stress_close = summary.loc[summary["scenario"] == "stress_next_close"].iloc[0].to_dict()
    final_verdict = verdict(base_open, base_close, stress_open, stress_close)
    summary["verdict"] = final_verdict

    monthly = (
        trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str))
        .groupby(["friction_case", "entry_mode", "month"], as_index=False)
        .agg(trades=("net_r", "size"), total_r=("net_r", "sum"), ev_r=("net_r", "mean"))
    )
    by_symbol = (
        trades.groupby(["friction_case", "entry_mode", "symbol"], as_index=False)
        .agg(trades=("net_r", "size"), total_r=("net_r", "sum"), ev_r=("net_r", "mean"))
    )

    trades.to_csv(reports_dir / "trades.csv", index=False)
    summary.to_csv(reports_dir / "summary.csv", index=False)
    monthly.to_csv(reports_dir / "monthly_distribution.csv", index=False)
    by_symbol.to_csv(reports_dir / "symbol_distribution.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nVerdict: {final_verdict}")
    print(f"Reports written to {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
