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
TIMEFRAME = "15m"
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
EVENT_RETURN_THRESHOLD = -0.02
ATR_PERIOD = 14
HOLD_BARS = 12


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class ExitModel:
    name: str
    stop_atr_mult: float | None
    target_r: float | None
    hold_bars: int = HOLD_BARS


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


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def data_quality(symbol: str, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    expected = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    missing = expected.difference(timestamps)
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(minutes=15)
    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
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
            needs_download = not (
                probe["timestamp"].min() <= start
                and probe["timestamp"].max() >= end - pd.Timedelta(minutes=15)
            )
        if needs_download:
            print(f"Downloading {symbol} {TIMEFRAME}")
            download_klines(symbol, TIMEFRAME, start, end, path)
        df = load_ohlcv(path)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        loaded[symbol] = add_features(df)
        quality_rows.append(data_quality(symbol, df, start, end))
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
    out["atr_14_prior"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().shift(1)
    out["event_return"] = out["close"] / prev_close - 1.0
    return out


def apply_entry_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate)


def apply_exit_slippage(raw_price: float, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate)


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


def simulate_symbol(symbol: str, df: pd.DataFrame, entry_mode: str, model: ExitModel, friction: Friction) -> list[dict[str, object]]:
    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs = df["atr_14_prior"].to_numpy(dtype=float)
    event_returns = df["event_return"].to_numpy(dtype=float)
    events = np.flatnonzero(event_returns <= EVENT_RETURN_THRESHOLD)
    trades: list[dict[str, object]] = []
    next_signal_i = 0
    for signal_i in events:
        if signal_i < next_signal_i:
            continue
        entry_i = signal_i + 1
        if entry_i >= len(df):
            break
        raw_entry = float(opens[entry_i] if entry_mode == "next_open" else closes[entry_i])
        entry = apply_entry_slippage(raw_entry, friction)
        atr = float(atrs[signal_i])
        risk = atr if math.isfinite(atr) and atr > 0 else np.nan
        stop = None if model.stop_atr_mult is None or not math.isfinite(risk) else entry - model.stop_atr_mult * risk
        target = None if model.target_r is None or not math.isfinite(risk) else entry + model.target_r * model.stop_atr_mult * risk
        first_exit_i = entry_i if entry_mode == "next_open" else entry_i + 1
        last_i = min(entry_i + model.hold_bars, len(df) - 1)
        if first_exit_i > last_i:
            continue
        raw_exit = float(closes[last_i])
        exit_i = last_i
        exit_reason = "horizon"
        for j in range(first_exit_i, last_i + 1):
            hit_stop = stop is not None and float(lows[j]) <= stop
            hit_target = target is not None and float(highs[j]) >= target
            if hit_stop:
                raw_exit = float(stop)
                exit_i = j
                exit_reason = "stop"
                break
            if hit_target:
                raw_exit = float(target)
                exit_i = j
                exit_reason = "target"
                break
        exit_price = apply_exit_slippage(raw_exit, friction)
        gross_return = exit_price / entry - 1.0
        fee_return = friction.taker_fee_rate * (entry + exit_price) / entry
        net_return = gross_return - fee_return
        gross_r = np.nan if not math.isfinite(risk) or risk <= 0 else (exit_price - entry) / risk
        fee_r = np.nan if not math.isfinite(risk) or risk <= 0 else friction.taker_fee_rate * (entry + exit_price) / risk
        net_r = np.nan if not math.isfinite(gross_r) else gross_r - fee_r
        trades.append(
            {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "event_type": "extreme_down_pct",
                "event_threshold": EVENT_RETURN_THRESHOLD,
                "entry_mode": entry_mode,
                "exit_model": model.name,
                "friction_case": friction.name,
                "signal_time": timestamps[signal_i],
                "entry_time": timestamps[entry_i],
                "exit_time": timestamps[exit_i],
                "event_return": float(event_returns[signal_i]),
                "raw_entry": raw_entry,
                "entry": entry,
                "raw_exit": raw_exit,
                "exit": exit_price,
                "atr_at_signal": atr,
                "risk": risk,
                "stop": stop,
                "target": target,
                "gross_return": gross_return,
                "fee_return": fee_return,
                "net_return": net_return,
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "exit_reason": exit_reason,
                "hold_bars": model.hold_bars,
                "stop_atr_mult": model.stop_atr_mult,
                "target_r": model.target_r,
                "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                "slippage_bps_per_side": friction.slippage_rate * 10_000,
            }
        )
        next_signal_i = exit_i + 1
    return trades


def profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = values[values < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else 0.0
    return float(wins / abs(losses))


def summarize_group(group: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_return"]
    r_values = ordered["net_r"].dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    month_values = ordered["exit_time"].dt.strftime("%Y-%m")
    years = max((ordered["entry_time"].max() - ordered["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "total_trades": int(len(ordered)),
            "trades_per_year": float(len(ordered) / years),
            "avg_net_return_pct": float(returns.mean() * 100.0),
            "median_net_return_pct": float(returns.median() * 100.0),
            "win_rate": float((returns > 0).mean()),
            "total_net_return_units": float(returns.sum()),
            "ev_per_trade_r": float(r_values.mean()) if not r_values.empty else np.nan,
            "total_ev_r": float(r_values.sum()) if not r_values.empty else np.nan,
            "max_drawdown_return_units": max_drawdown(returns),
            "max_drawdown_r": max_drawdown(r_values) if not r_values.empty else np.nan,
            "max_losing_streak": max_losing_streak(returns),
            "profit_factor": profit_factor(returns),
            "average_win_pct": float(wins.mean() * 100.0) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean() * 100.0) if not losses.empty else 0.0,
            "best_symbol": ordered.groupby("symbol")["net_return"].sum().idxmax(),
            "worst_symbol": ordered.groupby("symbol")["net_return"].sum().idxmin(),
            "best_month": ordered.assign(month=month_values).groupby("month")["net_return"].sum().idxmax(),
            "worst_month": ordered.assign(month=month_values).groupby("month")["net_return"].sum().idxmin(),
        }
    )
    total_positive = max(float(returns.sum()), 0.0)
    symbol_ev = ordered.groupby("symbol")["net_return"].sum()
    month_ev = ordered.assign(month=month_values).groupby("month")["net_return"].sum()
    out["best_symbol_contribution"] = float(symbol_ev.max() / total_positive) if total_positive > 0 else np.nan
    out["best_month_contribution"] = float(month_ev.max() / total_positive) if total_positive > 0 else np.nan
    return out


def build_reports(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "entry_mode",
        "exit_model",
        "friction_case",
        "event_threshold",
        "hold_bars",
        "stop_atr_mult",
        "target_r",
        "taker_fee_bps_per_side",
        "slippage_bps_per_side",
    ]
    overall = pd.DataFrame([summarize_group(group, scenario_cols) for _, group in trades.groupby(scenario_cols, sort=False, dropna=False)])
    by_symbol = pd.DataFrame([summarize_group(group, scenario_cols + ["symbol"]) for _, group in trades.groupby(scenario_cols + ["symbol"], sort=False, dropna=False)])
    month_trades = trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m"))
    by_month = pd.DataFrame([summarize_group(group, scenario_cols + ["month"]) for _, group in month_trades.groupby(scenario_cols + ["month"], sort=False, dropna=False)])
    pivot_keys = ["exit_model", "friction_case", "event_threshold", "hold_bars", "stop_atr_mult", "target_r"]
    open_rows = overall[overall["entry_mode"] == "next_open"].set_index(pivot_keys)
    close_rows = overall[overall["entry_mode"] == "next_close"].set_index(pivot_keys)
    comparison = open_rows[["total_trades", "avg_net_return_pct", "ev_per_trade_r", "win_rate", "profit_factor"]].join(
        close_rows[["total_trades", "avg_net_return_pct", "ev_per_trade_r", "win_rate", "profit_factor"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["avg_return_delta_close_minus_open_pct"] = comparison["avg_net_return_pct_next_close"] - comparison["avg_net_return_pct_next_open"]
    comparison["ev_r_delta_close_minus_open"] = comparison["ev_per_trade_r_next_close"] - comparison["ev_per_trade_r_next_open"]
    return overall, by_symbol, by_month, comparison


def pass_fail_summary(overall: pd.DataFrame) -> dict[str, object]:
    base = overall[overall["friction_case"] == "base"].copy()
    if base.empty:
        return {"overall_pass": False, "reason": "no base scenarios"}
    best = base.sort_values("avg_net_return_pct", ascending=False).iloc[0]
    stress = overall[
        (overall["friction_case"] == "stress")
        & (overall["entry_mode"] == best["entry_mode"])
        & (overall["exit_model"] == best["exit_model"])
    ]
    stress_avg = float(stress["avg_net_return_pct"].iloc[0]) if not stress.empty else np.nan
    checks = {
        "at_least_250_trades": int(best["total_trades"]) >= 250,
        "base_avg_return_positive": float(best["avg_net_return_pct"]) > 0,
        "stress_not_deeply_negative": pd.notna(stress_avg) and stress_avg > -0.20,
        "symbol_concentration_ok": pd.notna(best["best_symbol_contribution"]) and best["best_symbol_contribution"] <= 0.40,
        "month_concentration_ok": pd.notna(best["best_month_contribution"]) and best["best_month_contribution"] <= 0.40,
        "strong_ev_r_ge_0_2": pd.notna(best["ev_per_trade_r"]) and float(best["ev_per_trade_r"]) >= 0.2,
    }
    checks["minimum_pass"] = all(checks[key] for key in ["at_least_250_trades", "base_avg_return_positive", "stress_not_deeply_negative", "symbol_concentration_ok", "month_concentration_ok"])
    checks["strong_pass"] = checks["minimum_pass"] and checks["strong_ev_r_ge_0_2"]
    return {"best_base_scenario": best.to_dict(), "matched_stress_avg_net_return_pct": stress_avg, "checks": checks}


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="15m extreme down mean reversion feasibility backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "extreme_down_mean_reversion")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--trades-sample-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    symbols = [symbol.upper() for symbol in args.symbols]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded, quality = ensure_data(symbols, args.data_dir, start, end, args.download)
    models = [
        ExitModel("fixed_horizon_12", stop_atr_mult=None, target_r=None),
        ExitModel("stop_1atr_horizon_12", stop_atr_mult=1.0, target_r=None),
        ExitModel("stop_1atr_tp_1_5r", stop_atr_mult=1.0, target_r=1.5),
        ExitModel("wide_1_5atr_tp_1_5r", stop_atr_mult=1.5, target_r=1.5),
        ExitModel("wide_1_5atr_tp_2r", stop_atr_mult=1.5, target_r=2.0),
    ]
    frictions = [Friction("base", taker_fee_rate=0.0004, slippage_rate=0.0005), Friction("stress", taker_fee_rate=0.0008, slippage_rate=0.0010)]
    records: list[dict[str, object]] = []
    for model in models:
        for friction in frictions:
            for entry_mode in ["next_open", "next_close"]:
                for symbol, df in loaded.items():
                    records.extend(simulate_symbol(symbol, df, entry_mode, model, friction))
    trades = pd.DataFrame(records)
    if not trades.empty:
        for col in ["signal_time", "entry_time", "exit_time"]:
            trades[col] = pd.to_datetime(trades[col], utc=True)
        trades = trades.sort_values(["exit_model", "friction_case", "entry_mode", "exit_time", "symbol"]).reset_index(drop=True)
    overall, by_symbol, by_month, comparison = build_reports(trades)
    pass_fail = pass_fail_summary(overall)
    trades_path = args.out_dir / "trades.csv"
    trades_sample_path = args.out_dir / "trades_sample.csv"
    overall_path = args.out_dir / "overall_results.csv"
    by_symbol_path = args.out_dir / "by_symbol.csv"
    by_month_path = args.out_dir / "by_month.csv"
    comparison_path = args.out_dir / "execution_comparison.csv"
    quality_path = args.out_dir / "data_quality_report.csv"
    summary_path = args.out_dir / "phase_summary.json"
    trades.to_csv(trades_path, index=False)
    trades.head(args.trades_sample_size).to_csv(trades_sample_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_month.to_csv(by_month_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    quality.to_csv(quality_path, index=False)
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "15m extreme down mean reversion feasibility backtest.",
        "symbols": symbols,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "event_return_threshold": EVENT_RETURN_THRESHOLD,
        "hold_bars": HOLD_BARS,
        "exit_models": [asdict(model) for model in models],
        "friction": [asdict(friction) for friction in frictions],
        "total_trade_rows": int(trades.shape[0]),
        "scenario_count": int(overall.shape[0]),
        "reports": {
            "trades_csv": str(trades_path),
            "trades_sample_csv": str(trades_sample_path),
            "overall_results_csv": str(overall_path),
            "by_symbol_csv": str(by_symbol_path),
            "by_month_csv": str(by_month_path),
            "execution_comparison_csv": str(comparison_path),
            "data_quality_report_csv": str(quality_path),
            "phase_summary_json": str(summary_path),
        },
        "assumptions": [
            "Event is fixed: 15m close-to-close return <= -2.0%.",
            "Long only.",
            "Signal known only after event candle close.",
            "Entry is next candle open or next candle close.",
            "No limit orders, no ML, no OI, no funding.",
            "If stop and target are both touched in one candle, stop wins.",
        ],
        "top_10_by_avg_net_return_pct": overall.sort_values("avg_net_return_pct", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(overall.sort_values("avg_net_return_pct", ascending=False).head(12).to_string(index=False))
    print(json.dumps(pass_fail["checks"], indent=2, default=str))


if __name__ == "__main__":
    main()
