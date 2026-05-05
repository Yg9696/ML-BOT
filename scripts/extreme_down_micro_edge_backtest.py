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
EVENT_RETURN_THRESHOLD = -0.02
ATR_PERIOD = 14


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class ManagementModel:
    name: str
    horizon_bars: int
    stop_atr_mult: float | None = None
    scaled_exit: bool = False
    allow_stacking: bool = True


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
        cursor = int(batch[-1][0]) + step_ms
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
            needs_download = not (probe["timestamp"].min() <= start and probe["timestamp"].max() >= end - pd.Timedelta(minutes=15))
        if needs_download:
            print(f"Downloading {symbol} {TIMEFRAME}")
            download_klines(symbol, TIMEFRAME, start, end, path)
        raw = load_ohlcv(path)
        raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)].reset_index(drop=True)
        loaded[symbol] = add_features(raw)
        quality_rows.append(data_quality(symbol, raw, start, end))
    return loaded, pd.DataFrame(quality_rows)


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


def simulate_symbol(symbol: str, df: pd.DataFrame, entry_mode: str, model: ManagementModel, friction: Friction) -> list[dict[str, object]]:
    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs = df["atr_14_prior"].to_numpy(dtype=float)
    event_returns = df["event_return"].to_numpy(dtype=float)
    event_indices = np.flatnonzero(event_returns <= EVENT_RETURN_THRESHOLD)
    trades: list[dict[str, object]] = []
    blocked_until = -1

    for signal_i in event_indices:
        if not model.allow_stacking and signal_i <= blocked_until:
            continue
        entry_i = signal_i + 1
        final_exit_i = entry_i + model.horizon_bars
        if entry_i >= len(df) or final_exit_i >= len(df):
            continue
        atr = float(atrs[signal_i])
        if not np.isfinite(atr) or atr <= 0:
            continue

        raw_entry = opens[entry_i] if entry_mode == "next_open" else closes[entry_i]
        entry_time = timestamps[entry_i]
        entry_price = apply_entry_slippage(float(raw_entry), friction)
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
                first_leg_i = entry_i + 6
                second_leg_i = entry_i + 12
                if second_leg_i >= len(df):
                    continue
                exit_legs = [(first_leg_i, 0.5, "scaled_6"), (second_leg_i, 0.5, "scaled_12")]
                exit_reason = "scaled_horizon"
            else:
                exit_legs = [(final_exit_i, 1.0, "horizon")]

        gross_return = 0.0
        exit_fee = 0.0
        weighted_exit_price = 0.0
        weighted_exit_time = timestamps[max(leg[0] for leg in exit_legs)]
        leg_details: list[str] = []
        for exit_i, weight, leg_reason in exit_legs:
            raw_exit = stop if leg_reason == "stop" else closes[exit_i]
            exit_price = apply_exit_slippage(float(raw_exit), friction)
            gross_return += weight * (exit_price / entry_price - 1.0)
            exit_fee += weight * friction.taker_fee_rate
            weighted_exit_price += weight * exit_price
            leg_details.append(f"{leg_reason}:{weight}@{pd.Timestamp(timestamps[exit_i]).isoformat()}")

        net_return = gross_return - entry_fee - exit_fee
        exit_i_for_block = max(leg[0] for leg in exit_legs)
        trades.append(
            {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "entry_mode": entry_mode,
                "model": model.name,
                "allow_stacking": model.allow_stacking,
                "friction_case": friction.name,
                "event_threshold": EVENT_RETURN_THRESHOLD,
                "signal_time": timestamps[signal_i],
                "signal_return": event_returns[signal_i],
                "entry_time": entry_time,
                "exit_time": weighted_exit_time,
                "entry_price": entry_price,
                "exit_price": weighted_exit_price,
                "atr_14_prior": atr,
                "stop": stop,
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
        blocked_until = exit_i_for_block
    return trades


def drawdown_period(returns: pd.Series, times: pd.Series) -> tuple[str | None, str | None, float]:
    if returns.empty:
        return None, None, 0.0
    equity = returns.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    trough_idx = drawdown.idxmin()
    if pd.isna(trough_idx):
        return None, None, 0.0
    peak_slice = equity.loc[:trough_idx]
    peak_idx = peak_slice.idxmax()
    return pd.Timestamp(times.loc[peak_idx]).isoformat(), pd.Timestamp(times.loc[trough_idx]).isoformat(), float(drawdown.loc[trough_idx])


def exposure_stats(group: pd.DataFrame) -> tuple[float, int]:
    changes: list[tuple[pd.Timestamp, int]] = []
    for row in group.itertuples(index=False):
        changes.append((row.entry_time, 1))
        changes.append((row.exit_time, -1))
    active = 0
    max_active = 0
    samples: list[int] = []
    for _, delta in sorted(changes, key=lambda item: (item[0], -item[1])):
        active += delta
        max_active = max(max_active, active)
        samples.append(active)
    return (float(np.mean(samples)) if samples else 0.0, int(max_active))


def summarize_group(group: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_return"]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    years = max((ordered["entry_time"].max() - ordered["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    dd_start, dd_end, dd = drawdown_period(returns, ordered["exit_time"])
    avg_exposure, max_exposure = exposure_stats(ordered)
    by_month = ordered.assign(month=ordered["exit_time"].dt.strftime("%Y-%m")).groupby("month")["net_return"].sum()
    by_symbol = ordered.groupby("symbol")["net_return"].sum()
    trade_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe_like = float((returns.mean() / trade_std) * np.sqrt(len(returns))) if trade_std > 0 else 0.0
    annualized_trade_sharpe = float((returns.mean() / trade_std) * np.sqrt(len(returns) / years)) if trade_std > 0 else 0.0
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "total_trades": int(len(ordered)),
            "trades_per_year": float(len(ordered) / years),
            "average_return_pct": float(returns.mean() * 100.0),
            "median_return_pct": float(returns.median() * 100.0),
            "cumulative_return_units": float(returns.sum()),
            "win_rate": float((returns > 0).mean()),
            "return_volatility_pct": trade_std * 100.0,
            "sharpe_like": sharpe_like,
            "annualized_trade_sharpe_like": annualized_trade_sharpe,
            "max_drawdown_units": dd,
            "max_drawdown_pct": dd * 100.0,
            "worst_drawdown_start": dd_start,
            "worst_drawdown_end": dd_end,
            "max_losing_streak": max_losing_streak(returns),
            "average_win_pct": float(wins.mean() * 100.0) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean() * 100.0) if not losses.empty else 0.0,
            "positive_month_rate": float((by_month > 0).mean()) if not by_month.empty else 0.0,
            "positive_symbol_rate": float((by_symbol > 0).mean()) if not by_symbol.empty else 0.0,
            "best_month": by_month.idxmax(),
            "worst_month": by_month.idxmin(),
            "best_symbol": by_symbol.idxmax(),
            "worst_symbol": by_symbol.idxmin(),
            "best_month_contribution": float(by_month.max() / returns.sum()) if returns.sum() > 0 else np.nan,
            "best_symbol_contribution": float(by_symbol.max() / returns.sum()) if returns.sum() > 0 else np.nan,
            "average_concurrent_positions": avg_exposure,
            "max_concurrent_positions": max_exposure,
        }
    )
    return out


def build_equity_curve(trades: pd.DataFrame, scenario_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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


def build_reports(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "entry_mode",
        "model",
        "allow_stacking",
        "friction_case",
        "event_threshold",
        "horizon_bars",
        "taker_fee_bps_per_side",
        "slippage_bps_per_side",
    ]
    metrics = pd.DataFrame([summarize_group(group, scenario_cols) for _, group in trades.groupby(scenario_cols, sort=False, dropna=False)])
    by_symbol = pd.DataFrame([summarize_group(group, scenario_cols + ["symbol"]) for _, group in trades.groupby(scenario_cols + ["symbol"], sort=False, dropna=False)])
    month_trades = trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m"))
    by_month = pd.DataFrame([summarize_group(group, scenario_cols + ["month"]) for _, group in month_trades.groupby(scenario_cols + ["month"], sort=False, dropna=False)])
    equity = build_equity_curve(trades, scenario_cols)
    return metrics, by_symbol, by_month, equity


def pass_fail_summary(metrics: pd.DataFrame) -> dict[str, object]:
    base = metrics[(metrics["friction_case"] == "base") & (metrics["allow_stacking"] == True)].copy()
    if base.empty:
        return {"minimum_pass": False, "reason": "no base stacked scenarios"}
    best = base.sort_values("cumulative_return_units", ascending=False).iloc[0]
    stress = metrics[
        (metrics["friction_case"] == "stress")
        & (metrics["entry_mode"] == best["entry_mode"])
        & (metrics["model"] == best["model"])
        & (metrics["allow_stacking"] == best["allow_stacking"])
    ]
    stress_return = float(stress["cumulative_return_units"].iloc[0]) if not stress.empty else np.nan
    checks = {
        "base_cumulative_return_positive": float(best["cumulative_return_units"]) > 0,
        "stress_not_destroyed": pd.notna(stress_return) and stress_return > 0,
        "month_concentration_ok": pd.notna(best["best_month_contribution"]) and float(best["best_month_contribution"]) <= 0.40,
        "symbol_concentration_ok": pd.notna(best["best_symbol_contribution"]) and float(best["best_symbol_contribution"]) <= 0.40,
        "drawdown_less_than_cumulative_return": abs(float(best["max_drawdown_units"])) < float(best["cumulative_return_units"]),
    }
    checks["minimum_pass"] = all(checks.values())
    return {"best_base_stacked_scenario": best.to_dict(), "matched_stress_cumulative_return_units": stress_return, "checks": checks}


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
    parser = argparse.ArgumentParser(description="15m extreme down micro-edge accumulation backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "extreme_down_micro_edge")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    symbols = [symbol.upper() for symbol in args.symbols]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded, quality = ensure_data(symbols, args.data_dir, start, end, args.download)

    models = [
        ManagementModel("pure_horizon_12_single_position", horizon_bars=12, allow_stacking=False),
        ManagementModel("pure_horizon_12_stacked", horizon_bars=12, allow_stacking=True),
        ManagementModel("wide_stop_2atr_horizon_12_stacked", horizon_bars=12, stop_atr_mult=2.0, allow_stacking=True),
        ManagementModel("pure_horizon_6_stacked", horizon_bars=6, allow_stacking=True),
        ManagementModel("scaled_exit_6_12_stacked", horizon_bars=12, scaled_exit=True, allow_stacking=True),
    ]
    frictions = [
        Friction("base", taker_fee_rate=0.0004, slippage_rate=0.0005),
        Friction("stress", taker_fee_rate=0.0008, slippage_rate=0.0010),
    ]

    records: list[dict[str, object]] = []
    for model in models:
        for friction in frictions:
            for entry_mode in ["next_close", "next_open"]:
                for symbol, df in loaded.items():
                    records.extend(simulate_symbol(symbol, df, entry_mode, model, friction))

    trades = pd.DataFrame(records)
    if trades.empty:
        raise RuntimeError("No trades generated.")
    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades.sort_values(["model", "friction_case", "entry_mode", "exit_time", "symbol"]).reset_index(drop=True)

    metrics, by_symbol, by_month, equity = build_reports(trades)
    pass_fail = pass_fail_summary(metrics)

    trades_path = args.out_dir / "trade_returns.csv"
    equity_path = args.out_dir / "equity_curve.csv"
    metrics_path = args.out_dir / "cumulative_metrics.csv"
    by_symbol_path = args.out_dir / "by_symbol.csv"
    by_month_path = args.out_dir / "by_month.csv"
    quality_path = args.out_dir / "data_quality_report.csv"
    summary_path = args.out_dir / "phase_summary.json"

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    metrics.sort_values("cumulative_return_units", ascending=False).to_csv(metrics_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_month.to_csv(by_month_path, index=False)
    quality.to_csv(quality_path, index=False)

    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "15m extreme down micro-edge accumulation test.",
        "symbols": symbols,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "event_return_threshold": EVENT_RETURN_THRESHOLD,
        "models": [asdict(model) for model in models],
        "friction": [asdict(friction) for friction in frictions],
        "total_trade_rows": int(len(trades)),
        "scenario_count": int(len(metrics)),
        "reports": {
            "equity_curve_csv": str(equity_path),
            "trade_returns_csv": str(trades_path),
            "cumulative_metrics_csv": str(metrics_path),
            "by_symbol_csv": str(by_symbol_path),
            "by_month_csv": str(by_month_path),
            "data_quality_report_csv": str(quality_path),
            "phase_summary_json": str(summary_path),
        },
        "assumptions": [
            "Event is fixed: 15m close-to-close return <= -2.0%.",
            "Long only.",
            "Signal known only after event candle close.",
            "Entry is next candle close or next candle open.",
            "Stacked models allow every signal to create a new fixed-size position.",
            "Single-position model is included only as a stacking reference.",
            "Returns are fixed-size per signal and additive, not compounded.",
            "No TP, no narrow R-based stop, no ML, no OI, no funding.",
        ],
        "top_10_by_cumulative_return": metrics.sort_values("cumulative_return_units", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(metrics.sort_values("cumulative_return_units", ascending=False).head(12).to_string(index=False))
    print(json.dumps(json_safe(pass_fail["checks"]), indent=2))


if __name__ == "__main__":
    main()
