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
START = "2025-05-04T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
ATR_PERIOD = 14
ENTRY_INTERVAL = 50


@dataclass(frozen=True)
class Friction:
    name: str
    taker_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class ManagementModel:
    name: str
    target_r: float | None
    max_hold_15m: int
    max_hold_1h: int
    trail_to_be_at_r: float | None = None
    tighten_after_bars_15m: int | None = None
    tighten_after_bars_1h: int | None = None
    tighten_stop_to_r: float | None = None
    tighten_on_favorable_r: float | None = None
    early_exit_bars_15m: int | None = None
    early_exit_bars_1h: int | None = None
    early_exit_min_favorable_r: float | None = None

    def max_hold(self, timeframe: str) -> int:
        return self.max_hold_15m if timeframe == "15m" else self.max_hold_1h

    def tighten_after_bars(self, timeframe: str) -> int | None:
        return self.tighten_after_bars_15m if timeframe == "15m" else self.tighten_after_bars_1h

    def early_exit_bars(self, timeframe: str) -> int | None:
        return self.early_exit_bars_15m if timeframe == "15m" else self.early_exit_bars_1h


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


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def data_quality(symbol: str, timeframe: str, df: pd.DataFrame, start: datetime, end: datetime) -> dict[str, object]:
    expected = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=interval_to_pandas_freq(timeframe), inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    missing = expected.difference(timestamps)
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(milliseconds=interval_to_ms(timeframe))
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
        "duplicate_rows": int(df["timestamp"].duplicated().sum()),
        "large_gap_count": int((gaps > expected_delta).sum()),
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
                needs_download = not (
                    probe["timestamp"].min().to_pydatetime() <= start
                    and probe["timestamp"].max().to_pydatetime() >= end - pd.Timedelta(milliseconds=interval_to_ms(timeframe))
                )
            if needs_download:
                print(f"Downloading {symbol} {timeframe}")
                download_klines(symbol, timeframe, start, end, path)
            df = load_ohlcv(path)
            df = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end))].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"No OHLCV rows for {symbol} {timeframe}")
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
    out["atr_14_prior"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().shift(1)
    out["prior_return"] = out["close"].pct_change().shift(1)
    return out


def apply_entry_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 + friction.slippage_rate) if side == "long" else raw_price * (1.0 - friction.slippage_rate)


def apply_exit_slippage(raw_price: float, side: str, friction: Friction) -> float:
    return raw_price * (1.0 - friction.slippage_rate) if side == "long" else raw_price * (1.0 + friction.slippage_rate)


def r_multiple(entry: float, exit_price: float, side: str, risk: float) -> float:
    return (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk


def stop_for_r(entry: float, risk: float, side: str, r_level: float) -> float:
    return entry + r_level * risk if side == "long" else entry - r_level * risk


def simulate_trade_path(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    entry_i: int,
    side: str,
    model: ManagementModel,
    timeframe: str,
    friction: Friction,
) -> dict[str, object] | None:
    raw_entry = float(opens[entry_i])
    entry = apply_entry_slippage(raw_entry, side, friction)
    risk = float(atrs[entry_i - 1])
    if not math.isfinite(risk) or risk <= 0:
        return None
    stop = entry - risk if side == "long" else entry + risk
    target = None if model.target_r is None else stop_for_r(entry, risk, side, model.target_r)
    max_hold = model.max_hold(timeframe)
    last_i = min(entry_i + max_hold, len(opens) - 1)
    if entry_i > last_i:
        return None
    raw_exit = float(closes[last_i])
    exit_i = last_i
    exit_reason = "time_stop"
    moved_to_be = False
    tightened = False

    for j in range(entry_i, last_i + 1):
        high = float(highs[j])
        low = float(lows[j])
        close = float(closes[j])
        favorable_r = (high - entry) / risk if side == "long" else (entry - low) / risk

        if model.trail_to_be_at_r is not None and favorable_r >= model.trail_to_be_at_r:
            stop = max(stop, entry) if side == "long" else min(stop, entry)
            moved_to_be = True

        tighten_after = model.tighten_after_bars(timeframe)
        if model.tighten_stop_to_r is not None and (
            (tighten_after is not None and j - entry_i + 1 >= tighten_after)
            or (model.tighten_on_favorable_r is not None and favorable_r >= model.tighten_on_favorable_r)
        ):
            tightened_stop = stop_for_r(entry, risk, side, model.tighten_stop_to_r)
            stop = max(stop, tightened_stop) if side == "long" else min(stop, tightened_stop)
            tightened = True

        early_exit_bars = model.early_exit_bars(timeframe)
        if (
            early_exit_bars is not None
            and model.early_exit_min_favorable_r is not None
            and j - entry_i + 1 >= early_exit_bars
            and favorable_r < model.early_exit_min_favorable_r
        ):
            raw_exit = close
            exit_i = j
            exit_reason = "early_time"
            break

        if side == "long":
            hit_stop = low <= stop
            hit_target = target is not None and high >= target
        else:
            hit_stop = high >= stop
            hit_target = target is not None and low <= target
        if hit_stop:
            raw_exit = stop
            exit_i = j
            exit_reason = "stop"
            break
        if hit_target:
            raw_exit = float(target)
            exit_i = j
            exit_reason = "target"
            break

    exit_price = apply_exit_slippage(raw_exit, side, friction)
    gross_r = r_multiple(entry, exit_price, side, risk)
    fee_r = friction.taker_fee_rate * (entry + exit_price) / risk
    return {
        "entry_time": timestamps[entry_i],
        "exit_time": timestamps[exit_i],
        "side": side,
        "raw_entry": raw_entry,
        "entry": entry,
        "raw_exit": raw_exit,
        "exit": exit_price,
        "risk": risk,
        "stop_final": stop,
        "target": target,
        "gross_r": gross_r,
        "fee_r": fee_r,
        "net_r": gross_r - fee_r,
        "exit_reason": exit_reason,
        "moved_to_be": moved_to_be,
        "tightened": tightened,
        "max_hold_bars": max_hold,
    }


def simulate_symbol(
    symbol: str,
    timeframe: str,
    rows: pd.DataFrame,
    entry_logic: str,
    model: ManagementModel,
    friction: Friction,
    interval: int,
) -> list[dict[str, object]]:
    trades: list[dict[str, object]] = []
    timestamps = rows["timestamp"].to_numpy()
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    atrs = rows["atr_14_prior"].to_numpy(dtype=float)
    prior_returns = rows["prior_return"].to_numpy(dtype=float)
    i = max(ATR_PERIOD + 1, interval)
    while i < len(opens) - 1:
        signal_i = i
        entry_i = signal_i + 1
        if entry_logic == "periodic_long":
            side = "long"
        elif entry_logic == "prior_return_sign":
            prior_return = float(prior_returns[signal_i])
            if not math.isfinite(prior_return) or prior_return == 0:
                i += interval
                continue
            side = "long" if prior_return > 0 else "short"
        else:
            raise ValueError(f"Unsupported entry logic: {entry_logic}")
        result = simulate_trade_path(timestamps, opens, highs, lows, closes, atrs, entry_i, side, model, timeframe, friction)
        if result is not None:
            result.update(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "entry_logic": entry_logic,
                    "management_model": model.name,
                    "friction_case": friction.name,
                    "signal_time": timestamps[signal_i],
                    "entry_interval": interval,
                    "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                    "slippage_bps_per_side": friction.slippage_rate * 10_000,
                }
            )
            trades.append(result)
        i += interval
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
    out = work.groupby(group_cols, sort=False).agg(
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
    scenario_cols = ["timeframe", "entry_logic", "management_model", "friction_case", "entry_interval", "taker_fee_bps_per_side", "slippage_bps_per_side", "max_hold_bars"]
    if trades.empty:
        empty = fast_group_metrics(trades, scenario_cols, include_path_metrics=True)
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    overall = fast_group_metrics(trades, scenario_cols, include_path_metrics=True)
    symbol_contrib = contribution_shares(trades, scenario_cols, "symbol")
    month_trades = trades.assign(month=trades["exit_time"].dt.to_period("M").astype(str))
    month_contrib = contribution_shares(month_trades, scenario_cols, "month")
    overall = overall.merge(symbol_contrib, on=scenario_cols, how="left").merge(month_contrib, on=scenario_cols, how="left")
    overall = overall.sort_values(["timeframe", "entry_logic", "friction_case", "ev_per_trade_r"], ascending=[True, True, True, False])
    by_symbol = fast_group_metrics(trades, scenario_cols + ["symbol"], include_path_metrics=False).sort_values(["timeframe", "entry_logic", "management_model", "symbol"])
    by_month = fast_group_metrics(month_trades, scenario_cols + ["month"], include_path_metrics=False).sort_values(["timeframe", "entry_logic", "management_model", "month"])
    comparison_keys = ["timeframe", "entry_logic", "friction_case", "entry_interval"]
    comparison = overall.pivot_table(index=comparison_keys, columns="management_model", values=["total_trades", "ev_per_trade_r", "total_ev_r", "max_drawdown_r"], aggfunc="first")
    comparison.columns = ["_".join(map(str, col)).strip() for col in comparison.columns.to_flat_index()]
    comparison = comparison.reset_index()
    return overall, by_symbol, by_month, comparison


def management_models() -> list[ManagementModel]:
    return [
        ManagementModel("fixed_1r_2r", target_r=2.0, max_hold_15m=32, max_hold_1h=24),
        ManagementModel("trail_be_after_1r", target_r=None, max_hold_15m=32, max_hold_1h=24, trail_to_be_at_r=1.0),
        ManagementModel("tighten_after_move", target_r=2.0, max_hold_15m=32, max_hold_1h=24, tighten_after_bars_15m=8, tighten_after_bars_1h=6, tighten_stop_to_r=0.25, tighten_on_favorable_r=0.75),
        ManagementModel("early_no_move", target_r=2.0, max_hold_15m=32, max_hold_1h=24, early_exit_bars_15m=8, early_exit_bars_1h=6, early_exit_min_favorable_r=0.5),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entry-agnostic trade management edge test.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "trade_management_test")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--entry-interval", type=int, default=ENTRY_INTERVAL)
    parser.add_argument("--trades-sample-size", type=int, default=5000)
    return parser.parse_args()


def pass_fail_summary(overall: pd.DataFrame) -> dict[str, object]:
    if overall.empty:
        return {"overall_pass": False, "reason": "no trades"}
    base = overall[overall["friction_case"] == "base"]
    stress = overall[overall["friction_case"] == "stress"]
    best = base.sort_values("ev_per_trade_r", ascending=False).iloc[0]
    stress_match = stress[
        (stress["timeframe"] == best["timeframe"])
        & (stress["entry_logic"] == best["entry_logic"])
        & (stress["management_model"] == best["management_model"])
        & (stress["entry_interval"] == best["entry_interval"])
    ]
    stress_ev = float(stress_match["ev_per_trade_r"].iloc[0]) if not stress_match.empty else np.nan
    symbol_share = float(best["best_symbol_ev_share"]) if pd.notna(best["best_symbol_ev_share"]) else np.nan
    month_share = float(best["best_month_ev_share"]) if pd.notna(best["best_month_ev_share"]) else np.nan
    checks = {
        "ev_trade_ge_0_2r_base": float(best["ev_per_trade_r"]) >= 0.2,
        "ev_trade_ge_0_3r_base": float(best["ev_per_trade_r"]) >= 0.3,
        "stress_ev_not_below_0": pd.notna(stress_ev) and stress_ev >= 0,
        "symbol_concentration_ok": pd.notna(symbol_share) and symbol_share <= 0.40,
        "month_concentration_ok": pd.notna(month_share) and month_share <= 0.40,
    }
    checks["overall_pass"] = checks["ev_trade_ge_0_2r_base"] and checks["stress_ev_not_below_0"] and checks["symbol_concentration_ok"] and checks["month_concentration_ok"]
    return {"best_base_scenario": best.to_dict(), "matched_stress_ev_per_trade_r": stress_ev, "checks": checks}


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
    frictions = [Friction("base", 0.0004, 0.0005), Friction("stress", 0.0008, 0.0010)]
    entry_logics = ["periodic_long", "prior_return_sign"]
    records: list[dict[str, object]] = []
    for timeframe in timeframes:
        for entry_logic in entry_logics:
            for model in management_models():
                for friction in frictions:
                    for symbol in symbols:
                        records.extend(simulate_symbol(symbol, timeframe, featured[(symbol, timeframe)], entry_logic, model, friction, args.entry_interval))
    trades = pd.DataFrame(records)
    if not trades.empty:
        for col in ["signal_time", "entry_time", "exit_time"]:
            trades[col] = pd.to_datetime(trades[col], utc=True)
        trades = trades.sort_values(["timeframe", "entry_logic", "management_model", "friction_case", "exit_time", "symbol"]).reset_index(drop=True)
    overall, by_symbol, by_month, comparison = build_summaries(trades)
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
        "scope": "Entry-agnostic trade management edge test.",
        "symbols": symbols,
        "timeframes": timeframes,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "entry_interval": args.entry_interval,
        "entry_logics": entry_logics,
        "management_models": [asdict(model) for model in management_models()],
        "total_trade_rows": int(trades.shape[0]),
        "scenario_count": int(overall.shape[0]),
        "friction": [asdict(friction) for friction in frictions],
        "assumptions": [
            "No ML and no signal optimization; entries are intentionally weak baselines.",
            "Entry is next candle open only, market-style.",
            "R is one ATR(14) from prior closed candle.",
            "If target and stop are both touched in a candle, stop wins.",
            "Management models are tested against the same entry schedule and friction.",
        ],
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
        "top_10_by_ev_per_trade": overall.sort_values("ev_per_trade_r", ascending=False).head(10).to_dict(orient="records"),
        "pass_fail": pass_fail,
    }
    write_json(summary_path, summary)
    print(f"Wrote trade management reports to {args.out_dir.resolve()}")
    if not overall.empty:
        print(overall.sort_values("ev_per_trade_r", ascending=False).head(12).to_string(index=False))
    print(json.dumps(pass_fail.get("checks", pass_fail), indent=2, default=str))


if __name__ == "__main__":
    main()
