from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
NON_BTC_SYMBOLS = [symbol for symbol in SYMBOLS if symbol != "BTCUSDT"]
TIMEFRAME = "15m"
START = "2020-01-01T00:00:00+00:00"
END = "2026-05-04T00:00:00+00:00"
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
ATR_PERIOD = 14
VOL_LOOKBACK = 96
BASE_FEE = 0.0004
BASE_SLIPPAGE = 0.0005
STRESS_FEE = 0.0008
STRESS_SLIPPAGE = 0.0010
PANIC_BASELINE = {
    "name": "panic_reversal_extended_primary",
    "trades": 966,
    "cumulative_return_units": 26.8981,
    "avg_return_pct": 2.7845,
    "median_return_pct": 1.7634,
    "win_rate": 0.6460,
    "note": "Reference from management-approved Panic-Reversal extended validation.",
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
class EventDef:
    name: str
    column: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class DislocationDef:
    name: str
    column: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class CapScenario:
    name: str
    concurrent_cap: int | None
    symbol_cap: int | None
    selection_policy: str


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def listing_start(symbol: str, requested_start: pd.Timestamp) -> pd.Timestamp:
    known = parse_utc(LISTING_STARTS.get(symbol, requested_start.isoformat()))
    return max(requested_start, known)


def add_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
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
    out["ret_15m"] = out["close"].pct_change()
    out["atr14"] = true_range.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["atr_pct_prior"] = (out["atr14"] / out["close"]).shift(1)
    out["ret_vol_prior"] = out["ret_15m"].shift(1).rolling(VOL_LOOKBACK, min_periods=48).std()
    return out


def data_quality(symbol: str, df: pd.DataFrame, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> dict[str, object]:
    listing_aware_start = listing_start(symbol, requested_start)
    expected = pd.date_range(start=listing_aware_start, end=requested_end, freq="15min", inclusive="left")
    timestamps = pd.DatetimeIndex(df["timestamp"])
    gaps = timestamps.to_series().diff().dropna()
    expected_delta = pd.Timedelta(minutes=15)
    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "listing_aware_start": listing_aware_start.isoformat(),
        "first_timestamp": timestamps.min().isoformat() if len(timestamps) else None,
        "last_timestamp": timestamps.max().isoformat() if len(timestamps) else None,
        "rows": int(len(df)),
        "expected_rows_from_listing_start": int(len(expected)),
        "true_internal_missing_rows": int(len(expected.difference(timestamps))),
        "duplicate_rows": int(df["timestamp"].duplicated().sum()),
        "large_gap_count": int((gaps > expected_delta).sum()) if not gaps.empty else 0,
        "max_gap_minutes": float(gaps.max().total_seconds() / 60.0) if not gaps.empty else 0.0,
        "zero_volume_rows": int((df["volume"] <= 0).sum()),
    }


def load_frames(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    quality_rows = []
    for symbol in SYMBOLS:
        path = data_dir / TIMEFRAME / f"{symbol}.csv"
        raw = load_ohlcv(path)
        raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)].reset_index(drop=True)
        quality_rows.append(data_quality(symbol, raw, start, end))
        frames[symbol] = add_symbol_features(raw)
    return frames, pd.DataFrame(quality_rows)


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = pd.concat(
        [frame.set_index("timestamp")["ret_15m"].rename(symbol) for symbol, frame in frames.items()],
        axis=1,
    ).sort_index()
    context = pd.DataFrame(index=returns.index)
    context["basket_median_return"] = returns[SYMBOLS].median(axis=1, skipna=True)
    context["btc_return"] = returns["BTCUSDT"]
    for threshold in [0.01, 0.02]:
        context[f"breadth_down_{int(threshold * 100)}pct"] = (returns[SYMBOLS] <= -threshold).sum(axis=1)
    context = context.reset_index()

    rank_rows = []
    for timestamp, row in returns[NON_BTC_SYMBOLS].iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        ordered = valid.sort_values()
        for rank, (symbol, value) in enumerate(ordered.items(), start=1):
            rank_rows.append({"timestamp": timestamp, "symbol": symbol, "underperformance_rank": rank, "symbol_return": value})
    ranks = pd.DataFrame(rank_rows)

    rows = []
    for symbol in NON_BTC_SYMBOLS:
        frame = frames[symbol][["timestamp", "open", "close", "ret_15m", "atr_pct_prior", "ret_vol_prior"]].copy()
        frame["symbol"] = symbol
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(context, on="timestamp", how="left").merge(ranks[["timestamp", "symbol", "underperformance_rank"]], on=["timestamp", "symbol"], how="left")
    panel["rel_to_btc"] = panel["ret_15m"] - panel["btc_return"]
    panel["rel_to_basket"] = panel["ret_15m"] - panel["basket_median_return"]
    panel["symbol_return_zscore"] = panel["ret_15m"] / panel["ret_vol_prior"]
    panel["rel_btc_atr_equiv"] = panel["rel_to_btc"] / panel["atr_pct_prior"]
    panel["rel_basket_atr_equiv"] = panel["rel_to_basket"] / panel["atr_pct_prior"]
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def event_defs() -> list[EventDef]:
    events = []
    for threshold in [-0.01, -0.02, -0.03]:
        events.append(EventDef(f"basket_return_le_{abs(threshold):.0%}", "basket_median_return", "<=", threshold))
        events.append(EventDef(f"btc_return_le_{abs(threshold):.0%}", "btc_return", "<=", threshold))
    for down_threshold in [1, 2]:
        for count in [4, 5, 6]:
            events.append(EventDef(f"breadth_down_{down_threshold}pct_ge_{count}", f"breadth_down_{down_threshold}pct", ">=", count))
    return events


def dislocation_defs() -> list[DislocationDef]:
    defs = []
    for threshold in [-0.01, -0.02, -0.03]:
        defs.append(DislocationDef(f"rel_btc_le_{abs(threshold):.0%}", "rel_to_btc", "<=", threshold))
        defs.append(DislocationDef(f"rel_basket_le_{abs(threshold):.0%}", "rel_to_basket", "<=", threshold))
    defs.extend(
        [
            DislocationDef("symbol_zscore_le_-1", "symbol_return_zscore", "<=", -1.0),
            DislocationDef("symbol_zscore_le_-2", "symbol_return_zscore", "<=", -2.0),
            DislocationDef("rel_btc_le_-1_atr_equiv", "rel_btc_atr_equiv", "<=", -1.0),
            DislocationDef("rel_basket_le_-1_atr_equiv", "rel_basket_atr_equiv", "<=", -1.0),
        ]
    )
    for rank in [1, 2, 3]:
        defs.append(DislocationDef(f"bottom_{rank}_performer", "underperformance_rank", "<=", rank))
    return defs


def compare(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == "<=":
        return series <= threshold
    if operator == ">=":
        return series >= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def build_signal_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in event_defs():
        event_mask = compare(panel[event.column], event.operator, event.threshold)
        for dislocation in dislocation_defs():
            mask = event_mask & compare(panel[dislocation.column], dislocation.operator, dislocation.threshold)
            sample = panel.loc[mask].copy()
            if sample.empty:
                continue
            sample["event_definition"] = event.name
            sample["event_column"] = event.column
            sample["event_threshold"] = event.threshold
            sample["dislocation_definition"] = dislocation.name
            sample["dislocation_column"] = dislocation.column
            sample["dislocation_threshold"] = dislocation.threshold
            rows.append(sample)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def apply_entry_slippage(price: float, friction: Friction) -> float:
    return price * (1.0 + friction.slippage_rate)


def apply_exit_slippage(price: float, friction: Friction) -> float:
    return price * (1.0 - friction.slippage_rate)


def build_price_lookup(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {symbol: frame[["timestamp", "open", "close"]].reset_index(drop=True) for symbol, frame in frames.items()}


def simulate_trades(signals: pd.DataFrame, prices: dict[str, pd.DataFrame], entry_mode: str, model: ExitModel, friction: Friction) -> pd.DataFrame:
    records = []
    bar_delta = pd.Timedelta(minutes=15)
    for symbol, group in signals.groupby("symbol", sort=False):
        price_df = prices[symbol].copy().reset_index(drop=True)
        price_df["bar_index"] = np.arange(len(price_df))
        lookup = price_df[["timestamp", "bar_index"]].merge(group.reset_index(drop=True), on="timestamp", how="inner")
        if lookup.empty:
            continue
        opens = price_df["open"].to_numpy(dtype=float)
        closes = price_df["close"].to_numpy(dtype=float)
        timestamps = price_df["timestamp"].to_numpy()
        for row in lookup.itertuples(index=False):
            signal_i = int(row.bar_index)
            entry_i = signal_i + 1
            final_exit_i = entry_i + model.horizon_bars
            if entry_i >= len(price_df) or final_exit_i >= len(price_df):
                continue
            raw_entry = closes[entry_i] if entry_mode == "next_close" else opens[entry_i]
            entry_price = apply_entry_slippage(float(raw_entry), friction)
            if model.scaled_exit:
                exit_legs = [(entry_i + 6, 0.5), (entry_i + 12, 0.5)]
            else:
                exit_legs = [(final_exit_i, 1.0)]
            gross_return = 0.0
            exit_fee = 0.0
            weighted_exit_price = 0.0
            for exit_i, weight in exit_legs:
                exit_price = apply_exit_slippage(float(closes[exit_i]), friction)
                gross_return += weight * (exit_price / entry_price - 1.0)
                exit_fee += weight * friction.taker_fee_rate
                weighted_exit_price += weight * exit_price
            exit_i = max(item[0] for item in exit_legs)
            net_return = gross_return - friction.taker_fee_rate - exit_fee
            signal_time = pd.Timestamp(timestamps[signal_i])
            entry_candle_open_time = pd.Timestamp(timestamps[entry_i])
            entry_time = entry_candle_open_time + bar_delta if entry_mode == "next_close" else entry_candle_open_time
            records.append(
                {
                    "symbol": symbol,
                    "signal_time": signal_time,
                    "signal_close_time": signal_time + bar_delta,
                    "entry_candle_open_time": entry_candle_open_time,
                    "entry_time": entry_time,
                    "entry_time_semantic": "next_candle_close" if entry_mode == "next_close" else "next_candle_open",
                    "exit_time": pd.Timestamp(timestamps[exit_i]) + bar_delta,
                    "entry_mode": entry_mode,
                    "exit_model": model.name,
                    "horizon_bars": model.horizon_bars,
                    "friction_case": friction.name,
                    "taker_fee_bps_per_side": friction.taker_fee_rate * 10_000,
                    "slippage_bps_per_side": friction.slippage_rate * 10_000,
                    "event_definition": row.event_definition,
                    "dislocation_definition": row.dislocation_definition,
                    "event_return": float(row.ret_15m),
                    "event_return_pct": float(row.ret_15m) * 100.0,
                    "btc_return": float(row.btc_return),
                    "basket_median_return": float(row.basket_median_return),
                    "rel_to_btc": float(row.rel_to_btc),
                    "rel_to_basket": float(row.rel_to_basket),
                    "symbol_return_zscore": float(row.symbol_return_zscore),
                    "underperformance_rank": int(row.underperformance_rank),
                    "breadth_down_1pct": int(row.breadth_down_1pct),
                    "breadth_down_2pct": int(row.breadth_down_2pct),
                    "entry_price": entry_price,
                    "exit_price": weighted_exit_price,
                    "gross_return": gross_return,
                    "fee_return": friction.taker_fee_rate + exit_fee,
                    "net_return": net_return,
                    "net_return_pct": net_return * 100.0,
                    "abs_dislocation": abs(float(getattr(row, "rel_to_basket"))),
                }
            )
    return pd.DataFrame(records)


def max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return 0.0
    equity = series.cumsum()
    return float((equity - equity.cummax()).min())


def concentration(series: pd.Series) -> float:
    total = float(series.sum())
    if total <= 0 or series.empty:
        return np.nan
    return float(series.max() / total)


def summarize_group(group: pd.DataFrame, group_cols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    ordered = group.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["net_return"]
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    monthly = ordered.assign(month=ordered["exit_time"].dt.strftime("%Y-%m")).groupby("month")["net_return"].sum()
    symbol = ordered.groupby("symbol")["net_return"].sum()
    row = {col: ordered[col].iloc[0] for col in group_cols}
    row.update(
        {
            "total_trades": int(len(ordered)),
            "trades_year": float(len(ordered) / years),
            "avg_return_pct": float(returns.mean() * 100.0),
            "median_return_pct": float(returns.median() * 100.0),
            "win_rate": float((returns > 0).mean()),
            "cumulative_return_units": float(returns.sum()),
            "max_drawdown_units": max_drawdown(returns),
            "sharpe_like": float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf,
            "best_month_contribution": concentration(monthly),
            "best_symbol_contribution": concentration(symbol),
            "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "positive_symbol_rate": float((symbol > 0).mean()) if len(symbol) else np.nan,
        }
    )
    return row


def summarize_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = ["event_definition", "dislocation_definition", "entry_mode", "exit_model", "friction_case"]
    overall_rows = [summarize_group(group, scenario_cols, start, end) for _, group in trades.groupby(scenario_cols, sort=False, dropna=False)]
    by_symbol_rows = [summarize_group(group, scenario_cols + ["symbol"], start, end) for _, group in trades.groupby(scenario_cols + ["symbol"], sort=False, dropna=False)]
    by_month_source = trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m"))
    by_month_rows = [summarize_group(group, scenario_cols + ["month"], start, end) for _, group in by_month_source.groupby(scenario_cols + ["month"], sort=False, dropna=False)]
    overall = pd.DataFrame(overall_rows)
    by_symbol = pd.DataFrame(by_symbol_rows)
    by_month = pd.DataFrame(by_month_rows)
    return overall, by_symbol, by_month, pd.DataFrame()


def build_event_scan(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizons = [1, 3, 6, 12]
    for (event_definition, dislocation_definition), group in signals.groupby(["event_definition", "dislocation_definition"], sort=False):
        row = {
            "event_definition": event_definition,
            "dislocation_definition": dislocation_definition,
            "occurrences": int(len(group)),
            "symbols": int(group["symbol"].nunique()),
            "first_signal": group["timestamp"].min().isoformat(),
            "last_signal": group["timestamp"].max().isoformat(),
            "avg_event_return_pct": float(group["ret_15m"].mean() * 100.0),
            "avg_rel_to_btc_pct": float(group["rel_to_btc"].mean() * 100.0),
            "avg_rel_to_basket_pct": float(group["rel_to_basket"].mean() * 100.0),
        }
        for horizon in horizons:
            col = f"forward_{horizon}_return"
            if col in group:
                row[f"avg_forward_{horizon}_pct"] = float(group[col].mean() * 100.0)
                row[f"median_forward_{horizon}_pct"] = float(group[col].median() * 100.0)
                row[f"win_rate_forward_{horizon}"] = float((group[col] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def add_forward_returns(signals: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    horizons = [1, 3, 6, 12]
    for symbol, group in signals.groupby("symbol", sort=False):
        close = frames[symbol][["timestamp", "close"]].copy().reset_index(drop=True)
        close["bar_index"] = np.arange(len(close))
        merged = close[["timestamp", "bar_index", "close"]].rename(columns={"close": "signal_close_price"}).merge(group, on="timestamp", how="inner")
        closes = close["close"].to_numpy(dtype=float)
        for horizon in horizons:
            idx = merged["bar_index"].to_numpy() + horizon
            values = np.full(len(merged), np.nan)
            valid = idx < len(closes)
            values[valid] = closes[idx[valid]] / merged.loc[valid, "signal_close_price"].to_numpy(dtype=float) - 1.0
            merged[f"forward_{horizon}_return"] = values
        out.append(merged.drop(columns=["bar_index", "close"]))
    return pd.concat(out, ignore_index=True)


def build_execution_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_definition", "dislocation_definition", "exit_model", "friction_case"]
    open_rows = overall[overall["entry_mode"] == "next_open"].set_index(keys)
    close_rows = overall[overall["entry_mode"] == "next_close"].set_index(keys)
    comparison = open_rows[["total_trades", "avg_return_pct", "median_return_pct", "cumulative_return_units"]].join(
        close_rows[["total_trades", "avg_return_pct", "median_return_pct", "cumulative_return_units"]],
        lsuffix="_next_open",
        rsuffix="_next_close",
        how="outer",
    ).reset_index()
    comparison["avg_return_delta_close_minus_open_pct"] = comparison["avg_return_pct_next_close"] - comparison["avg_return_pct_next_open"]
    comparison["cum_return_delta_close_minus_open"] = comparison["cumulative_return_units_next_close"] - comparison["cumulative_return_units_next_open"]
    return comparison


def order_candidates(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(["abs_dislocation", "event_return", "symbol"], ascending=[False, True, True])


def simulate_caps(base_trades: pd.DataFrame, scenario: CapScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: list[tuple[pd.Timestamp, str]] = []
    accepted = []
    skipped = []
    for signal_time, group in base_trades.sort_values(["signal_time", "symbol"]).groupby("signal_time", sort=True):
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > signal_time]
        active_by_symbol: dict[str, int] = {}
        for _, symbol in active:
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
        for trade in order_candidates(group).to_dict(orient="records"):
            symbol = str(trade["symbol"])
            reason = None
            if scenario.concurrent_cap is not None and len(active) >= scenario.concurrent_cap:
                reason = "concurrent_cap"
            elif scenario.symbol_cap is not None and active_by_symbol.get(symbol, 0) >= scenario.symbol_cap:
                reason = "symbol_cap"
            if reason:
                skipped.append({**trade, "cap_scenario": scenario.name, "skip_reason": reason, "active_positions": len(active)})
                continue
            accepted.append({**trade, "cap_scenario": scenario.name})
            active.append((trade["exit_time"], symbol))
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
    return pd.DataFrame(accepted), pd.DataFrame(skipped)


def cap_comparison(trades: pd.DataFrame, best_key: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = [
        CapScenario("unlimited", None, None, "largest_dislocation"),
        CapScenario("cap10_symbol2_largest_dislocation", 10, 2, "largest_dislocation"),
    ]
    subset = trades.copy()
    for key, value in best_key.items():
        subset = subset[subset[key] == value]
    rows = []
    skipped_rows = []
    for scenario in scenarios:
        accepted, skipped = (subset.copy(), pd.DataFrame()) if scenario.concurrent_cap is None else simulate_caps(subset, scenario)
        if accepted.empty:
            continue
        summary = summarize_group(accepted, [], start, end)
        summary.update(
            {
                "cap_scenario": scenario.name,
                "trades_taken": int(len(accepted)),
                "trades_skipped": int(len(skipped)),
                "skip_rate": float(len(skipped) / (len(accepted) + len(skipped))) if len(accepted) + len(skipped) else 0.0,
            }
        )
        rows.append(summary)
        if not skipped.empty:
            skipped_rows.append(skipped.head(1000))
    skipped_sample = pd.concat(skipped_rows, ignore_index=True) if skipped_rows else pd.DataFrame()
    return pd.DataFrame(rows), skipped_sample


def select_best_candidates(overall: pd.DataFrame) -> pd.DataFrame:
    sample = overall[
        (overall["entry_mode"] == "next_close")
        & (overall["exit_model"] == "fixed_horizon_12")
        & (overall["friction_case"] == "base")
        & (overall["total_trades"] >= 300)
        & (overall["avg_return_pct"] > 0)
        & (overall["median_return_pct"] > 0)
    ].copy()
    if sample.empty:
        return sample
    return sample.sort_values(["avg_return_pct", "cumulative_return_units", "total_trades"], ascending=[False, False, False]).head(20)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-Asset Dislocation Reversion Phase 1 scan/backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "binance_um_ohlcv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "cross_asset_dislocation")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames, quality = load_frames(args.data_dir, start, end)
    panel = build_panel(frames)
    signals = build_signal_events(panel)
    signals = add_forward_returns(signals, frames)
    event_scan = build_event_scan(signals)

    frictions = [Friction("base", BASE_FEE, BASE_SLIPPAGE), Friction("stress", STRESS_FEE, STRESS_SLIPPAGE)]
    models = [ExitModel("fixed_horizon_6", 6), ExitModel("fixed_horizon_12", 12), ExitModel("scaled_exit_6_12", 12, scaled_exit=True)]
    prices = build_price_lookup(frames)
    trade_parts = []
    for entry_mode in ["next_close", "next_open"]:
        for model in models:
            for friction in frictions:
                trade_parts.append(simulate_trades(signals, prices, entry_mode, model, friction))
    trades = pd.concat(trade_parts, ignore_index=True)
    for col in ["signal_time", "signal_close_time", "entry_candle_open_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)

    overall, by_symbol, by_month, _ = summarize_trades(trades, start, end)
    execution = build_execution_comparison(overall)
    best = select_best_candidates(overall)
    if best.empty:
        best_key = {
            "event_definition": overall.sort_values(["avg_return_pct", "total_trades"], ascending=[False, False]).iloc[0]["event_definition"],
            "dislocation_definition": overall.sort_values(["avg_return_pct", "total_trades"], ascending=[False, False]).iloc[0]["dislocation_definition"],
            "entry_mode": "next_close",
            "exit_model": "fixed_horizon_12",
            "friction_case": "base",
        }
    else:
        row = best.iloc[0]
        best_key = {
            "event_definition": row["event_definition"],
            "dislocation_definition": row["dislocation_definition"],
            "entry_mode": row["entry_mode"],
            "exit_model": row["exit_model"],
            "friction_case": row["friction_case"],
        }
    caps, skipped = cap_comparison(trades, best_key, start, end)

    paths = {
        "event_scan_summary": args.out_dir / "event_scan_summary.csv",
        "feasibility_results": args.out_dir / "feasibility_results.csv",
        "trade_returns_sample": args.out_dir / "trade_returns_sample.csv",
        "by_symbol": args.out_dir / "by_symbol.csv",
        "by_month": args.out_dir / "by_month.csv",
        "execution_comparison": args.out_dir / "execution_comparison.csv",
        "cap_comparison": args.out_dir / "cap_comparison.csv",
        "data_quality_report": args.out_dir / "data_quality_report.csv",
        "phase1_summary": args.out_dir / "phase1_summary.json",
    }
    event_scan.to_csv(paths["event_scan_summary"], index=False)
    overall.to_csv(paths["feasibility_results"], index=False)
    trades.head(5000).to_csv(paths["trade_returns_sample"], index=False)
    by_symbol.to_csv(paths["by_symbol"], index=False)
    by_month.to_csv(paths["by_month"], index=False)
    execution.to_csv(paths["execution_comparison"], index=False)
    caps.to_csv(paths["cap_comparison"], index=False)
    quality.to_csv(paths["data_quality_report"], index=False)
    if not skipped.empty:
        skipped.to_csv(args.out_dir / "skipped_signals_sample.csv", index=False)

    stress_match = pd.DataFrame()
    if best_key:
        stress_match = overall[
            (overall["event_definition"] == best_key["event_definition"])
            & (overall["dislocation_definition"] == best_key["dislocation_definition"])
            & (overall["entry_mode"] == best_key["entry_mode"])
            & (overall["exit_model"] == best_key["exit_model"])
            & (overall["friction_case"] == "stress")
        ]
    best_rows = best.to_dict(orient="records")
    pass_candidates = best[
        (best["median_return_pct"] > 0)
        & (best["best_month_contribution"].fillna(1.0) < 0.7523)
        & (best["avg_return_pct"] >= PANIC_BASELINE["avg_return_pct"])
    ] if not best.empty else pd.DataFrame()
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Cross-Asset Dislocation Reversion Phase 1 event scan and feasibility backtest.",
        "symbols": SYMBOLS,
        "non_btc_traded_symbols": NON_BTC_SYMBOLS,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "event_definitions_tested": [asdict(item) for item in event_defs()],
        "dislocation_definitions_tested": [asdict(item) for item in dislocation_defs()],
        "execution": {"entry_modes": ["next_close", "next_open"], "market_execution_only": True, "limit_orders": False},
        "exit_models": [asdict(item) for item in models],
        "friction": [asdict(item) for item in frictions],
        "total_signal_rows": int(len(signals)),
        "total_trade_rows": int(len(trades)),
        "best_candidates_next_close_h12_base": best_rows,
        "best_key_for_cap_test": best_key,
        "matched_stress_for_best_key": stress_match.to_dict(orient="records"),
        "cap_comparison": caps.to_dict(orient="records"),
        "panic_reversal_baseline": PANIC_BASELINE,
        "pass_fail": {
            "positive_candidate_found": bool(not best.empty),
            "candidate_beats_panic_reversal_avg_return": bool(not pass_candidates.empty),
            "criteria": [
                ">=300 trades",
                "avg and median return positive after base friction",
                "stress remains positive or near-flat",
                "month concentration better than Panic Reversal",
                "capital efficiency likely better than Panic Reversal",
                "execution is clean",
            ],
            "phase1_pass": bool(not pass_candidates.empty),
            "note": "Positive dislocation candidates exist, but Phase 1 pass also requires evidence of better return potential/capital efficiency than Panic-Reversal.",
        },
        "reports": {key: str(path) for key, path in paths.items()},
    }
    paths["phase1_summary"].write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print("Top next_close fixed_horizon_12 base candidates:")
    print(best.head(10).to_string(index=False) if not best.empty else "No candidate met base filters.")
    print("Cap comparison:")
    print(caps.to_string(index=False) if not caps.empty else "No cap comparison generated.")


if __name__ == "__main__":
    main()
