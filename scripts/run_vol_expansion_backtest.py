from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    timeframe: str = "15m"
    breakout_lookback: int = 48
    atr_lookback: int = 48
    range_median_lookback: int = 96
    volume_median_lookback: int = 96
    range_expansion_mult: float = 1.6
    volume_expansion_mult: float = 1.4
    min_body_fraction: float = 0.55
    max_range_atr_mult: float = 2.8
    max_extension_atr_mult: float = 1.8
    max_same_direction_bars: int = 3
    stop_atr_mult: float = 1.2
    target_r: float = 2.0
    max_hold_bars: int = 16
    cooldown_bars: int = 4
    base_fee_bps_per_side: float = 4.0
    base_slippage_bps_per_side: float = 2.0
    stress_fee_bps_per_side: float = 6.0
    stress_slippage_bps_per_side: float = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("..") / "ML-TRADE" / "training" / "fetchingData" / "generated_windows" / "basket_12_months",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "vol_expansion")
    return parser.parse_args()


def load_symbol_files(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*USDT_15m_12months.csv")):
        df = pd.read_csv(path, parse_dates=["time"])
        df = df.rename(columns={"ticker": "symbol"})
        df["symbol"] = df["symbol"].fillna(path.stem.split("_")[0])
        frames.append(df[["time", "open", "high", "low", "close", "volume", "symbol"]])
    if not frames:
        raise FileNotFoundError(f"No symbol CSV files found in {data_dir}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "time"]).reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("time").copy()
        prev_close = g["close"].shift(1)
        tr = pd.concat(
            [
                g["high"] - g["low"],
                (g["high"] - prev_close).abs(),
                (g["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr"] = tr.rolling(p.atr_lookback, min_periods=p.atr_lookback).mean()
        g["range"] = g["high"] - g["low"]
        g["body"] = (g["close"] - g["open"]).abs()
        g["body_fraction"] = g["body"] / g["range"].replace(0, np.nan)
        g["median_range"] = g["range"].shift(1).rolling(p.range_median_lookback, min_periods=p.range_median_lookback).median()
        g["median_volume"] = g["volume"].shift(1).rolling(p.volume_median_lookback, min_periods=p.volume_median_lookback).median()
        g["prior_high"] = g["high"].shift(1).rolling(p.breakout_lookback, min_periods=p.breakout_lookback).max()
        g["prior_low"] = g["low"].shift(1).rolling(p.breakout_lookback, min_periods=p.breakout_lookback).min()
        g["mid"] = (g["prior_high"] + g["prior_low"]) / 2.0
        direction = np.sign((g["close"] - g["open"]).to_numpy(dtype=float))
        up_streak = np.zeros(len(g), dtype=int)
        down_streak = np.zeros(len(g), dtype=int)
        current_dir = 0
        current_len = 0
        for idx, value in enumerate(direction):
            up_streak[idx] = current_len if current_dir == 1 else 0
            down_streak[idx] = current_len if current_dir == -1 else 0
            if value == current_dir and value != 0:
                current_len += 1
            elif value == 0:
                current_dir = 0
                current_len = 0
            else:
                current_dir = int(value)
                current_len = 1
        g["prior_up_streak"] = up_streak
        g["prior_down_streak"] = down_streak
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def make_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = add_features(df, p)
    has_history = df[["atr", "median_range", "median_volume", "prior_high", "prior_low"]].notna().all(axis=1)
    expansion = (df["range"] >= p.range_expansion_mult * df["median_range"]) & (
        df["volume"] >= p.volume_expansion_mult * df["median_volume"]
    )
    quality = (df["body_fraction"] >= p.min_body_fraction) & (df["range"] <= p.max_range_atr_mult * df["atr"])
    long_break = df["close"] > df["prior_high"]
    short_break = df["close"] < df["prior_low"]
    long_crowded = (
        (df["close"] - df["mid"] > p.max_extension_atr_mult * df["atr"])
        | (df["prior_up_streak"] > p.max_same_direction_bars)
    )
    short_crowded = (
        (df["mid"] - df["close"] > p.max_extension_atr_mult * df["atr"])
        | (df["prior_down_streak"] > p.max_same_direction_bars)
    )
    df["signal_direction"] = 0
    df.loc[has_history & expansion & quality & long_break & ~long_crowded, "signal_direction"] = 1
    df.loc[has_history & expansion & quality & short_break & ~short_crowded, "signal_direction"] = -1
    return df


def friction_r(entry: float, risk: float, fee_bps: float, slippage_bps: float) -> float:
    round_trip_bps = 2.0 * (fee_bps + slippage_bps)
    return (entry * round_trip_bps / 10_000.0) / risk


def simulate_symbol(g: pd.DataFrame, entry_mode: str, p: Params, fee_bps: float, slippage_bps: float) -> list[dict]:
    rows = g.sort_values("time").reset_index(drop=True)
    trades: list[dict] = []
    next_allowed_i = 0
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    atrs = rows["atr"].to_numpy(dtype=float)
    times = rows["time"].to_numpy()
    symbols = rows["symbol"].to_numpy()
    directions = rows["signal_direction"].to_numpy(dtype=int)
    signal_indices = np.flatnonzero(directions)
    for i in signal_indices:
        direction = int(directions[i])
        if direction == 0 or i < next_allowed_i:
            continue
        entry_i = i + 1 if entry_mode == "next_open" else i + 2
        if entry_i >= len(rows):
            continue
        entry = float(opens[entry_i] if entry_mode == "next_open" else closes[entry_i])
        risk = float(atrs[i] * p.stop_atr_mult)
        if not np.isfinite(entry) or not np.isfinite(risk) or risk <= 0:
            continue
        stop = entry - direction * risk
        target = entry + direction * risk * p.target_r
        exit_i = min(entry_i + p.max_hold_bars, len(rows) - 1)
        gross_r = 0.0
        exit_reason = "time_stop"
        for j in range(entry_i, min(entry_i + p.max_hold_bars, len(rows) - 1) + 1):
            hit_stop = bool(lows[j] <= stop) if direction == 1 else bool(highs[j] >= stop)
            hit_target = bool(highs[j] >= target) if direction == 1 else bool(lows[j] <= target)
            if hit_stop:
                gross_r = -1.0
                exit_reason = "stop"
                exit_i = j
                break
            if hit_target:
                gross_r = p.target_r
                exit_reason = "target"
                exit_i = j
                break
            exit_i = j
        if exit_reason == "time_stop":
            exit_close = float(closes[exit_i])
            gross_r = direction * (exit_close - entry) / risk
        cost_r = friction_r(entry, risk, fee_bps, slippage_bps)
        trades.append(
            {
                "symbol": symbols[i],
                "timeframe": p.timeframe,
                "entry_mode": entry_mode,
                "signal_time": times[i],
                "entry_time": times[entry_i],
                "exit_time": times[exit_i],
                "direction": "long" if direction == 1 else "short",
                "entry": entry,
                "risk_price": risk,
                "gross_r": gross_r,
                "fees_bps_per_side": fee_bps,
                "slippage_bps_per_side": slippage_bps,
                "friction_r": cost_r,
                "r_result": gross_r - cost_r,
                "exit_reason": exit_reason,
                "params": json.dumps(asdict(p), sort_keys=True),
            }
        )
        next_allowed_i = exit_i + p.cooldown_bars
    return trades


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity - equity.cummax()).min())


def max_losing_streak(values: pd.Series) -> int:
    worst = cur = 0
    for value in values:
        if value < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def summarize(trades: pd.DataFrame, scenario: str, p: Params) -> dict:
    if trades.empty:
        return {"scenario": scenario, "trade_count": 0}
    ordered = trades.sort_values("entry_time").copy()
    months = max((ordered["entry_time"].max() - ordered["entry_time"].min()).days / 365.25 * 12, 1 / 12)
    total_ev = float(ordered["r_result"].sum())
    by_symbol = ordered.groupby("symbol")["r_result"].sum().sort_values(ascending=False)
    by_month = ordered.groupby(ordered["entry_time"].dt.to_period("M").astype(str))["r_result"].sum().sort_values(ascending=False)
    positive_ev = total_ev if total_ev > 0 else np.nan
    return {
        "scenario": scenario,
        "entry_mode": str(ordered["entry_mode"].iloc[0]),
        "timeframe": p.timeframe,
        "fees_bps_per_side": float(ordered["fees_bps_per_side"].iloc[0]),
        "slippage_bps_per_side": float(ordered["slippage_bps_per_side"].iloc[0]),
        "trade_count": int(len(ordered)),
        "trades_per_year": float(len(ordered) / months * 12),
        "ev_per_trade_r": float(ordered["r_result"].mean()),
        "total_ev_r": total_ev,
        "max_drawdown_r": max_drawdown(ordered["r_result"].cumsum()),
        "max_losing_streak": max_losing_streak(ordered["r_result"]),
        "best_symbol_ev_share": float(by_symbol.iloc[0] / positive_ev) if by_symbol.iloc[0] > 0 and positive_ev == positive_ev else 0.0,
        "best_month_ev_share": float(by_month.iloc[0] / positive_ev) if by_month.iloc[0] > 0 and positive_ev == positive_ev else 0.0,
        "parameters": asdict(p),
    }


def write_report(out_dir: Path, summaries: list[dict], p: Params) -> None:
    lines = [
        "# Executable Volatility Expansion Bot - Crude OHLCV Validation",
        "",
        "## Scope",
        "",
        "- No ML, open interest, funding, complex filters, limit fills, or intrabar reconstruction.",
        "- Signals are known after candle close only.",
        "- Entry modes tested: `next_open` and `next_close`.",
        "- Same-candle exit ambiguity after entry is conservative: stop before target.",
        "",
        "## Fixed Parameters",
        "",
        "```json",
        json.dumps(asdict(p), indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| scenario | entry | fee bps/side | slip bps/side | trades | trades/year | EV/trade R | total EV R | max DD R | losing streak | top symbol EV share | top month EV share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        if s.get("trade_count", 0) == 0:
            continue
        lines.append(
            f"| {s['scenario']} | {s['entry_mode']} | {s['fees_bps_per_side']:.1f} | {s['slippage_bps_per_side']:.1f} | "
            f"{s['trade_count']} | {s['trades_per_year']:.1f} | {s['ev_per_trade_r']:.3f} | {s['total_ev_r']:.2f} | "
            f"{s['max_drawdown_r']:.2f} | {s['max_losing_streak']} | {s['best_symbol_ev_share']:.1%} | {s['best_month_ev_share']:.1%} |"
        )
    base_open = next(s for s in summaries if s["scenario"] == "base" and s["entry_mode"] == "next_open")
    base_close = next(s for s in summaries if s["scenario"] == "base" and s["entry_mode"] == "next_close")
    stress_open = next(s for s in summaries if s["scenario"] == "stress" and s["entry_mode"] == "next_open")
    close_ratio = base_close["ev_per_trade_r"] / base_open["ev_per_trade_r"] if base_open["ev_per_trade_r"] else np.nan
    pass_fail = {
        "base_next_open_ev_ge_0_3r": base_open["ev_per_trade_r"] >= 0.3,
        "base_next_close_ev_ge_0_3r": base_close["ev_per_trade_r"] >= 0.3,
        "base_next_open_trades_year_ge_100": base_open["trades_per_year"] >= 100,
        "base_next_close_trades_year_ge_100": base_close["trades_per_year"] >= 100,
        "stress_next_open_positive": stress_open["ev_per_trade_r"] > 0,
        "next_close_not_dramatically_worse": close_ratio >= 0.65,
        "top_symbol_share_ok": max(base_open["best_symbol_ev_share"], base_close["best_symbol_ev_share"]) <= 0.40,
        "top_month_share_ok": max(base_open["best_month_ev_share"], base_close["best_month_ev_share"]) <= 0.40,
    }
    lines.extend(["", "## Pass/Fail Checks", ""])
    for key, value in pass_fail.items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Overall crude-stage verdict: **{'PASS' if all(pass_fail.values()) else 'FAIL'}**",
            "",
            "See CSV outputs for trade-level R results, symbol distribution, monthly distribution, and equity drawdown path.",
        ]
    )
    (out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    p = Params()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_symbol_files(args.data_dir)
    signaled = make_signals(raw, p)
    all_trades: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for scenario, fee_bps, slippage_bps in [
        ("base", p.base_fee_bps_per_side, p.base_slippage_bps_per_side),
        ("stress", p.stress_fee_bps_per_side, p.stress_slippage_bps_per_side),
    ]:
        for entry_mode in ["next_open", "next_close"]:
            records: list[dict] = []
            for _, g in signaled.groupby("symbol", sort=False):
                records.extend(simulate_symbol(g, entry_mode, p, fee_bps, slippage_bps))
            trades = pd.DataFrame(records)
            if not trades.empty:
                trades["scenario"] = scenario
                trades["entry_time"] = pd.to_datetime(trades["entry_time"])
                trades["signal_time"] = pd.to_datetime(trades["signal_time"])
                trades["exit_time"] = pd.to_datetime(trades["exit_time"])
                trades["equity_r"] = trades.sort_values("entry_time")["r_result"].cumsum().values
                all_trades.append(trades)
                summaries.append(summarize(trades, scenario, p))
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    combined.to_csv(args.out_dir / "trades.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    if not combined.empty:
        combined.groupby(["scenario", "entry_mode", "symbol"]).agg(
            trades=("r_result", "size"),
            ev_per_trade_r=("r_result", "mean"),
            total_ev_r=("r_result", "sum"),
        ).reset_index().to_csv(args.out_dir / "symbol_distribution.csv", index=False)
        combined.assign(month=combined["entry_time"].dt.to_period("M").astype(str)).groupby(
            ["scenario", "entry_mode", "month"]
        ).agg(
            trades=("r_result", "size"),
            ev_per_trade_r=("r_result", "mean"),
            total_ev_r=("r_result", "sum"),
        ).reset_index().to_csv(args.out_dir / "monthly_distribution.csv", index=False)
        combined.sort_values("entry_time").assign(equity_r=lambda x: x["r_result"].cumsum()).to_csv(
            args.out_dir / "equity_curve.csv", index=False
        )
    write_report(args.out_dir, summaries, p)
    print(f"Wrote validation outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
