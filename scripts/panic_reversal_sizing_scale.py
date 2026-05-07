from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALLOCATIONS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
SYMBOL_ORDER = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
LIQUIDITY_PRIORITY = {symbol: rank for rank, symbol in enumerate(SYMBOL_ORDER)}


def load_trade_stream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["event_time", "signal_time", "signal_close_time", "entry_candle_open_time", "entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    df = df[
        (df["entry_mode"] == "next_close")
        & (df["exit_model"] == "pure_horizon_12")
        & (df["horizon_bars"] == 12)
    ].copy()
    df["symbol_rank"] = df["symbol"].map(LIQUIDITY_PRIORITY).fillna(999).astype(int)
    df["abs_drop"] = df["event_return_pct"].abs()
    return df


def scenario_trades(all_trades: pd.DataFrame, friction_mode: str) -> pd.DataFrame:
    if friction_mode == "cluster_stress":
        base = all_trades[all_trades["friction_case"] == "base"].copy()
        stress = all_trades[all_trades["friction_case"] == "stress"][["symbol", "signal_time", "net_return"]].rename(columns={"net_return": "stress_net_return"})
        base = base.merge(stress, on=["symbol", "signal_time"], how="left")
        extra = (base["stress_net_return"] - base["net_return"]).fillna(0.0)
        base["effective_net_return"] = np.where(
            base["breadth_count"] >= 7,
            base["stress_net_return"].fillna(base["net_return"]) + extra,
            np.where(base["breadth_count"] >= 5, base["stress_net_return"].fillna(base["net_return"]), base["net_return"]),
        )
        return base
    return all_trades[all_trades["friction_case"] == friction_mode].assign(effective_net_return=lambda x: x["net_return"]).copy()


def select_with_caps(candidates: pd.DataFrame, concurrent_cap: int = 10, symbol_cap: int = 2) -> pd.DataFrame:
    candidates = candidates.sort_values(["signal_time", "symbol_rank", "symbol"]).reset_index(drop=True)
    active: list[tuple[pd.Timestamp, str]] = []
    accepted: list[dict[str, object]] = []
    for signal_time, group in candidates.groupby("signal_time", sort=True):
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > signal_time]
        active_by_symbol: dict[str, int] = {}
        for _, symbol in active:
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
        ordered = group.sort_values(["abs_drop", "symbol_rank", "symbol"], ascending=[False, True, True])
        for trade in ordered.to_dict(orient="records"):
            symbol = str(trade["symbol"])
            if len(active) >= concurrent_cap:
                continue
            if active_by_symbol.get(symbol, 0) >= symbol_cap:
                continue
            trade["active_positions_before_entry"] = len(active)
            accepted.append(trade)
            active.append((trade["exit_time"], symbol))
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
    return pd.DataFrame(accepted)


def max_drawdown(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns), dtype=float)
    if series.empty:
        return 0.0
    equity = (1.0 + series).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def exposure_stats(trades: pd.DataFrame, allocation: float) -> tuple[float, int, float, float, float, float]:
    changes: list[tuple[pd.Timestamp, int]] = []
    for row in trades.itertuples(index=False):
        changes.append((row.entry_time, 1))
        changes.append((row.exit_time, -1))
    active = 0
    samples: list[int] = []
    for _, delta in sorted(changes, key=lambda item: (item[0], -item[1])):
        active += delta
        samples.append(active)
    avg = float(np.mean(samples)) if samples else 0.0
    max_active = int(np.max(samples)) if samples else 0
    p95 = float(np.quantile(samples, 0.95)) if samples else 0.0
    return avg, max_active, p95, avg * allocation, max_active * allocation, p95 * allocation


def equity_curve(trades: pd.DataFrame, allocation: float, friction_mode: str) -> pd.DataFrame:
    rows = []
    equity = 1.0
    for row in trades.sort_values(["exit_time", "symbol"]).itertuples(index=False):
        account_return = allocation * row.effective_net_return
        equity *= 1.0 + account_return
        rows.append(
            {
                "timestamp": row.exit_time,
                "symbol": row.symbol,
                "friction_mode": friction_mode,
                "allocation": allocation,
                "trade_return": row.effective_net_return,
                "account_return": account_return,
                "equity": equity,
                "total_return_pct": (equity - 1.0) * 100.0,
                "active_positions_before_entry": row.active_positions_before_entry,
            }
        )
    return pd.DataFrame(rows)


def monthly_returns(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve.copy()
    out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    return (
        out.groupby(["friction_mode", "allocation", "month"], sort=False)["account_return"]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index(name="monthly_return")
        .assign(monthly_return_pct=lambda x: x["monthly_return"] * 100.0)
    )


def summarize(curve: pd.DataFrame, trades: pd.DataFrame, allocation: float, friction_mode: str) -> dict[str, object]:
    account_returns = curve["account_return"]
    total_return = float(curve["equity"].iloc[-1] - 1.0)
    years = max((trades["entry_time"].max() - trades["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    months = monthly_returns(curve)
    monthly_std = float(months["monthly_return"].std(ddof=1)) if len(months) > 1 else 0.0
    std = float(account_returns.std(ddof=1)) if len(account_returns) > 1 else 0.0
    avg_conc, max_conc, p95_conc, avg_exp, max_exp, p95_exp = exposure_stats(trades, allocation)
    max_dd = max_drawdown(account_returns)
    return {
        "friction_mode": friction_mode,
        "allocation": allocation,
        "allocation_pct": allocation * 100.0,
        "trades": int(len(trades)),
        "account_total_return_pct": total_return * 100.0,
        "cagr_pct": ((1.0 + total_return) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "worst_month_pct": float(months["monthly_return_pct"].min()),
        "best_month_pct": float(months["monthly_return_pct"].max()),
        "monthly_volatility_pct": monthly_std * 100.0,
        "sharpe_like": float((account_returns.mean() / std) * np.sqrt(len(account_returns))) if std > 0 else 0.0,
        "max_exposure_pct": max_exp * 100.0,
        "average_exposure_pct": avg_exp * 100.0,
        "p95_exposure_pct": p95_exp * 100.0,
        "average_concurrent_positions": avg_conc,
        "max_concurrent_positions": max_conc,
        "p95_concurrent_positions": p95_conc,
        "conservative_dd_ok": bool(abs(max_dd) <= 0.10),
        "moderate_dd_ok": bool(abs(max_dd) <= 0.20),
        "aggressive_dd_ok": bool(abs(max_dd) <= 0.30),
    }


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
    parser = argparse.ArgumentParser(description="Panic-Reversal Bot position sizing scale test.")
    parser.add_argument("--trade-returns", type=Path, default=Path("reports") / "panic_reversal_extended" / "trade_returns.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "panic_reversal_sizing_scale")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_trades = load_trade_stream(args.trade_returns)
    accepted_by_mode = {mode: select_with_caps(scenario_trades(all_trades, mode)) for mode in ["base", "stress", "cluster_stress"]}
    curves = []
    results = []
    monthly = []
    for mode, trades in accepted_by_mode.items():
        for allocation in ALLOCATIONS:
            curve = equity_curve(trades, allocation, mode)
            curves.append(curve)
            results.append(summarize(curve, trades, allocation, mode))
            monthly.append(monthly_returns(curve))
    results_df = pd.DataFrame(results)
    curves_df = pd.concat(curves, ignore_index=True)
    monthly_df = pd.concat(monthly, ignore_index=True)
    stress_comparison = results_df.pivot(index="allocation", columns="friction_mode", values=["account_total_return_pct", "max_drawdown_pct", "cagr_pct"]).reset_index()
    stress_comparison.columns = ["_".join([str(part) for part in col if part != ""]).strip("_") for col in stress_comparison.columns.to_flat_index()]

    paths = {
        "sizing_results_csv": args.out_dir / "sizing_results.csv",
        "equity_curves_csv": args.out_dir / "equity_curves.csv",
        "monthly_returns_csv": args.out_dir / "monthly_returns.csv",
        "stress_comparison_csv": args.out_dir / "stress_comparison.csv",
        "sizing_summary_json": args.out_dir / "sizing_summary.json",
    }
    results_df.to_csv(paths["sizing_results_csv"], index=False)
    curves_df.to_csv(paths["equity_curves_csv"], index=False)
    monthly_df.to_csv(paths["monthly_returns_csv"], index=False)
    stress_comparison.to_csv(paths["stress_comparison_csv"], index=False)

    base = results_df[results_df["friction_mode"] == "base"].copy()
    recommendations = {
        "conservative": base[base["conservative_dd_ok"]].sort_values("account_total_return_pct", ascending=False).head(1).to_dict(orient="records"),
        "balanced": base[(base["moderate_dd_ok"]) & (base["allocation"] <= 0.02)].sort_values("account_total_return_pct", ascending=False).head(1).to_dict(orient="records"),
        "aggressive": base[(base["aggressive_dd_ok"]) & (base["allocation"] <= 0.05)].sort_values("account_total_return_pct", ascending=False).head(1).to_dict(orient="records"),
    }
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_trade_returns": str(args.trade_returns),
        "fixed_config": {
            "entry_mode": "next_close",
            "concurrent_cap": 10,
            "symbol_cap": 2,
            "selection_policy": "largest_drop",
            "exit_model": "pure_horizon_12",
        },
        "allocations": ALLOCATIONS,
        "reports": {key: str(path) for key, path in paths.items()},
        "recommendations": recommendations,
        "results": results_df.to_dict(orient="records"),
    }
    paths["sizing_summary_json"].write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
