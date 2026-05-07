from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SYMBOL_ORDER = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
LIQUIDITY_PRIORITY = {symbol: rank for rank, symbol in enumerate(SYMBOL_ORDER)}
ACCOUNT_ALLOCATIONS = [0.0025, 0.005, 0.01]
CONCURRENT_CAPS = [3, 5, 10, 15, 20, None]
SYMBOL_CAPS = [1, 2, None]
SELECTION_POLICIES = ["symbol_order", "largest_drop", "liquidity_priority"]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    entry_mode: str
    friction_mode: str
    concurrent_cap: int | None
    symbol_cap: int | None
    selection_policy: str
    cluster_stress: bool


def cap_label(value: int | None) -> str:
    return "unlimited" if value is None else str(value)


def make_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for entry_mode in ["next_close", "next_open"]:
        for friction_mode in ["base", "stress"]:
            for concurrent_cap in CONCURRENT_CAPS:
                for symbol_cap in SYMBOL_CAPS:
                    for selection_policy in SELECTION_POLICIES:
                        scenario_id = (
                            f"{entry_mode}_{friction_mode}_ccap{cap_label(concurrent_cap)}_"
                            f"scap{cap_label(symbol_cap)}_{selection_policy}"
                        )
                        scenarios.append(
                            Scenario(
                                scenario_id=scenario_id,
                                entry_mode=entry_mode,
                                friction_mode=friction_mode,
                                concurrent_cap=concurrent_cap,
                                symbol_cap=symbol_cap,
                                selection_policy=selection_policy,
                                cluster_stress=False,
                            )
                        )
    for entry_mode in ["next_close", "next_open"]:
        for concurrent_cap in CONCURRENT_CAPS:
            for symbol_cap in SYMBOL_CAPS:
                for selection_policy in SELECTION_POLICIES:
                    scenario_id = (
                        f"{entry_mode}_clusterstress_ccap{cap_label(concurrent_cap)}_"
                        f"scap{cap_label(symbol_cap)}_{selection_policy}"
                    )
                    scenarios.append(
                        Scenario(
                            scenario_id=scenario_id,
                            entry_mode=entry_mode,
                            friction_mode="cluster_stress",
                            concurrent_cap=concurrent_cap,
                            symbol_cap=symbol_cap,
                            selection_policy=selection_policy,
                            cluster_stress=True,
                        )
                    )
    return scenarios


def load_trade_stream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["event_time", "signal_time", "signal_close_time", "entry_candle_open_time", "entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    df = df[(df["exit_model"] == "pure_horizon_12") & (df["horizon_bars"] == 12)].copy()
    df["symbol_rank"] = df["symbol"].map(LIQUIDITY_PRIORITY).fillna(999).astype(int)
    df["abs_drop"] = df["event_return_pct"].abs()
    return df


def scenario_base_trades(all_trades: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    if scenario.cluster_stress:
        base = all_trades[
            (all_trades["entry_mode"] == scenario.entry_mode)
            & (all_trades["friction_case"] == "base")
        ].copy()
        stress = all_trades[
            (all_trades["entry_mode"] == scenario.entry_mode)
            & (all_trades["friction_case"] == "stress")
        ][["symbol", "signal_time", "net_return"]].rename(columns={"net_return": "stress_net_return"})
        base = base.merge(stress, on=["symbol", "signal_time"], how="left")
        extra_stress = (base["stress_net_return"] - base["net_return"]).fillna(0.0)
        base["effective_net_return"] = np.where(
            base["breadth_count"] >= 7,
            base["stress_net_return"].fillna(base["net_return"]) + extra_stress,
            np.where(base["breadth_count"] >= 5, base["stress_net_return"].fillna(base["net_return"]), base["net_return"]),
        )
        base["effective_friction_case"] = np.where(base["breadth_count"] >= 7, "extra_stress", np.where(base["breadth_count"] >= 5, "stress", "base"))
        return base
    return all_trades[
        (all_trades["entry_mode"] == scenario.entry_mode)
        & (all_trades["friction_case"] == scenario.friction_mode)
    ].assign(effective_net_return=lambda x: x["net_return"], effective_friction_case=scenario.friction_mode).copy()


def order_candidates(group: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "largest_drop":
        return group.sort_values(["abs_drop", "symbol_rank", "symbol"], ascending=[False, True, True])
    if policy == "liquidity_priority":
        return group.sort_values(["symbol_rank", "abs_drop", "symbol"], ascending=[True, False, True])
    return group.sort_values(["symbol_rank", "symbol"], ascending=[True, True])


def simulate_scenario(all_trades: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = scenario_base_trades(all_trades, scenario).sort_values(["signal_time", "symbol_rank", "symbol"]).reset_index(drop=True)
    stream = make_ordered_stream(candidates, scenario.selection_policy)
    return simulate_ordered_stream(stream, scenario)


def make_ordered_stream(candidates: pd.DataFrame, selection_policy: str) -> list[tuple[pd.Timestamp, list[dict[str, object]]]]:
    stream = []
    for signal_time, group in candidates.groupby("signal_time", sort=True):
        stream.append((signal_time, order_candidates(group, selection_policy).to_dict(orient="records")))
    return stream


def simulate_ordered_stream(stream: list[tuple[pd.Timestamp, list[dict[str, object]]]], scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: list[tuple[pd.Timestamp, str]] = []
    accepted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for signal_time, records in stream:
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > signal_time]
        active_by_symbol: dict[str, int] = {}
        for _, symbol in active:
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
        for trade in records:
            reason = None
            symbol = str(trade["symbol"])
            if scenario.concurrent_cap is not None and len(active) >= scenario.concurrent_cap:
                reason = "concurrent_cap"
            elif scenario.symbol_cap is not None and active_by_symbol.get(symbol, 0) >= scenario.symbol_cap:
                reason = "symbol_cap"
            if reason is not None:
                skipped.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "entry_mode": scenario.entry_mode,
                        "friction_mode": scenario.friction_mode,
                        "concurrent_cap": cap_label(scenario.concurrent_cap),
                        "symbol_cap": cap_label(scenario.symbol_cap),
                        "selection_policy": scenario.selection_policy,
                        "signal_time": signal_time,
                        "symbol": symbol,
                        "skip_reason": reason,
                        "breadth_count": trade["breadth_count"],
                        "event_return_pct": trade["event_return_pct"],
                        "active_positions": len(active),
                        "active_symbol_positions": active_by_symbol.get(symbol, 0),
                    }
                )
                continue
            trade["scenario_id"] = scenario.scenario_id
            trade["scenario_entry_mode"] = scenario.entry_mode
            trade["scenario_friction_mode"] = scenario.friction_mode
            trade["concurrent_cap"] = cap_label(scenario.concurrent_cap)
            trade["symbol_cap"] = cap_label(scenario.symbol_cap)
            trade["selection_policy"] = scenario.selection_policy
            trade["active_positions_before_entry"] = len(active)
            accepted.append(trade)
            active.append((trade["exit_time"], symbol))
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1
    accepted_df = pd.DataFrame(accepted)
    skipped_df = pd.DataFrame(skipped)
    return accepted_df, skipped_df


def max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return 0.0
    equity = (1.0 + series).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def additive_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return 0.0
    equity = series.cumsum()
    return float((equity - equity.cummax()).min())


def max_losing_streak(values: Iterable[float]) -> int:
    current = worst = 0
    for value in values:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


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
    if not samples:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0
    avg_concurrent = float(np.mean(samples))
    max_concurrent = int(np.max(samples))
    p95_concurrent = float(np.quantile(samples, 0.95))
    return (
        avg_concurrent,
        max_concurrent,
        p95_concurrent,
        avg_concurrent * allocation,
        max_concurrent * allocation,
        p95_concurrent * allocation,
    )


def summarize_trades(trades: pd.DataFrame, skipped: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    ordered = trades.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    returns = ordered["effective_net_return"]
    account_returns = ordered["account_return"]
    cumulative_units = float(returns.sum())
    total_account_return = float((1.0 + account_returns).prod() - 1.0)
    years = max((ordered["entry_time"].max() - ordered["entry_time"].min()).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    std = float(account_returns.std(ddof=1)) if len(account_returns) > 1 else 0.0
    monthly = ordered.assign(month=ordered["exit_time"].dt.strftime("%Y-%m")).groupby("month")["account_return"].apply(lambda x: (1.0 + x).prod() - 1.0)
    symbol_units = ordered.groupby("symbol")["effective_net_return"].sum()
    avg_conc, max_conc, p95_conc, avg_exp, max_exp, p95_exp = exposure_stats(ordered, float(ordered["account_allocation"].iloc[0]))
    skipped_count = int(len(skipped)) if not skipped.empty else 0
    total_signals = int(len(ordered) + skipped_count)
    out = {col: ordered[col].iloc[0] for col in group_cols}
    out.update(
        {
            "trades_taken": int(len(ordered)),
            "trades_skipped": skipped_count,
            "skip_rate": float(skipped_count / total_signals) if total_signals else 0.0,
            "cumulative_return_units": cumulative_units,
            "average_trade_return_pct": float(returns.mean() * 100.0),
            "median_trade_return_pct": float(returns.median() * 100.0),
            "win_rate": float((returns > 0).mean()),
            "account_total_return_pct": total_account_return * 100.0,
            "cagr_pct": ((1.0 + total_account_return) ** (1.0 / years) - 1.0) * 100.0,
            "max_drawdown_pct": max_drawdown(account_returns) * 100.0,
            "max_drawdown_units": additive_drawdown(returns),
            "sharpe_like": float((account_returns.mean() / std) * np.sqrt(len(account_returns))) if std > 0 else 0.0,
            "max_losing_streak": max_losing_streak(returns),
            "average_concurrent_positions": avg_conc,
            "max_concurrent_positions": max_conc,
            "p95_concurrent_positions": p95_conc,
            "average_exposure_pct": avg_exp * 100.0,
            "max_exposure_pct": max_exp * 100.0,
            "p95_exposure_pct": p95_exp * 100.0,
            "worst_month": monthly.idxmin(),
            "worst_month_return_pct": float(monthly.min() * 100.0),
            "best_month": monthly.idxmax(),
            "best_month_contribution": float(monthly.max() / total_account_return) if total_account_return > 0 else np.nan,
            "best_symbol": symbol_units.idxmax(),
            "best_symbol_contribution": float(symbol_units.max() / cumulative_units) if cumulative_units > 0 else np.nan,
            "positive_symbol_rate": float((symbol_units > 0).mean()),
            "liquidation_margin_risk_proxy": "high" if max_exp >= 0.5 else ("medium" if max_exp >= 0.25 else "low"),
        }
    )
    return out


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_id, group in trades.groupby("scenario_id", sort=False):
        equity = 1.0
        for row in group.sort_values(["exit_time", "symbol"]).itertuples(index=False):
            equity *= 1.0 + row.account_return
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": row.exit_time,
                    "symbol": row.symbol,
                    "account_return": row.account_return,
                    "equity": equity,
                    "cumulative_return_pct": (equity - 1.0) * 100.0,
                    "active_positions_before_entry": row.active_positions_before_entry,
                    "account_allocation": row.account_allocation,
                    "concurrent_cap": row.concurrent_cap,
                    "symbol_cap": row.symbol_cap,
                    "selection_policy": row.selection_policy,
                    "entry_mode": row.scenario_entry_mode,
                    "friction_mode": row.scenario_friction_mode,
                }
            )
    return pd.DataFrame(rows)


def expand_allocations(taken: pd.DataFrame, skipped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    taken_parts = []
    skipped_parts = []
    for allocation in ACCOUNT_ALLOCATIONS:
        t = taken.copy()
        t["selection_scenario_id"] = t["scenario_id"]
        t["scenario_id"] = t["scenario_id"] + f"_alloc{allocation:g}"
        t["account_allocation"] = allocation
        t["account_return"] = t["effective_net_return"] * allocation
        taken_parts.append(t)
        if not skipped.empty:
            s = skipped.copy()
            s["selection_scenario_id"] = s["scenario_id"]
            s["scenario_id"] = s["scenario_id"] + f"_alloc{allocation:g}"
            s["account_allocation"] = allocation
            skipped_parts.append(s)
    return pd.concat(taken_parts, ignore_index=True), pd.concat(skipped_parts, ignore_index=True) if skipped_parts else pd.DataFrame()


def build_reports(taken: pd.DataFrame, skipped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_cols = [
        "scenario_id",
        "scenario_entry_mode",
        "scenario_friction_mode",
        "concurrent_cap",
        "symbol_cap",
        "selection_policy",
        "account_allocation",
    ]
    skipped_groups = {scenario_id: group for scenario_id, group in skipped.groupby("scenario_id", sort=False)} if not skipped.empty else {}
    rows = []
    for scenario_id, group in taken.groupby("scenario_id", sort=False):
        skip_group = skipped_groups.get(scenario_id, pd.DataFrame())
        rows.append(summarize_trades(group, skip_group, scenario_cols))
    exposure_results = pd.DataFrame(rows)

    detail = taken[
        (taken["scenario_entry_mode"] == "next_close")
        & (taken["concurrent_cap"].isin(["5", "10", "unlimited"]))
        & (taken["symbol_cap"].isin(["1", "2", "unlimited"]))
    ].copy()
    year_rows = []
    month_rows = []
    symbol_rows = []
    for scenario_id, group in detail.groupby("scenario_id", sort=False):
        by_year = group.assign(year=group["exit_time"].dt.strftime("%Y"))
        for _, yg in by_year.groupby("year", sort=False):
            year_rows.append(summarize_trades(yg, pd.DataFrame(), scenario_cols + ["year"]))
        by_month = group.assign(month=group["exit_time"].dt.strftime("%Y-%m"))
        for _, mg in by_month.groupby("month", sort=False):
            month_rows.append(summarize_trades(mg, pd.DataFrame(), scenario_cols + ["month"]))
        for _, sg in group.groupby("symbol", sort=False):
            symbol_rows.append(summarize_trades(sg, pd.DataFrame(), scenario_cols + ["symbol"]))
    by_year = pd.DataFrame(year_rows)
    by_month = pd.DataFrame(month_rows)
    by_symbol = pd.DataFrame(symbol_rows)
    equity = build_equity_curve(detail)
    cap_cols = ["scenario_entry_mode", "scenario_friction_mode", "concurrent_cap", "symbol_cap", "selection_policy", "account_allocation"]
    cap_comparison = exposure_results.sort_values("account_total_return_pct", ascending=False)[
        cap_cols
        + [
            "trades_taken",
            "trades_skipped",
            "skip_rate",
            "account_total_return_pct",
            "cagr_pct",
            "max_drawdown_pct",
            "max_exposure_pct",
            "sharpe_like",
            "best_month_contribution",
            "best_symbol_contribution",
        ]
    ]
    return exposure_results, equity, by_year, by_month, by_symbol, cap_comparison


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
    parser = argparse.ArgumentParser(description="Panic-Reversal Bot Phase 2 exposure control simulation.")
    parser.add_argument("--trade-returns", type=Path, default=Path("reports") / "panic_reversal_extended" / "trade_returns.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "panic_reversal_phase2_exposure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_trades = load_trade_stream(args.trade_returns)
    scenarios = make_scenarios()
    stream_cache: dict[tuple[str, str, str], list[tuple[pd.Timestamp, list[dict[str, object]]]]] = {}
    for entry_mode in ["next_close", "next_open"]:
        for friction_mode in ["base", "stress", "cluster_stress"]:
            dummy = Scenario(
                scenario_id=f"{entry_mode}_{friction_mode}_cache",
                entry_mode=entry_mode,
                friction_mode=friction_mode,
                concurrent_cap=None,
                symbol_cap=None,
                selection_policy="symbol_order",
                cluster_stress=friction_mode == "cluster_stress",
            )
            candidates = scenario_base_trades(all_trades, dummy).sort_values(["signal_time", "symbol_rank", "symbol"]).reset_index(drop=True)
            for policy in SELECTION_POLICIES:
                stream_cache[(entry_mode, friction_mode, policy)] = make_ordered_stream(candidates, policy)
    taken_parts = []
    skipped_parts = []
    for scenario in scenarios:
        stream = stream_cache[(scenario.entry_mode, scenario.friction_mode, scenario.selection_policy)]
        taken, skipped = simulate_ordered_stream(stream, scenario)
        if not taken.empty:
            taken_parts.append(taken)
        if not skipped.empty:
            skipped_parts.append(skipped)
    taken_base = pd.concat(taken_parts, ignore_index=True)
    skipped_base = pd.concat(skipped_parts, ignore_index=True) if skipped_parts else pd.DataFrame()
    taken_all, skipped_all = expand_allocations(taken_base, skipped_base)
    for col in ["event_time", "signal_time", "signal_close_time", "entry_candle_open_time", "entry_time", "exit_time"]:
        if col in taken_all.columns:
            taken_all[col] = pd.to_datetime(taken_all[col], utc=True, format="mixed")
    exposure_results, equity, by_year, by_month, by_symbol, cap_comparison = build_reports(taken_all, skipped_all)

    paths = {
        "exposure_results_csv": args.out_dir / "exposure_results.csv",
        "equity_curves_csv": args.out_dir / "equity_curves.csv",
        "by_year_csv": args.out_dir / "by_year.csv",
        "by_month_csv": args.out_dir / "by_month.csv",
        "by_symbol_csv": args.out_dir / "by_symbol.csv",
        "skipped_signals_report_csv": args.out_dir / "skipped_signals_report.csv",
        "cap_comparison_csv": args.out_dir / "cap_comparison.csv",
        "phase2_summary_json": args.out_dir / "phase2_summary.json",
    }
    exposure_results.to_csv(paths["exposure_results_csv"], index=False)
    equity.to_csv(paths["equity_curves_csv"], index=False)
    by_year.to_csv(paths["by_year_csv"], index=False)
    by_month.to_csv(paths["by_month_csv"], index=False)
    by_symbol.to_csv(paths["by_symbol_csv"], index=False)
    skipped_all.to_csv(paths["skipped_signals_report_csv"], index=False)
    cap_comparison.to_csv(paths["cap_comparison_csv"], index=False)

    practical = exposure_results[
        (exposure_results["scenario_entry_mode"] == "next_close")
        & (exposure_results["scenario_friction_mode"] == "base")
        & (exposure_results["concurrent_cap"].isin(["5", "10"]))
        & (exposure_results["symbol_cap"].isin(["1", "2"]))
        & (exposure_results["account_allocation"] == 0.005)
    ].sort_values(["account_total_return_pct", "max_drawdown_pct"], ascending=[False, False])
    stress_pairs = exposure_results[
        (exposure_results["scenario_entry_mode"] == "next_close")
        & (exposure_results["scenario_friction_mode"] == "stress")
    ]
    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_trade_returns": str(args.trade_returns),
        "reports": {key: str(path) for key, path in paths.items()},
        "scenario_count": int(len(exposure_results)),
        "account_allocations": ACCOUNT_ALLOCATIONS,
        "concurrent_caps": [cap_label(x) for x in CONCURRENT_CAPS],
        "symbol_caps": [cap_label(x) for x in SYMBOL_CAPS],
        "selection_policies": SELECTION_POLICIES,
        "top_15_by_account_return": exposure_results.sort_values("account_total_return_pct", ascending=False).head(15).to_dict(orient="records"),
        "best_practical_0_5pct": practical.head(10).to_dict(orient="records"),
        "best_stress_next_close": stress_pairs.sort_values("account_total_return_pct", ascending=False).head(10).to_dict(orient="records"),
        "pass_checks": {
            "cap5_or_cap10_positive": bool((practical["account_total_return_pct"] > 0).any()),
            "cap5_or_cap10_stress_positive": bool(
                (
                    (exposure_results["scenario_entry_mode"] == "next_close")
                    & (exposure_results["scenario_friction_mode"] == "stress")
                    & (exposure_results["concurrent_cap"].isin(["5", "10"]))
                    & (exposure_results["account_total_return_pct"] > 0)
                ).any()
            ),
            "cap5_or_cap10_has_150_trades": bool((practical["trades_taken"] >= 150).any()),
        },
    }
    paths["phase2_summary_json"].write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote reports to {args.out_dir.resolve()}")
    print(exposure_results.sort_values("account_total_return_pct", ascending=False).head(20).to_string(index=False))
    print(json.dumps(json_safe(summary["pass_checks"]), indent=2))


if __name__ == "__main__":
    main()
