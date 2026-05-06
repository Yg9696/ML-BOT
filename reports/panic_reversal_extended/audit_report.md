# Panic-Reversal Extended Validation Audit

Branch audited: `research/panic-reversal-extended-validation-agent1`

Audit role: Agent 2 - Execution & Leakage Auditor

## 1. What I Checked

- Reviewed `scripts/panic_reversal_extended_validation.py` for lookahead leakage in the event return, realized-volatility regime filter, simultaneous down-event breadth, BTC 24h return, EMA50/EMA200 slopes, and future-return use.
- Reviewed cross-symbol timestamp alignment for breadth and BTC context joins.
- Reviewed execution timing for `next_open` and `next_close`, market-only entry, fee/slippage direction, and base vs stress friction.
- Reviewed fixed-size additive return accounting, equity curve chronology, overlap/stacking behavior, and exposure observability.
- Reviewed metrics in `overall_results.csv`, `by_year.csv`, `by_month.csv`, `by_symbol.csv`, `execution_comparison.csv`, `equity_curve.csv`, and `extended_validation_summary.json`.
- Reviewed 15m data quality from 2020-01-01 through 2026-05-04, including pre-listing missing periods, duplicate rows, large gaps, zero-volume rows, and OHLC sanity.
- Re-ran the current script into a temporary audit output folder without changing committed reports, then deleted the temporary folder.

## 2. Critical Issues

### Checked-in reports are not reproducible from the current script

The committed `extended_validation_summary.json` and `overall_results.csv` claim the primary full-period `next_close` / `pure_horizon_12` / `base` scenario has:

- `total_trades = 1134`
- `trades_per_year = 178.9`
- `average_return_pct = 2.37`
- `cumulative_return_units = 26.91`
- `minimum_pass = true`

Re-running the current `scripts/panic_reversal_extended_validation.py` with its default arguments produced materially different primary full-period results:

- `total_trades = 71`
- `trades_per_year = 11.20`
- `average_return_pct = 3.10`
- `cumulative_return_units = 2.20`
- `minimum_pass = false`

The rerun failed because `trades_at_least_150` is false and `month_concentration_ok` is false. This blocks trust in the reported pass. The current code and checked-in reports appear out of sync.

### `next_close` entry timestamps are one candle too early

For `next_close`, the code enters at `closes[entry_i]`, but records `entry_time = timestamps[entry_i]`. Binance kline timestamps are candle-open timestamps, so a next-candle close fill should be timestamped at `timestamps[entry_i] + 15 minutes`, not `timestamps[entry_i]`.

This does not change the saved return math directly, but it contaminates chronological accounting, exposure reconstruction, window attribution near boundaries, and the specific `next_open` vs `next_close` question.

### Exposure/stacking risk is not reported in the extended validation

The strategy allows fixed-size additive trades across symbols and timestamps. In an in-memory reconstruction of the current primary scenario, maximum concurrent positions reached 26, with up to 9 signals sharing the same signal timestamp and entry timestamp.

The extended reports do not include trade-level rows or `max_concurrent_positions`, and the equity curve omits entry times. Therefore the reported cumulative return is not directly interpretable as account return or capital-normalized return. This is especially important for a panic basket strategy where simultaneous signals cluster.

## 3. Medium Issues

### Realized-volatility regime uses the signal candle return

`realized_vol_96` is computed from `returns.rolling(...).std()` without shifting, while only the threshold is shifted. Therefore `high_vol_regime` includes the current event candle's return in the realized-volatility value.

This is known at signal close, so it is not future leakage, but it violates the requested audit condition that the realized-volatility high-regime filter use prior data only. In a reconstruction, using prior-only realized volatility reduced primary candidate signals from 71 to 57, with 14 current-only signals.

### BTC 24h return uses same-timestamp BTC close

BTC context is merged on the signal timestamp and uses `btc["close"] / btc["close"].shift(96) - 1`. This is known only after the same 15m candle close. That is acceptable only if the system explicitly assumes all symbol closes and BTC context are available before the next-candle entry decision. It should be documented as signal-close information, not pre-close information.

### Breadth uses same-timestamp close-to-close returns across all symbols

The breadth count is built from each symbol's same-timestamp `event_return <= -0.02`. This is also signal-close information. It does not use future rows, and missing pre-listing symbols are absent rather than counted as down events, but the execution model must assume the full basket's just-closed candle data is available before placing entries.

### Data quality report does not distinguish pre-listing gaps from true missing data

Several symbols have large `missing_rows_vs_requested_start` because they listed after 2020-01-01. The code comments acknowledge this, but the report does not separate pre-listing unavailable rows from internal missing gaps. Internal large gaps are zero, which is good, but the report needs a clearer listing-aware quality status.

### No full trade-return report is produced

Only an equity curve is written. Without signal time, entry time, exit time, signal features, and net return per trade in a report or sample, independent audit of overlap, event clustering, and return attribution is unnecessarily difficult.

## 4. Minor Issues

### Pass/fail criteria are weaker than the stated Phase 2 concern

The code's `pass_fail` only checks the primary `next_close` / `pure_horizon_12` / `base` row and a matched stress cumulative return. It does not explicitly test whether `next_open` outperformance is explainable, whether recent subperiods remain robust, or whether exposure is acceptable.

### Best-month contribution can exceed 1.0 in subperiod reports

Some year rows show `best_month_contribution > 1.0` when cumulative return is small and offset by losing months. This is mathematically possible, but it should be labeled as a concentration ratio over net cumulative return, not as a bounded share.

### Zero-volume rows exist

The data-quality report shows 6-7 zero-volume rows per symbol. This is small relative to the dataset, but these rows should be documented or excluded if they ever affect feature windows.

## 5. Whether Results Are Trustworthy

No, not in the current committed state.

The strong positive checked-in result is not trustworthy because the committed reports do not reproduce from the current script. The current script's own rerun fails the minimum pass criteria. There is no evidence of direct future-return leakage in the fixed-horizon exits, and stress friction is correctly worse than base, but report/code inconsistency, `next_close` timestamp labeling, current-candle realized-volatility regime use, and unreported concurrent exposure are enough to block Phase 2.

The suspicious `next_open` outperformance is not explained by future leakage in the price selection. A paired reconstruction of the current primary scenario showed `next_open` beating `next_close` on 63.4% of paired trades, but the mean open-minus-close return was negative over the current rerun because 2025 and 2026 differed sharply. The checked-in report's larger `next_open` advantage cannot be trusted until reports are regenerated from the current code and entry timestamps are fixed.

## 6. Required Fixes Before Phase 2

1. Regenerate all extended validation reports from the exact committed script, or commit the script version that produced the checked-in reports. The code and reports must be reproducible.
2. Fix `next_close` timestamp semantics by recording next-candle close time, or add explicit open-time/close-time fields and use the close timestamp for exposure and window attribution.
3. Change the realized-volatility regime filter to prior-only if that is the intended rule, e.g. compare `realized_vol_96.shift(1)` to the shifted threshold.
4. Add trade-level output or at least a `trades_sample.csv` with signal time, entry time, exit time, features, and net return.
5. Report `average_concurrent_positions` and `max_concurrent_positions` for every scenario/window, and decide whether additive fixed-size return should be capital-normalized.
6. Add listing-aware data-quality fields so pre-listing unavailable periods are not mixed with true internal data gaps.
7. Re-audit `next_open` vs `next_close` after the timestamp and reproducibility fixes.
