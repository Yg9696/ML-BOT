# Panic-Reversal Extended Validation Re-Audit After Fix

Branch audited: `research/panic-reversal-extended-validation-agent1`

Latest Agent 1 commit audited: `a7966c5ed90ae32636615161148824e5ca9ccb6e`

Audit role: Agent 2 - Execution & Leakage Auditor

## 1. What I Checked

- Ran the requested command:

  `python scripts/panic_reversal_extended_validation.py --start 2020-01-01T00:00:00+00:00 --end 2026-05-04T00:00:00+00:00`

- Confirmed the regenerated headline primary scenario:
  - `next_close / pure_horizon_12 / base`
  - `total_trades = 966`
  - `cumulative_return_units = 26.8981`
  - `average_return_pct = 2.7845`
  - `minimum_pass = true`
- Reviewed `scripts/panic_reversal_extended_validation.py` for lookahead leakage, timestamp semantics, friction, additive return accounting, data quality, and exposure logic.
- Checked `trade_returns.csv`, `trade_returns_sample.csv`, `overall_results.csv`, `by_year.csv`, `by_month.csv`, `by_symbol.csv`, `execution_comparison.csv`, `equity_curve.csv`, `data_quality_report.csv`, `run_config.json`, and `extended_validation_summary.json`.
- Recomputed primary scenario count, cumulative return, average return, max drawdown, concentration metrics, and exposure from `trade_returns.csv`.

## 2. Critical Issues

No critical implementation issue found after the reproducibility fix.

The prior audit blocker is resolved: current code and reports now reproduce the strong primary result within exact/rounding tolerance.

## 3. Medium Issues

### High concurrent exposure is a real strategy risk

The primary full-period row reports:

- `average_concurrent_positions = 8.80`
- `p95_concurrent_positions = 27`
- `max_concurrent_positions = 41`

This is not a code bug. It is a major execution/capital assumption for Phase 2. The reported cumulative return is fixed-size additive return across overlapping positions, not account-level compounded return under a leverage/margin constraint.

### Recent-year robustness is weaker than the full-period headline

The primary scenario remains positive by checked years, but 2023 has only 22 trades and `0.1149` cumulative units. 2025 and 2026 YTD are positive but have high month-contribution ratios because gains are concentrated in a few months. This does not invalidate Phase 1, but Phase 2 should not lean only on the full-period aggregate.

### Same-close cross-symbol inputs require operational care

Breadth and BTC 24h return use same completed 15m candle information. This is not future leakage because entries occur after the event candle close, but live execution must assume reliable full-basket OHLCV availability before placing the next-candle entry.

## 4. Minor Issues

### Data contains a few zero-volume candles

`data_quality_report.csv` reports 55 total zero-volume rows across all symbols. There are no true internal missing rows, no duplicate rows, no large gaps, and no bad OHLC rows. The zero-volume rows are small relative to the dataset, but Phase 2 should decide whether to exclude or explicitly tolerate them.

### `next_close` is slower than `next_open` by construction

The fixed timestamp semantics are correct:

- `next_open` entry is 15 minutes after `signal_time`, at `signal_close_time`.
- `next_close` entry is 30 minutes after `signal_time`, or 15 minutes after `signal_close_time`.

This means `next_close` waits one full extra candle after the signal closes. The comparison is valid, but the semantic difference should stay explicit in reports and strategy docs.

## 5. Whether Fixed Results Are Trustworthy

Yes, for Phase 1 extended-validation purposes, with an exposure caveat.

I do not see direct lookahead leakage in the repaired implementation:

- Event return uses the completed signal candle and entry is after that close.
- Realized volatility is prior-only via `returns.shift(1)`.
- Realized-vol threshold is rolling and shifted.
- Breadth uses same completed candle across symbols.
- BTC 24h return uses BTC close at the same completed signal candle.
- EMA50/EMA200 slopes use current and prior completed closes only.
- `next_close` and `next_open` timestamps now reflect their intended execution semantics.
- Entry and exit slippage are adverse for a long-only reversal strategy.
- Stress friction is strictly worse than base in all matched overall rows checked.
- Cumulative return is additive, and max drawdown is based on chronological net returns.
- By-symbol, by-month, and concentration metrics reconcile to primary trade-level net returns.

The primary fixed result is reproducible and materially strong:

- Base `next_close / pure_horizon_12`: `+26.8981` cumulative units, 966 trades.
- Stress matched row: `+25.1321` cumulative units.
- Best month contribution: `30.0%`.
- Best symbol contribution: `17.0%`.

The result should be treated as a credible Phase 1 candidate, not yet as a deployable bot, because concurrent exposure can reach 41 simultaneous fixed-size positions.

## 6. Remaining Required Fixes Before Phase 2

1. Define capital, margin, and max-concurrent-position constraints, then convert additive units into account-level returns.
2. Add Phase 2 execution stress for clustered panic events: liquidity, spread, order fan-out, and partial failure assumptions.
3. Decide how to handle zero-volume candles in feature windows.
4. Keep `run_config.json` and exact command/hash with every future report bundle.
5. Re-test the primary configuration under explicit exposure caps, because the uncapped fixed-size result may overstate deployable capacity.
