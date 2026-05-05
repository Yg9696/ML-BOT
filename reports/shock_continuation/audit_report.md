# Shock Continuation OHLCV-Only Audit

## Scope

Focused audit of `scripts/shock_continuation_backtest.py` on branch
`research/shock-continuation-phase1-agent1`.

The audit checks whether the negative Phase 1-2 result is mechanically
trustworthy. It does not improve the strategy, add filters, add ML, add
liquidation data, add open interest, or optimize parameters.

## Inputs Reviewed

- Source: `scripts/shock_continuation_backtest.py`
- Reports: `reports/shock_continuation/*.csv`
- Summary: `reports/shock_continuation/phase_1_2_summary.json`
- Full local trade file: `reports/shock_continuation/trades.csv`
- Data quality report: `reports/shock_continuation/data_quality_report.csv`

## Result Reconciliation

- Full trade rows in summary and `trades.csv`: 1,616,265.
- Scenario rows in summary and `overall_results.csv`: 768.
- Best reported EV/trade across all scenarios: -0.0682369661 R.
- Data quality totals across all symbols/timeframes:
  - missing rows: 0
  - duplicate rows: 0
  - large gaps: 0
  - zero-volume rows: 0
  - bad OHLC rows: 0

## Lookahead

Status: PASS.

- Shock candle body, range, close location, and return use the signal candle's
  own OHLC values. These are known only after the signal candle closes.
- ATR uses `rolling(...).mean().shift(1)`, so the signal candle true range is
  excluded.
- ATR percent divides prior ATR by `prev_close`, so it does not use future
  closes.
- Volume z-score uses prior rolling mean/std with `.shift(1)`, so signal
  candle volume is compared against prior candles only.
- Entries use `entry_i = signal_i + 1`, so no trade enters inside the signal
  candle.

## Execution

Status: PASS with a timestamp-label caveat.

- `next_open` entry uses the next candle open.
- `next_close` entry uses the next candle close.
- Both modes use `signal_i + 1`; no entry occurs on the shock candle.
- No limit-order or limit-fill assumption is present.
- Reconciliation check found zero cases where `entry_time <= signal_time`.
- Caveat: for `next_close`, `entry_time` records the next candle's timestamp
  column, which is the candle open time in Binance kline data. The price used is
  the next candle close, and exit checking starts after that candle, so R results
  are not affected; the timestamp label is imprecise.

## TP/SL

Status: PASS.

- Long stop is `entry - risk`; long target is `entry + target_r * risk`.
- Short stop is `entry + risk`; short target is `entry - target_r * risk`.
- Risk is `stop_atr_mult * atr_at_signal`.
- Recomputed max absolute formula error across the full local trade file:
  - risk: approximately 4.55e-13
  - stop: approximately 4.37e-11
  - target: approximately 4.37e-11
- Same-candle ambiguity is conservative: stop is checked before target.
- Time stop defaults to the configured last holding candle close and then
  applies exit slippage and fee.

## Friction

Status: PASS.

- Entry slippage is adverse:
  - long entry price is increased
  - short entry price is decreased
- Exit slippage is adverse:
  - long exit price is decreased
  - short exit price is increased
- Fee is charged on entry and exit with
  `fee_rate * (entry + exit_price) / risk`.
- Net R is `gross_r - fee_r`.
- Stress settings are worse than base:
  - base: 4 bps taker fee per side, 5 bps slippage per side
  - stress: 8 bps taker fee per side, 10 bps slippage per side
- Matched scenario check found 0 cases where stress EV/trade exceeded base
  EV/trade.

## Metrics

Status: PASS with reporting caveats.

- EV/trade, total EV, by-symbol EV, and by-month EV all use `net_r`.
- Recomputed the best reported row from full local trades:
  - reported trades: 535
  - recomputed trades: 535
  - reported EV/trade: -0.0682369661 R
  - recomputed EV/trade: -0.0682369661 R
  - reported total EV: -36.5067768730 R
  - recomputed total EV: -36.5067768730 R
- By-symbol recomputation of the first by-symbol row matched reported
  trade count and total EV exactly.
- Drawdown is chronological within each scenario by `exit_time` then `symbol`.
- Trades/year is computed using first trade to last trade within each scenario,
  not the pinned one-year research window. This can inflate annualized trade
  frequency for sparse scenarios. It does not change the negative EV result.
- Best symbol/month contribution shares are `NaN` when total scenario EV is
  negative. This is acceptable for pass/fail concentration checks but leaves
  concentration less informative for losing scenarios.

## Data Quality

Status: PASS.

- `data_quality_report.csv` checks expected timestamp grid, missing timestamps,
  duplicate timestamps, large gaps, zero-volume rows, and invalid OHLC rows.
- For the pinned range `2025-05-04T00:00:00Z` to
  `2026-05-04T00:00:00Z`, the report shows:
  - 15m total rows: 315,360 / expected 315,360
  - 1h total rows: 78,840 / expected 78,840
  - missing rows: 0
  - large gaps: 0
  - duplicate rows: 0
  - zero-volume rows: 0
  - bad OHLC rows: 0

## Critical Issues

None found.

No lookahead, execution, TP/SL, friction, data-gap, or net-R aggregation issue
was found that would explain away the negative result.

## Medium Issues

1. `next_close` trade timestamps are imprecise.
   - The entry price is correctly taken from the next candle close, but
     `entry_time` stores that candle's open timestamp. This affects time labels
     and could slightly affect duration-style diagnostics, but not EV/trade.

2. Trades/year uses scenario first-to-last trade span rather than the pinned
   one-year window.
   - This can overstate trade frequency for sparse scenarios. In this run the
     best EV rows still fail on EV, so the scenario conclusion is unchanged.

## Minor Issues

1. Best symbol/month contribution shares are not populated for negative-total-EV
   scenarios.
   - The concentration pass criteria are not meaningful when total EV is
     negative, but the report could still include absolute contribution shares
     for diagnostics.

2. `RET_MULT` is tied to `RANGE_MULT`.
   - This is documented in the summary JSON and avoids adding an unrequested
     optimization axis, but it should be kept explicit in any research notes.

## Trust Assessment

The negative Shock Continuation OHLCV-only result is trustworthy for Phase 1-2.

The strongest reported scenario is still negative:

- `1h`, `model_b`, base friction, `next_open`
- 535 trades
- EV/trade: -0.0682369661 R
- Profit factor: 0.8999693027

The audit did not find a mechanical bug that would plausibly turn the result
positive. The scenario fails because the tested shock-continuation rules lose
after realistic execution friction, not because of an obvious backtest
implementation error.

## Required Fixes

No fixes are required before accepting the negative Phase 1-2 conclusion.

Recommended cleanup before future reuse:

- Store actual close timestamps for `next_close` entries, or add a separate
  `entry_bar_timestamp` field to avoid ambiguity.
- Compute an additional `trades_per_pinned_year` metric using the configured
  research date range.
- Add absolute symbol/month contribution diagnostics for negative scenarios.
