# Execution & Leakage Audit Report

Audit date: 2026-05-04  
Scope audited:

- `volatility_expansion_backtest.py`
- `scripts/backtest_executable_vol_expansion.py`
- `scripts/run_vol_expansion_backtest.py`
- Available OHLCV files under `data/ohlcv` and `data/1h`
- Validation helper: `research/volatility_expansion/src/audit_checks.py`
- Data-check output: `research/volatility_expansion/reports/audit_checks.json`

## Category Verdicts

| Category | Verdict | Notes |
| --- | --- | --- |
| 1. Lookahead leakage | PASS with caveats | Breakout levels are shifted to previous candles in all audited implementations. Signal candle OHLCV is used only to decide after close, which is allowed. Caveat: volume "z-score" is not actually implemented; only volume multiple filters exist. |
| 2. Execution timing | FAIL | `scripts/run_vol_expansion_backtest.py` implements `next_close` as `i + 2`, not the next candle close. The script variants also use intrabar exits after entry. |
| 3. Friction | PARTIAL FAIL | Base and stress friction differ and net R is after friction. Top-level script applies adverse price adjustment by side. Script variants subtract round-trip bps from R instead of applying side-specific adverse entry/exit prices, so directional slippage behavior is not directly auditable at the price level. |
| 4. TP/SL sequencing | FAIL | Script variants assume high/low intrabar stop/target touches and exact stop/target fills. They do check stop before target when both are touched, but the intrabar touch/fill model violates the project constraint against intrabar assumptions. |
| 5. Data quality | PARTIAL PASS | Available 1h files have UTC timestamps, no duplicate timestamps, and no missing 1h candles. No 4h files are present, so 1h/4h consistency cannot be checked. Data validation is not integrated into the backtest runners. |
| 6. Metrics | FAIL | Profit factor is missing. `scripts/run_vol_expansion_backtest.py` calculates realized equity/drawdown by `entry_time`, not `exit_time`. Best-trade removal is not implemented. Some trade/year calculations use first/last trade dates rather than the full requested test range. |
| 7. Concentration | PARTIAL PASS | Best symbol and best month contribution are calculated in all implementations. Removing best 5/10 trades is not implemented. |

## Critical Issues

1. Intrabar TP/SL assumptions make script-generated results untrustworthy.
   - `scripts/backtest_executable_vol_expansion.py:241-260` checks `high`/`low` inside each candle and exits at exact `stop` or `target`.
   - `scripts/run_vol_expansion_backtest.py:139-152` does the same and assigns fixed `-1.0R` or `target_r`.
   - This violates the explicit constraint: no intrabar assumptions and no ideal stop/target fills inside a candle. Even with stop-first sequencing, the model assumes the candle path and executable price.

2. `next_close` entry timing is wrong in `scripts/run_vol_expansion_backtest.py`.
   - `scripts/run_vol_expansion_backtest.py:126` sets `entry_i = i + 1 if entry_mode == "next_open" else i + 2`.
   - The project rule says next-close entry must be the close of the next candle after the signal candle, i.e. `i + 1` close, not `i + 2` close.
   - This invalidates any next-close comparison produced by that runner.

3. Realized drawdown chronology is wrong in `scripts/run_vol_expansion_backtest.py`.
   - `scripts/run_vol_expansion_backtest.py:202` sorts trades by `entry_time`.
   - `scripts/run_vol_expansion_backtest.py:218` calculates max drawdown on that order.
   - PnL is realized at exit, so portfolio-level realized drawdown should be ordered by `exit_time`, especially because symbols can overlap.

4. Top-level pass/fail does not stress-test next-close.
   - `volatility_expansion_backtest.py:443-447` passes only `stress_next_open` into `pass_fail`.
   - `volatility_expansion_backtest.py:379-398` therefore cannot detect whether `stress_next_close` collapses.
   - This leaves a required execution mode outside the final gate.

## Medium Issues

1. Friction is not consistently modeled as adverse execution prices.
   - Top-level implementation is directionally correct:
     - long entry worse / short entry worse: `volatility_expansion_backtest.py:178-180`
     - long exit worse / short exit worse: `volatility_expansion_backtest.py:183-185`
   - Script variants use a round-trip R subtraction:
     - `scripts/backtest_executable_vol_expansion.py:195-197`
     - `scripts/run_vol_expansion_backtest.py:113-115`
   - This makes net R after costs, but it does not explicitly prove side-specific slippage direction or exit-price degradation.

2. Profit factor is not implemented.
   - No audited summary function reports profit factor or handles zero-loss/zero-win cases.
   - Relevant summary functions:
     - `volatility_expansion_backtest.py:303-339`
     - `scripts/backtest_executable_vol_expansion.py:310-350`
     - `scripts/run_vol_expansion_backtest.py:199-223`

3. Removing best 5/10 trades is not implemented.
   - Concentration checks exist by symbol/month, but no audited implementation computes EV after excluding the best 5 and best 10 trades.

4. Data quality checks are external, not enforced before backtesting.
   - `volatility_expansion_backtest.py:139-142` sorts timestamps but does not fail on gaps.
   - `scripts/backtest_executable_vol_expansion.py:144` silently drops duplicate timestamps.
   - No runner blocks execution when candles are missing.

5. Trade/year calculation is inconsistent across implementations.
   - Top-level uses requested start/end range: `volatility_expansion_backtest.py:316`.
   - `scripts/backtest_executable_vol_expansion.py:327-329` uses first entry to last exit.
   - `scripts/run_vol_expansion_backtest.py:203-215` uses first/last entry span.
   - The project requirement is better served by a fixed test-window denominator.

6. Volume z-score is not present.
   - Top-level uses `volume_mean_prev` and a max-volume multiple: `volatility_expansion_backtest.py:160,166`.
   - `scripts/backtest_executable_vol_expansion.py:163,185` uses a prior mean multiple.
   - `scripts/run_vol_expansion_backtest.py:78,94` uses a prior median and expansion multiple.
   - These may be acceptable crude OHLCV filters, but they are not z-scores.

## Minor Issues

1. Duplicate handling is silent.
   - `volatility_expansion_backtest.py:134` and `scripts/backtest_executable_vol_expansion.py:144` drop duplicates instead of reporting them.

2. `numpy` is imported but unused in the top-level script.
   - `volatility_expansion_backtest.py:12`

3. There are multiple divergent backtest entry points.
   - This increases audit risk because results depend on which runner is used.
   - Files: `volatility_expansion_backtest.py`, `scripts/backtest_executable_vol_expansion.py`, `scripts/run_vol_expansion_backtest.py`

4. Current report directory is empty.
   - `reports/vol_expansion` exists but contains no report files at audit time.

## Checklist Detail

### 1. Lookahead Leakage

PASS with caveats.

- Breakout high/low uses previous candles:
  - `volatility_expansion_backtest.py:158-159`
  - `scripts/backtest_executable_vol_expansion.py:164-165`
  - `scripts/run_vol_expansion_backtest.py:79-80`
- ATR is known at signal time:
  - Top-level and `scripts/backtest_executable_vol_expansion.py` use prior ATR via `.shift(1)`: `volatility_expansion_backtest.py:157`, `scripts/backtest_executable_vol_expansion.py:162`.
  - `scripts/run_vol_expansion_backtest.py:73` includes the signal candle in ATR. This is not future leakage because the signal is evaluated after candle close, but it is less conservative than prior-only ATR.
- Signal candle data is used only after close:
  - Current close/range/volume are used in signal logic, but entries occur later.
- No future candle is used for signal generation found in the audited code.

### 2. Execution Timing

FAIL.

- No trade enters inside the signal candle in the audited implementations.
- `next_open` uses next candle open:
  - `volatility_expansion_backtest.py:212-214`
  - `scripts/backtest_executable_vol_expansion.py:218-229`
  - `scripts/run_vol_expansion_backtest.py:126-130`
- `next_close` is correct in:
  - `volatility_expansion_backtest.py:212-217`
  - `scripts/backtest_executable_vol_expansion.py:218-229`
- `next_close` is incorrect in:
  - `scripts/run_vol_expansion_backtest.py:126-130`
- Trades without valid next candle are skipped or blocked:
  - Top-level loop condition: `volatility_expansion_backtest.py:205`
  - Script checks: `scripts/backtest_executable_vol_expansion.py:221-222`, `scripts/run_vol_expansion_backtest.py:127-128`

### 3. Friction

PARTIAL FAIL.

- Base and stress friction are different:
  - `volatility_expansion_backtest.py:414-415`
  - `scripts/backtest_executable_vol_expansion.py:68-71`
  - `scripts/run_vol_expansion_backtest.py:29-32`
- Net R is after friction:
  - `volatility_expansion_backtest.py:254-255`
  - `scripts/backtest_executable_vol_expansion.py:263-264`
  - `scripts/run_vol_expansion_backtest.py:157-173`
- Side-specific adverse price adjustment is confirmed only in the top-level script:
  - `volatility_expansion_backtest.py:178-185`

### 4. TP/SL Sequencing

FAIL.

- Stop-first sequencing is present in the script variants:
  - Longs and shorts both compute stop/target hits: `scripts/backtest_executable_vol_expansion.py:245-250`
  - Stop is checked before target: `scripts/backtest_executable_vol_expansion.py:252-260`
  - Same pattern in `scripts/run_vol_expansion_backtest.py:141-152`
- However, both variants use intrabar high/low and exact stop/target fills, so this category fails under the project rules.
- Top-level script avoids intrabar high/low exit checks and uses close-triggered next-open exits: `volatility_expansion_backtest.py:240-252`.

### 5. Data Quality

PARTIAL PASS.

`research/volatility_expansion/src/audit_checks.py` found:

- `data/ohlcv`: 9 symbols, 29,185 rows each, UTC, no duplicates, no missing 1h steps, from `2023-01-01T00:00:00+00:00` to `2026-05-01T00:00:00+00:00`.
- `data/1h`: 10 symbols, 29,272 rows each, UTC, no duplicates, no missing 1h steps, from `2023-01-01T00:00:00+00:00` to `2026-05-04T15:00:00+00:00`.
- 4h consistency was not checked because no 4h files are available.

### 6. Metrics

FAIL.

- Max drawdown is chronological by exit time in:
  - Top-level runner: `volatility_expansion_backtest.py:438-439`
  - `scripts/backtest_executable_vol_expansion.py:325-346`
- Max drawdown is not realized-exit chronological in:
  - `scripts/run_vol_expansion_backtest.py:202-218`
- Losing streak implementation is mechanically correct in all audited files:
  - `volatility_expansion_backtest.py:291-300`
  - `scripts/backtest_executable_vol_expansion.py:298-307`
  - `scripts/run_vol_expansion_backtest.py:188-196`
- Profit factor is missing.
- By-symbol and by-month reports aggregate net R / R result:
  - `volatility_expansion_backtest.py:354-366`
  - `scripts/backtest_executable_vol_expansion.py:459-467`
  - `scripts/run_vol_expansion_backtest.py:314-325`

### 7. Concentration

PARTIAL PASS.

- Best symbol contribution is calculated:
  - `volatility_expansion_backtest.py:317-338`
  - `scripts/backtest_executable_vol_expansion.py:330-349`
  - `scripts/run_vol_expansion_backtest.py:205-221`
- Best month contribution is calculated:
  - `volatility_expansion_backtest.py:318-338`
  - `scripts/backtest_executable_vol_expansion.py:331-349`
  - `scripts/run_vol_expansion_backtest.py:206-221`
- Removing best 5/10 trades is not implemented.

## Required Fixes Before Trusting Results

1. Choose one canonical Phase 1-2 runner and remove or clearly deprecate the divergent copies.
2. Remove intrabar high/low TP/SL fills from the canonical runner, or label those outputs as non-compliant and unusable for this project.
3. Fix `scripts/run_vol_expansion_backtest.py` next-close timing if that runner remains in scope: next-close must be `i + 1` close.
4. Calculate realized portfolio drawdown in chronological `exit_time` order.
5. Add profit factor with safe handling for zero-loss and zero-win cases.
6. Add best-5 and best-10 trade removal metrics.
7. Include `stress_next_close` in the final stress pass/fail gate.
8. Enforce data-quality checks before backtesting: duplicates, gaps, timezone, minimum rows, and symbol coverage.
9. Add or fetch 4h data before claiming 1h/4h consistency.

## Trust Status

Current reports are **not trustworthy** as production-feasibility evidence.

Reason: the script variants contain non-compliant intrabar execution assumptions, one runner has incorrect next-close timing, required metrics are missing, and no current validation report exists under the active `reports/vol_expansion` directory. The top-level `volatility_expansion_backtest.py` is closer to the requested execution model because it avoids intrabar TP/SL fills, but its final gate is incomplete and it still lacks required audit metrics. Treat any current or previously generated result as exploratory only until the required fixes are completed.
