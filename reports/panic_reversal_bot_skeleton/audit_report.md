# Panic-Reversal Dry-Run Bot Skeleton Audit

Branch audited: `implementation/panic-reversal-dry-run-skeleton-agent1`

Audit role: Agent 2 - Execution & Leakage Auditor

## 1. What I Checked

- Reviewed `bot/panic_reversal/config.py`, `data_fetcher.py`, `signal_engine.py`, `risk_engine.py`, `position_manager.py`, `state_store.py`, `execution_client.py`, `scheduler.py`, `main.py`, `reporting.py`, and `README.md`.
- Reviewed `reports/panic_reversal_bot_skeleton/implementation_notes.md`.
- Verified safety guards by running:
  - `BOT_MODE=live LIVE_TRADING_ENABLED=false python -m bot.panic_reversal.main --once`
  - `BOT_MODE=dry_run LIVE_TRADING_ENABLED=true python -m bot.panic_reversal.main --once`
- Ran one local dry-run cycle with temporary state/log directories.
- Checked `.gitignore` coverage for default dry-run state and logs.
- Searched for order endpoints, API key usage, and private Binance paths.

## 2. Critical Issues

### Incomplete Binance candles are not excluded

When `PANIC_BOT_DATA_SOURCE=binance`, `_fetch_binance_klines()` requests recent klines and returns all rows from Binance. Binance kline responses commonly include the currently forming candle. The fetcher does not inspect `close_time`, compare with wall-clock time, or drop the open/incomplete candle.

`fetch_latest_completed_candles()` then treats each symbol's max timestamp as completed and may generate signals from an incomplete 15m candle. This violates the strategy requirement that event return, breadth, BTC context, and EMA filters use only completed candles. It also creates a possible entry schedule from data that was not known at signal close.

This is a blocker before continuous dry-run or paper mode with Binance polling.

### Stale local candles are accepted as actionable

The default local dry-run source uses the max timestamp in local CSV files as `aligned_timestamp` without checking freshness against current UTC time. In my dry-run test on 2026-05-07, the bot processed `aligned_candle_open_time = 2026-05-04T00:00:00Z` as if it were the latest completed candle.

This is dry-run only, so it cannot place real orders, but operationally it can create misleading accepted/skipped signals, pending positions, and run summaries from stale historical data. Continuous dry-run needs a freshness guard.

## 3. Medium Issues

### Missing aligned candles do not block signal generation

The fetcher aligns to the minimum latest timestamp across symbols, trims all frames to `<= aligned_timestamp`, and records `has_aligned_timestamp` in the quality report. However, the scheduler does not fail or skip the cycle if one or more symbols lack that aligned candle due to an internal gap.

`generate_candidates()` silently skips symbols with no row at `aligned_timestamp`, and breadth is computed from the available rows. This can silently corrupt breadth and candidate generation. It is likely conservative when a missing symbol would have been down, but it still breaks the required all-symbol timestamp alignment.

### Dry-run PnL does not model research friction

The execution stub is safe, but `PositionManager` records entry/exit at raw close prices and computes `net_return = price / entry_price - 1.0` without taker fees or adverse slippage. The validated research result used fee/slippage assumptions. If closed-position logs are later used as performance evidence, they will be optimistic relative to the research accounting.

### State writes are not atomic

`StateStore` writes JSON files directly. `record_decisions()` marks a signal as processed before appending an accepted position. A crash between those two writes could permanently mark a signal processed without creating the corresponding pending position. This is an operational durability risk for restart safety.

### Default state/log directories are ignored, but override paths are not

`.gitignore` covers `bot/panic_reversal/state/` and `bot/panic_reversal/logs/`. If a user sets `PANIC_BOT_STATE_DIR` or `PANIC_BOT_LOG_DIR` elsewhere, generated state can still be accidentally tracked unless the override path is also ignored.

## 4. Minor Issues

### `selection_policy` is configured but not enforced generically

`selection_policy` defaults to `largest_drop`, and `RiskEngine` does sort by largest absolute drop. It does not validate that the configured policy is `largest_drop`; if the config changes later, the engine will still silently use largest-drop ordering.

### Skipped signals are only logged when there are candidates

This is reasonable, but run summaries with zero candidates do not explain whether there were no event candles, no breadth, no high-vol regime, or missing/stale data. More diagnostic counters would help debugging without changing strategy logic.

### README is accurate on order safety but underspecifies data freshness

The README correctly states dry-run safety and no live order adapter. It does not warn that local files must be fresh or that Binance current-candle exclusion is required before continuous polling.

## 5. Whether The Dry-Run Skeleton Is Safe And Faithful

Partially.

Order safety is good:

- `BOT_MODE` defaults to `dry_run`.
- `LIVE_TRADING_ENABLED=true` blocks startup.
- `BOT_MODE` other than `dry_run` blocks startup.
- No API keys are required or read.
- No private Binance/order endpoint is called.
- `ExecutionClientStub.place_order()` always raises.
- `simulate_market_order()` only returns local dry-run fill records.

Strategy logic is mostly faithful when the input candle is truly complete and aligned:

- 15m timeframe.
- Long-only positions.
- Event return `<= -3%`.
- Realized-vol regime is prior-only via shifted returns.
- Breadth uses same timestamp across symbol frames.
- BTC 24h return is same completed-candle context.
- EMA50/EMA200 slope semantics match the research code.
- Primary entry is `next_close`, with pending entry due one full candle after signal close.
- Exit due is 12 completed 15m candles after entry.
- Stacking, concurrent cap 10, symbol cap 2, largest-drop ordering, and 2% allocation are implemented.
- Processed signal keys and open positions are durable across restarts under the default state path.

However, the skeleton is not yet operationally safe enough for continuous dry-run/paper mode because incomplete Binance candles and stale local candles can be treated as actionable. Those are data-timing failures, not strategy improvements.

## 6. Required Fixes Before Continuous Dry-Run/Paper Mode

1. Drop or reject incomplete Binance candles using `close_time` and current UTC time before computing aligned candles.
2. Add a freshness guard for local and Binance data, e.g. aligned candle close must be within an expected tolerance of current UTC time for continuous mode.
3. Fail the cycle if any configured symbol lacks the aligned completed candle, instead of only recording `has_aligned_timestamp=false`.
4. Include fee/slippage-adjusted simulated PnL fields, or clearly label closed-position `net_return` as raw dry-run return not comparable to research net return.
5. Make state writes atomic enough for restart safety, especially processed-signal marking plus position creation.
6. Add a warning or ignore pattern guidance for custom state/log directories.
7. Add run-summary diagnostic counts for filter-stage skips or data-quality skips.

## Additional Checks Passed

- Safety guard tests failed closed as expected for `BOT_MODE=live` and `LIVE_TRADING_ENABLED=true`.
- A local dry-run cycle completed without placing orders and wrote temporary positions, data-quality, and scheduler-run files.
- Default `bot/panic_reversal/state/` and `bot/panic_reversal/logs/` are ignored by Git.
- The execution client contains no real order implementation.
- Public Binance data access is limited to `/fapi/v1/klines`.
