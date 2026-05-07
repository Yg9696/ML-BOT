# Panic-Reversal Bot Live Feasibility Plan

## Scope

This document is an implementation plan only. It does not change the researched strategy and does not place live orders.

Selected configuration:

- Timeframe: 15m
- Direction: long only
- Event return: close-to-close return <= -3%
- Volatility regime: high prior realized-volatility regime
- Breadth: simultaneous 2% down events >= 4
- BTC condition: BTC 24h return < 0
- Trend: EMA50 slope < 0 and EMA200 slope < 0
- Entry: next_close
- Exit: pure_horizon_12
- Stacking: enabled
- Concurrent cap: 10
- Symbol cap: 2
- Same-timestamp selection: largest_drop
- Allocation: 2% equity per accepted position
- Max exposure target: about 20%

## 1. Live Data Requirements

Required Binance USD-M Futures endpoints:

- `GET /fapi/v1/klines`
  - Pull 15m OHLCV for all traded symbols.
  - Required fields: open time, open, high, low, close, volume, close time.
- `GET /fapi/v1/time`
  - Clock synchronization and drift checks.
- `GET /fapi/v1/exchangeInfo`
  - Symbol status, precision, min notional, step size, tick size.
- `GET /fapi/v2/account` or equivalent signed account endpoint
  - Equity, available balance, open exposure.
- `GET /fapi/v2/positionRisk`
  - Reconcile open positions.
- Signed order endpoints later, only after paper/dry-run phases.

Candle polling:

- Poll all symbols shortly after each 15m boundary.
- Treat a candle as complete only when Binance returns the candle whose `close_time` is earlier than current exchange time.
- Use one canonical candle close timestamp across all symbols.
- Do not compute signals until every active symbol has the same completed 15m candle timestamp or the fetch timeout policy marks the cycle invalid.

Alignment:

- Maintain an in-memory/data-store candle table keyed by `symbol + candle_open_time`.
- A signal cycle is valid only when all required symbols have the same completed candle.
- Breadth must be computed on that shared completed candle timestamp.

Duplicate-signal prevention:

- Store `processed_signal_timestamps` keyed by `strategy_version + candle_close_time`.
- If a cycle was processed, never generate another entry batch for the same timestamp.
- Restart-safe duplicate control must live in persistent storage, not only process memory.

## 2. Signal Engine

Signal calculation order for each completed event candle:

1. Fetch/confirm the completed 15m candle for all symbols.
2. Update rolling OHLCV history.
3. Compute close-to-close return for each symbol.
4. Compute event candidates where return <= -3%.
5. Compute breadth count: number of symbols with return <= -2% at the same candle timestamp.
6. Compute BTC 24h return using BTC close now versus BTC close 96 completed 15m candles earlier.
7. Compute EMA50 and EMA200 from completed candles.
8. Compute EMA50 slope and EMA200 slope over 24 bars.
9. Compute realized volatility prior-only:
   - Use returns strictly before the event candle.
   - Do not include the event candle return in `realized_vol_96`.
10. Compute high-volatility regime using the existing rolling threshold method.
11. Apply all filters.
12. If candidates exceed available caps, rank by largest drop magnitude.

Breadth:

- `breadth_count = count(symbol_return <= -2%)` across the full symbol universe at the same completed candle close.
- Breadth is not computed only over symbols that pass the -3% event threshold.

Prior-only realized volatility:

- At event candle `t`, volatility uses returns through `t-1`.
- Threshold is also shifted so it is known before the event candle is evaluated.

## 3. Execution Timing

`next_close` live meaning:

- The bot does not enter immediately after the event candle closes.
- It waits one full 15m candle.
- Entry is attempted at the close of the next candle using a market order.
- In live trading this means the order is submitted immediately after the next candle is confirmed closed.

Timeline:

1. Event candle opens at `T`.
2. Event candle closes at `T + 15m`.
3. Bot calculates signal after `T + 15m` close is confirmed.
4. Next candle opens at `T + 15m`.
5. Bot waits through the next candle.
6. Next candle closes at `T + 30m`.
7. Bot submits market buy after confirming the `T + 15m` candle has closed.
8. Position exit is scheduled after 12 candles from entry timing.
9. Exit market order is submitted after the 12th holding candle closes.

Operational note:

- Backtest entry price uses next candle close. Live cannot trade exactly at the close print without latency. Paper mode must measure close-to-order slippage.

## 4. Position Management

State tracked per position:

- Position ID
- Strategy version
- Symbol
- Signal candle open/close time
- Entry candle open/close time
- Entry order ID
- Entry price
- Allocation percent
- Notional size
- Expected exit candle close time
- Exit order ID
- Status: pending_entry, open, pending_exit, closed, rejected, manual_review

Stacking:

- Multiple positions in the same symbol are allowed up to symbol cap 2.
- Total open strategy positions capped at 10.
- If caps are full, new signals are skipped and logged.

Same-timestamp selection:

- Build candidate list for the signal timestamp.
- Remove candidates blocked by symbol cap.
- Sort remaining candidates by largest absolute event drop.
- Accept until concurrent cap 10 is reached.
- Persist both accepted and skipped candidates.

Scheduled exits:

- Each accepted position gets a deterministic exit candle close timestamp.
- Exit is not recalculated by indicator state.
- Exit is market-only unless management changes later.

## 5. Risk Engine

Sizing:

- Allocate 2% of account equity per accepted position.
- Use equity snapshot from the cycle before order placement.
- Cap total strategy exposure at about 20% through max 10 concurrent positions.

Emergency controls:

- Hard max open positions: 10.
- Hard max active per symbol: 2.
- Hard max notional exposure: configurable, default 20% of account equity.
- Daily realized loss pause: recommended starting threshold 3% account equity.
- Intraday drawdown pause: recommended starting threshold 5% account equity.
- Consecutive order rejection pause: pause after 2 rejections in a cycle.
- Data quality pause: pause if any required symbol is missing the completed candle.
- Spread/slippage pause: pause symbol if estimated spread or recent execution slippage exceeds configured threshold.

Cluster/slippage stress mode:

- When breadth >= 5, mark the cycle as stress-sensitive.
- When breadth >= 7, mark as extra-stress-sensitive.
- In stress-sensitive cycles, require stricter pre-order checks:
  - current spread below threshold
  - order book available
  - no Binance incident detected
  - account exposure below emergency cap

## 6. Persistence

Store:

- Strategy config and config version
- Symbol universe and exchange filters
- Completed candles used for signals
- Processed signal timestamps
- Candidate signals
- Accepted signals
- Skipped signals and skip reason
- Open positions
- Closed trades
- Pending orders
- Order responses
- Account/equity snapshots
- Cycle logs
- Error logs

Minimum persistence technology:

- SQLite is sufficient for local/paper mode.
- Postgres is preferred for hosted live operation.

State must survive:

- Process restart
- Render redeploy
- Temporary Binance outage
- Duplicate scheduler trigger

## 7. Failure Modes

Binance API unavailable:

- Do not trade.
- Mark cycle incomplete.
- Retry with bounded backoff.

Delayed candle fetch:

- Wait a short grace period after candle boundary.
- If any symbol remains missing, skip the entire signal cycle.

Duplicate poll/webhook:

- Check persistent `processed_signal_timestamps`.
- Idempotently return without generating new orders.

Render restart:

- Reload open positions and processed timestamps.
- Reconcile Binance positions before resuming.

Partial order fill:

- Store partial fill.
- Decide whether to continue working the remainder only in later execution design.
- Initial implementation should use market IOC where appropriate and reconcile actual filled notional.

Order rejected:

- Mark position as rejected.
- Do not retry blindly.
- Log exchange error and trigger risk pause if repeated.

Position already exists:

- Count existing strategy positions toward symbol cap and concurrent cap.
- Reconcile by strategy position IDs, not just symbol net exposure.

Symbol delisted or non-trading:

- Remove from active universe after exchangeInfo confirms non-trading status.
- Do not compute breadth from missing/delisted symbols without explicit universe version change.

Large spread/slippage:

- Skip symbol or full cycle depending on severity.
- Log as skipped due to execution quality.

Clock drift:

- Compare local clock to Binance server time.
- Pause if drift exceeds threshold, e.g. 1 second.

## 8. Implementation Phases

Phase 0: dry-run signal logging

- No API keys.
- Poll public candles.
- Log signals, accepted/skipped decisions, scheduled exits.
- Compare against historical script for matching signal counts.

Phase 1: simulated orders

- Still no live orders.
- Simulate market entry/exit at fetched candle close plus measured slippage model.
- Validate state store, caps, scheduled exits, and restart recovery.

Phase 2: read-only Binance diagnostics

- Use read-only API permissions.
- Pull account, position risk, exchange filters.
- Validate sizing calculations and symbol precision.

Phase 3: paper trading for fixed event count

- Run until at least 50 accepted live/paper events or 3 months, whichever comes first.
- Compare paper fills to backtest assumptions.

Phase 4: small-capital pilot

- Enable live orders with strict notional cap.
- Start below target sizing, e.g. 0.25%-0.5% per accepted position.
- Require manual review before scaling.

Phase 5: target sizing pilot

- Move toward 2% only if slippage, rejection rate, and drawdown remain within go/no-go thresholds.

## 9. Required Code Modules

Suggested structure:

```text
src/panic_reversal/
  config.py
  data_fetcher.py
  candle_store.py
  signal_engine.py
  risk_engine.py
  position_manager.py
  execution_client.py
  state_store.py
  scheduler.py
  reporting.py
  reconciliation.py
  run_bot.py
tests/
  test_signal_engine.py
  test_risk_engine.py
  test_position_manager.py
  test_reconciliation.py
```

Module responsibilities:

- `data_fetcher`: Binance kline/time/exchangeInfo access.
- `candle_store`: rolling OHLCV storage and candle completeness.
- `signal_engine`: deterministic indicator and signal calculation.
- `risk_engine`: sizing, caps, emergency pauses.
- `position_manager`: open/closed position state and scheduled exits.
- `execution_client`: market order abstraction, initially paper-only.
- `state_store`: SQLite/Postgres persistence.
- `scheduler`: 15m cycle trigger and idempotency.
- `reporting`: cycle, signal, trade, and risk reports.
- `reconciliation`: compare local state to Binance account/positions/orders.

## 10. Go/No-Go Criteria For Live Pilot

Go to paper mode if:

- Live signal engine matches historical backtest on replay fixtures.
- No duplicate signals across restarts.
- Candle alignment failures are detected and skipped.
- Caps and largest-drop selection pass unit tests.
- Scheduled exits fire exactly after 12 candles in simulation.

Go to small live pilot if:

- Paper mode records at least 50 accepted events or 3 months.
- Average live/paper slippage remains within stress assumptions.
- No unresolved state reconciliation errors.
- Order rejection rate below 1%.
- No missed scheduled exits.
- Max paper drawdown within expected range.

Do not go live if:

- Data gaps are frequent during signal windows.
- Entry latency makes fills materially worse than stress assumptions.
- Binance outage handling is not fully idempotent.
- Restart recovery cannot reconcile open positions.
- Realized exposure can exceed emergency cap.

