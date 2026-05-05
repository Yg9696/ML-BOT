# Phase 1-2 Execution and Leakage Audit

Branch audited: `research/vol-expansion-phase1-agent1`

Audit role: Agent 2 - Execution & Leakage Auditor

## 1. What I Checked

- Reviewed `scripts/phase_1_2_backtest.py` for lookahead leakage in breakout levels, ATR, volume z-score, room-to-target, EMA200 trend, and EMA slope.
- Reviewed execution timing for `next_open` and `next_close`, including whether signals enter inside the signal candle.
- Reviewed TP/SL behavior for long and short trades, including same-candle ambiguity.
- Reviewed fee and slippage application for base and stress cases.
- Reviewed Task 2 filters: room-to-target, EMA200 trend plus slope, and EMA distance overextension.
- Reviewed metric construction for net R, drawdown, losing streak, trades/year, by-symbol, by-month, and profit factor.
- Reviewed generated Phase 1-2 artifacts and checked whether large `trades.csv` output is tracked despite `.gitignore`.
- Ran lightweight consistency checks on `reports/phase_1_2/trades.csv` and local OHLCV files.

## 2. Critical Issues

### Room-to-target fallback is optimistic and materially changes the filter

In `simulate_symbol`, if the prior room structure is already behind or inside the breakout, the code treats available room as the full target distance:

- Long: `available_room = structure - entry if structure > entry else target - entry`
- Short: `available_room = entry - structure if structure < entry else entry - target`

This means a breakout beyond the recent swing high/low is not penalized for having no known nearby structure ahead. It is automatically assigned exactly `target_r` room, which passes the default `min_room_r=1.5` whenever `target_r=2.0`.

This is the specific optimistic assumption flagged in the task. In the generated `trades.csv`, this fallback condition appears in 14,140 of 17,238 trade rows, about 82.0% of recorded trades. That makes the room filter much weaker than its stated purpose and can materially overstate trade quality.

Required interpretation: if room to target cannot be confirmed from prior structure, the conservative assumption should not be "full TP distance is available."

## 3. Medium Issues

### `next_close` entry timestamps are one candle too early

For both entry modes, `entry_i = signal_i + 1`. For `next_close`, the price uses `closes[entry_i]`, but the recorded `entry_time` is `timestamps[entry_i]`, which is the open timestamp of that candle, not the close timestamp at which the close price becomes known.

This does not enter inside the signal candle by price, but it mislabels execution time and slightly contaminates duration-based metrics and trade ordering whenever close timestamps matter. A `next_close` fill should be timestamped at the next candle close, or the timestamp convention should be explicitly documented and adjusted in metrics.

### Room filter decides entry using the entry candle price

The room filter is applied after the raw entry price is selected. For `next_open`, it uses the next candle open; for `next_close`, it uses the next candle close. This avoids future candles after the modeled entry, but it means the filter is not purely a signal-close filter.

For `next_close`, the system decides whether the trade passes the room filter using the same close price at which it claims to enter. That assumes a market-on-close style decision and fill with no decision latency. This should be made explicit or changed to a conservative, already-known reference price.

### No explicit gap validation in the loader

The checked local Binance 1h and 4h files have continuous timestamps and no duplicates, but `load_ohlcv`/`ensure_data` do not enforce gap-free series. A future stale or corrupted file could pass silently after sorting/deduplication. This is a data quality control issue before trusting future reruns.

### Generated reports use dynamic default date ranges

`default_start_end()` uses the current UTC time. Unless `--start` and `--end` are always pinned in the run command, repeated runs are not directly reproducible and may silently compare different windows.

## 4. Minor Issues

### `atr_period` is parameterized but effectively fixed

`ParamSet` contains `atr_period`, but the implementation computes only `atr_14_prior`. The current grid only uses `atr_period=14`, so this is not currently corrupting results, but it is a footgun if ATR periods are later expanded.

### Month attribution uses exit month

`by_month` groups trades by `exit_time` month. This is defensible for realized PnL, but it should be documented because some audits expect signal month or entry month.

### Signal timestamps are candle open timestamps

The report language says signals are evaluated after the signal candle close, while `signal_time` records the candle's timestamp column, which appears to be open time. This is a labeling issue and can confuse later execution reviews.

### Large generated CSV is tracked despite `.gitignore`

`.gitignore` contains `reports/phase_1_2/trades.csv`, but commit `af5fed2` tracks that same file. It is about 8.5 MB locally and was added as 17,239 CSV lines. This is not a trading logic issue, but it contradicts the ignore rule and will keep large generated artifacts in Git history unless intentionally accepted.

## 5. Whether Results Are Trustworthy

Not yet.

The core prior-window indicators mostly avoid direct future-candle leakage: breakout high/low, ATR, volume z-score reference windows, room structure, EMA, and EMA slope are not using future rows. TP/SL sequencing is conservative because stop is checked before target for both long and short trades. Fees and slippage are applied against the trader, and stress mode doubles base fee and doubles slippage.

However, the room-to-target fallback is material enough that Phase 1-2 results should not be trusted as evidence of strategy viability. The `next_close` timestamp/decision timing issue also needs correction or explicit conservative documentation before using timing-sensitive metrics.

## 6. Required Fixes Before Continuing

1. Redefine the room-to-target filter conservatively. Do not treat "breakout already beyond recent structure" as full TP room by default. Either reject unknown-room cases, mark them separately, or use a predeclared conservative rule.
2. Re-run Phase 1-2 after fixing room handling and compare trade counts, EV/trade, drawdown, by-symbol, and by-month. Expect trade count and EV to change materially.
3. Correct `next_close` `entry_time` labeling to the actual modeled close time, or add explicit timestamp convention fields so metrics do not treat close-entry trades as entered at candle open.
4. Decide whether the room filter is allowed to use the entry candle price. If not, compute the filter from signal-close information only. If yes, document that the trade is conditional on the next open/close state and model any practical latency conservatively.
5. Add an explicit OHLCV gap check before running reports, even though the currently checked local data files were continuous.
6. Pin start/end dates for report generation so later reruns are comparable.
7. Remove `reports/phase_1_2/trades.csv` from tracking if generated trade logs should remain ignored; otherwise remove the ignore rule and document that large result artifacts are intentionally versioned.

## Additional Checks Passed

- Breakout rolling high/low use shifted prior candles only.
- ATR uses true range through the prior candle via rolling mean shifted by one.
- Volume z-score compares current closed-candle volume to prior rolling mean/std.
- EMA200 and EMA slope use current and past closes only, consistent with signal-known-after-close.
- Entries do not use signal-candle prices.
- Stop wins if stop and target are both touched in the same evaluated candle.
- Long and short stop/target geometry in generated trades matches 1.5 ATR risk and 2R target within floating point tolerance.
- Net R equals gross R minus fee R in generated trades within floating point tolerance.
- Base friction is 4 bps fee and 5 bps slippage per side; stress is 8 bps fee and 10 bps slippage per side.
- EV/trade, by-symbol, by-month, profit factor, and path metrics are based on `net_r`.
