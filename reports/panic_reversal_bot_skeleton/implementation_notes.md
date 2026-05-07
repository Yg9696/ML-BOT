# Panic-Reversal Bot Skeleton Implementation Notes

## Scope

This deliverable builds the initial dry-run and paper-mode skeleton only. It does not place real orders, does not connect a live order adapter, does not add ML, and does not change the frozen Panic-Reversal strategy.

## How To Run Dry-Run

PowerShell:

```powershell
$env:BOT_MODE = "dry_run"
$env:LIVE_TRADING_ENABLED = "false"
python -m bot.panic_reversal.main --once
```

Optional live candle fetch without order execution:

```powershell
$env:BOT_MODE = "dry_run"
$env:LIVE_TRADING_ENABLED = "false"
$env:PANIC_BOT_DATA_SOURCE = "binance"
python -m bot.panic_reversal.main --once
```

Default local candle path:

```text
data/binance_um_ohlcv/15m
```

## Environment Variables

- `BOT_MODE`: must be `dry_run`.
- `LIVE_TRADING_ENABLED`: must be `false`.
- `PANIC_BOT_DATA_SOURCE`: optional, `local` or `binance`; default is `local`.
- `PANIC_BOT_DATA_DIR`: optional local 15m OHLCV folder override.
- `PANIC_BOT_STATE_DIR`: optional state output folder override.
- `PANIC_BOT_LOG_DIR`: optional log output folder override.

No Binance API keys are required because no private or order endpoints are used.

## Implemented

- Config module with frozen selected strategy parameters.
- Local and public Binance kline data fetcher.
- 15m candle alignment across the 9-symbol universe.
- Signal engine for:
  - return <= -3%
  - prior-only high realized-volatility regime
  - simultaneous 2% down breadth >= 4
  - BTC 24h return < 0
  - EMA50 slope < 0
  - EMA200 slope < 0
- Risk engine for:
  - concurrent cap 10
  - symbol cap 2
  - largest-drop deterministic selection
  - 2% allocation per accepted position
  - 20% exposure cap
- Position manager with stacking state, pending entries, simulated opens, and scheduled exits after 12 completed 15m candles.
- Durable duplicate prevention by `config_version|symbol|event_time`.
- Stub execution client that refuses real orders.
- Local JSON/JSONL state and scheduler run logs.

## Not Implemented Yet

- Continuous daemon scheduler.
- Private Binance account diagnostics.
- Exchange reconciliation.
- Live order adapter.
- Partial-fill handling beyond the data model/stub boundary.
- Spread/slippage monitoring.
- Web dashboard or alerting.
- Production deployment config.

## Safety Guards

- The bot refuses to run unless `BOT_MODE=dry_run`.
- The bot refuses to run if `LIVE_TRADING_ENABLED=true`.
- `ExecutionClientStub.place_order()` always raises.
- Simulated orders are local records only.
- No API keys are needed or read.
- The default data source is local files, not live execution infrastructure.
