# Panic-Reversal Bot Dry-Run Skeleton

This folder contains a dry-run-only skeleton for the selected Panic-Reversal Bot configuration.

The source-of-truth architecture is:

- `reports/panic_reversal_live_feasibility/live_feasibility.md`
- `reports/panic_reversal_live_feasibility/live_architecture.json`

## Safety Defaults

The skeleton refuses to run unless:

- `BOT_MODE=dry_run`
- `LIVE_TRADING_ENABLED=false`

The execution client has no live Binance order adapter. `place_order()` always raises.

## Run One Dry-Run Cycle

```bash
BOT_MODE=dry_run LIVE_TRADING_ENABLED=false python -m bot.panic_reversal.main --once
```

By default, the bot reads local candles from:

```text
data/binance_um_ohlcv/15m
```

To fetch the latest Binance USD-M klines instead of local files:

```bash
BOT_MODE=dry_run LIVE_TRADING_ENABLED=false PANIC_BOT_DATA_SOURCE=binance python -m bot.panic_reversal.main --once
```

## State Outputs

Dry-run state is written under:

```text
bot/panic_reversal/state/
bot/panic_reversal/logs/
```
These files record processed signals, accepted signals, skipped signals, simulated positions, closed positions, scheduler runs, and latest data-quality checks.
