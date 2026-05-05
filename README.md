# Executable Volatility Expansion Validation

Standalone OHLCV validation harness for the crude, execution-first version of:

`Executable Volatility Expansion Bot with Crowd-Avoidance Filters`

This repo intentionally does not contain ML, open interest, funding, limit-fill logic, or intrabar reconstruction.

## Run

```powershell
python scripts/run_vol_expansion_backtest.py
```

Default input points at the previous project's 12-month basket:

`..\ML-TRADE\training\fetchingData\generated_windows\basket_12_months`

Outputs are written to `reports/vol_expansion/`.

## Execution Rules

- Signals are computed only after candle close.
- Entry is tested at `next_open` and `next_close`.
- No entry is allowed inside the signal candle.
- Future exit ambiguity is conservative: if stop and target are both touched in one candle, stop wins.
- Fees and slippage are included as an R debit.
