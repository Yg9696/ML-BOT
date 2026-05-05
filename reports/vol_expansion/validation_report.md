# Executable Volatility Expansion Bot - Crude OHLCV Validation

## Scope

- No ML, open interest, funding, complex filters, limit fills, or intrabar reconstruction.
- Signals are known after candle close only.
- Entry modes tested: `next_open` and `next_close`.
- Same-candle exit ambiguity after entry is conservative: stop before target.

## Fixed Parameters

```json
{
  "atr_lookback": 48,
  "base_fee_bps_per_side": 4.0,
  "base_slippage_bps_per_side": 2.0,
  "breakout_lookback": 48,
  "cooldown_bars": 4,
  "max_extension_atr_mult": 1.8,
  "max_hold_bars": 16,
  "max_range_atr_mult": 2.8,
  "max_same_direction_bars": 3,
  "min_body_fraction": 0.55,
  "range_expansion_mult": 1.6,
  "range_median_lookback": 96,
  "stop_atr_mult": 1.2,
  "stress_fee_bps_per_side": 6.0,
  "stress_slippage_bps_per_side": 5.0,
  "target_r": 2.0,
  "timeframe": "15m",
  "volume_expansion_mult": 1.4,
  "volume_median_lookback": 96
}
```

## Summary

| scenario | entry | fee bps/side | slip bps/side | trades | trades/year | EV/trade R | total EV R | max DD R | losing streak | top symbol EV share | top month EV share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | next_open | 4.0 | 2.0 | 4 | 9.0 | 1.010 | 4.04 | -1.19 | 1 | 43.3% | 46.0% |
| base | next_close | 4.0 | 2.0 | 4 | 9.0 | -0.346 | -1.39 | -2.00 | 2 | 0.0% | 0.0% |
| stress | next_open | 6.0 | 5.0 | 4 | 9.0 | 0.810 | 3.24 | -1.34 | 1 | 47.6% | 53.8% |
| stress | next_close | 6.0 | 5.0 | 4 | 9.0 | -0.546 | -2.18 | -2.47 | 2 | 0.0% | 0.0% |

## Pass/Fail Checks

- base_next_open_ev_ge_0_3r: PASS
- base_next_close_ev_ge_0_3r: FAIL
- base_next_open_trades_year_ge_100: FAIL
- base_next_close_trades_year_ge_100: FAIL
- stress_next_open_positive: PASS
- next_close_not_dramatically_worse: FAIL
- top_symbol_share_ok: FAIL
- top_month_share_ok: FAIL

Overall crude-stage verdict: **FAIL**

See CSV outputs for trade-level R results, symbol distribution, monthly distribution, and equity drawdown path.