# OI Flush Reversal Data Feasibility Report

## Objective

Check whether Binance USD-M Futures Open Interest history is available in enough depth to test the proposed Liquidation / Open-Interest Flush Reversal scenario.

This is a data feasibility report only. No strategy, backtest, ML model, live bot, or execution logic was built.

## Scenario Candidate

Liquidation / Open-Interest Flush Reversal.

The research question is whether forced-positioning events, visible through Open Interest contraction or liquidation-like positioning stress, can improve profitability and capital efficiency versus the current OHLCV-only Panic-Reversal module.

## Requested Data

- Exchange: Binance USD-M Futures
- Endpoint tested: `/futures/data/openInterestHist`
- Timeframe: `15m`
- Symbols:
  - BTCUSDT
  - ETHUSDT
  - SOLUSDT
  - BNBUSDT
  - XRPUSDT
  - ADAUSDT
  - AVAXUSDT
  - LINKUSDT
  - LTCUSDT
- Desired backtest range: `2020-01-01` to `2026-05-04`
- Minimum useful range: 12 months
- Minimum acceptable first feasibility range: 6 months

## Existing Local OI Data

No local Open Interest history was found in the repository.

Checks performed:

- Searched `data/` recursively for filenames and paths matching `open`, `interest`, or `oi`.
- Searched repo text for `openInterestHist`, `open interest`, and `oi`.

Existing local data relevant to this task:

- OHLCV exists under `data/binance_um_ohlcv/`.
- Funding exists under `data/binance_um_funding/`.
- No Open Interest history folder or CSVs were found.

## Binance Fetch Test

### Recent Fetch

Recent `15m` Open Interest data was successfully fetched for all 9 symbols.

Observed Binance server time during the probe:

```text
2026-05-07T10:46:59.656000+00:00
```

Backward pagination was tested using `endTime` and `limit=500`.

Result:

- Each 500-row page covers about 5.2 days at 15m resolution.
- Pagination worked back to approximately `2026-04-07T11:00:00Z`.
- The next backward page returned an empty response.
- Effective maximum retrievable span was about 30 days.

### Old Historical Fetch Probes

Explicit `startTime` / `endTime` probes were tested for:

- `2020-01-01` to `2020-01-08`
- `2025-05-04` to `2025-05-10`
- `2026-04-01` to `2026-04-07`

For all 9 symbols, these older-window probes returned HTTP `400` with:

```text
{"msg":"parameter 'startTime' is invalid.","code":-1130}
```

This confirms that backfilling beyond the recent Binance Open Interest history window is blocked from this endpoint in practice.

## Coverage Summary

Coverage file:

```text
reports/oi_flush_reversal/oi_data_coverage.csv
```

All symbols showed the same coverage:

| Symbol | First OI Timestamp | Last OI Timestamp | Rows | Missing 15m Intervals | Span Days | 6 Month Test | 12 Month Test | 2020-2026 Test |
|---|---|---|---:|---:|---:|---|---|---|
| BTCUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| ETHUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| SOLUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| BNBUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| XRPUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| ADAUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| AVAXUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| LINKUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |
| LTCUSDT | 2026-04-07T11:00:00Z | 2026-05-07T10:45:00Z | 2880 | 0 | 29.99 | No | No | No |

Within the retrievable recent 30-day window, no missing 15m intervals were detected.

Additional probe/sample files:

```text
reports/oi_flush_reversal/oi_fetch_probe_results.csv
reports/oi_flush_reversal/oi_fetch_sample.csv
```

The sample file contains only the earliest and latest 5 rows per symbol from the retrievable window.

## Feasibility Classification

Classification: **C. NOT FEASIBLE**

Reason:

Binance `openInterestHist` provides only about 30 days of retrievable `15m` Open Interest data from the tested endpoint. This is below the minimum acceptable first feasibility range of 6 months and far below the desired `2020-01-01` to `2026-05-04` range.

The endpoint is usable for forward collection and very recent diagnostics, but not for a reliable historical backtest of the OI Flush Reversal scenario.

## Recommendation

Do not build the OI Flush Reversal strategy backtest from Binance `openInterestHist` alone. The available history is not enough to validate robustness, year coverage, month concentration, or stress behavior.

Recommended next actions:

1. Start forward OI collection now.
   - Store `15m` Open Interest snapshots for the 9-symbol universe.
   - Use this for future paper research once at least 6-12 months accumulates.

2. Look for an external historical OI/liquidation data source.
   - The OI Flush scenario remains conceptually high value, but it needs a deeper source than Binance's public recent-history endpoint.

3. If external OI/liquidation data is unavailable, do not force the OI strategy.
   - Test Cross-Asset Dislocation Reversion next because it can be evaluated immediately with existing synchronized OHLCV data.

4. As an interim proxy only, scan OHLCV panic events with richer cross-asset breadth/dispersion features.
   - Treat this as a proxy research path, not a true OI/liquidation flush test.

## Decision

The OI / Liquidation Flush Reversal scenario is not rejected as a market idea. It is rejected as a historical backtest candidate using only currently accessible Binance `openInterestHist` data.

The best immediate research path is either:

- obtain external historical OI/liquidation data, or
- move to Cross-Asset Dislocation Reversion while forward-collecting OI for later validation.
