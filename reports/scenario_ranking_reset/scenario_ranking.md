# Scenario Ranking Reset

## Objective

The primary objective is to find profitable, executable, robust trading scenarios with strong EV and good capital efficiency. Bot-specific advantages are useful, but they are secondary to raw scenario quality.

This memo does not introduce a new backtest, strategy implementation, ML model, or live bot work. It ranks the next research directions by expected potential and feasibility.

## Scoring Method

Each scenario is scored from 1 to 5 across:

- Expected EV potential
- Expected annual return potential
- Expected trade frequency
- Execution reliability
- Low sensitivity to fees/slippage
- Drawdown/risk profile
- Data availability
- Low overfitting risk
- Capital efficiency
- Ability to test quickly with existing code/data

Higher score is better. The numeric ranking is not treated as a false precision machine; it is a forcing function for a research decision.

## Ranked Table

| Rank | Scenario | EV Potential | Annual Return | Frequency | Execution | Fee/Slip Robustness | Risk Profile | Data | Low Overfit Risk | Capital Efficiency | Quick Test | Likely Beats Current Panic-Reversal? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Liquidation / OI Flush Reversal | 5 | 5 | 3 | 4 | 3 | 3 | 4 | 3 | 5 | 5 | Yes |
| 2 | Panic Reversal with improved flow/OI filters | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 3 | 4 | 5 | Yes |
| 3 | Cross-Asset Dislocation Reversion | 4 | 4 | 4 | 4 | 4 | 3 | 5 | 2 | 4 | 3 | Possibly |
| 4 | Funding/carry with risk filters | 3 | 3 | 4 | 5 | 4 | 2 | 5 | 3 | 3 | 5 | Unclear |
| 5 | Portfolio rotation / relative momentum | 3 | 3 | 4 | 5 | 4 | 3 | 5 | 2 | 3 | 4 | Unclear |
| 6 | Trend-following / momentum breakout on higher timeframe | 3 | 4 | 2 | 5 | 5 | 3 | 5 | 2 | 3 | 4 | Unclear |
| 7 | Volatility compression to expansion | 2 | 3 | 3 | 4 | 3 | 3 | 5 | 2 | 3 | 4 | No |

Note: Cross-Asset Dislocation scores high on total feasibility, but it ranks below the top two because the current strongest observed edge is panic/flush behavior. The next research should first test whether adding positioning/flow data strengthens that already validated source of edge.

## 1. Liquidation / OI Flush Reversal

**Core market behavior:** Forced deleveraging, liquidation cascades, and rapid OI contraction can push prices beyond short-term fair value. After the forced selling or forced buying is exhausted, price may mean-revert sharply.

**Why it could be profitable:** This scenario targets the mechanism that the current Panic-Reversal module is only approximating with OHLCV. If the real edge is forced-position unwind, liquidation/OI data should separate genuine flushes from ordinary sell candles.

**Why it could fail:** Public liquidation feeds can be incomplete, delayed, exchange-specific, or noisy. OI changes may lag or be hard to align. The move can continue after the flush, especially during true macro/news repricing.

**Execution cleanliness:** Good. The signal should be known after candle close or after a confirmed event window. Entry can remain market-only on the next candle close/open. No limit-fill assumptions are required.

**Data required:** OHLCV, Open Interest history, liquidation prints or aggregated liquidation estimates, symbol metadata, and optionally funding as context.

**First feasibility test:** For 15m systemic down events, bucket forward returns by OI drop, liquidation notional spike, and breadth. Compare large liquidation/OI-flush events against similar OHLCV-only panic events without OI confirmation.

**Pass/fail threshold:** At least 150 events over extended history, average net forward return materially above the current panic baseline after base/stress friction, stress positive, no single month above 40-50% contribution, and positive performance across most symbols.

**Likely to beat current Panic-Reversal?** Yes. It tests the likely underlying cause directly instead of using OHLCV as a proxy.

## 2. Panic Reversal with Improved Flow/OI Filters

**Core market behavior:** The validated panic-reversal edge appears strongest during systemic downside panic. Adding OI/flow filters should identify when the panic represents crowded liquidation rather than a normal trend leg.

**Why it could be profitable:** It starts from the only scenario that has already shown robust positive extended-history results. Better regime confirmation may reduce month concentration and improve capital efficiency without changing the basic execution model.

**Why it could fail:** Additional filters may reduce trade count too much. OI/flow data may overfit to known crisis periods. If the current edge is mostly broad-market rebound beta, added filters may not improve it.

**Execution cleanliness:** Good. The base signal, next_close entry, 12-candle horizon, stacking, caps, and friction model already exist. Flow/OI would be an added confirmation layer, not a new execution assumption.

**Data required:** Current OHLCV, Open Interest history, possibly liquidation data, and funding as optional context.

**First feasibility test:** Re-run extended Panic-Reversal using fixed, non-optimized OI/flow buckets: OI contraction, OI expansion into drop, liquidation spike, and funding context. Compare against the current selected module.

**Pass/fail threshold:** Maintain at least 150 trades over extended history, improve average return per trade or Sharpe-like ratio, reduce best-month contribution below 50%, remain positive under stress, and preserve positive symbol breadth.

**Likely to beat current Panic-Reversal?** Yes, if OI/flow separates real forced events from ordinary large red candles.

## 3. Cross-Asset Dislocation Reversion

**Core market behavior:** Highly correlated crypto futures sometimes dislocate temporarily. One asset may overshoot relative to BTC, ETH, or a sector basket, then revert as market makers and arbitrage flows normalize pricing.

**Why it could be profitable:** It can produce more frequent opportunities than rare panic events and may be capital-efficient because entries are based on relative moves, not broad directional market timing.

**Why it could fail:** Crypto correlations change by regime. A coin-specific news event can look like a dislocation but continue repricing. Shorting weaker legs or hedging may add borrow/funding/execution complexity.

**Execution cleanliness:** Moderate to good. A long-only reversion version is clean, but market-neutral or pair execution requires synchronized orders and more careful exposure accounting.

**Data required:** OHLCV for the current symbol universe. Funding/OI can be added later but is not required for the first scan.

**First feasibility test:** Build an event scan for extreme residual moves versus BTC/ETH/basket over 15m and 1h. Measure forward relative and absolute returns over 1, 3, 6, and 12 candles.

**Pass/fail threshold:** At least 300 events, positive forward return after simple market execution, stable across symbols, stress not negative, and no single event cluster dominating.

**Likely to beat current Panic-Reversal?** Possibly. It may improve frequency and capital usage, but it does not yet have the same empirical support as panic reversal.

## 4. Funding/Carry With Risk Filters

**Core market behavior:** Perpetual futures funding transfers value between longs and shorts. Persistent funding extremes may provide carry or reversal opportunities when filtered by volatility and trend risk.

**Why it could be profitable:** Funding data is available, slow-moving, and execution is clean. It may create steadier trade frequency than panic-only systems.

**Why it could fail:** The previous funding reaction test failed. Funding can remain extreme during strong trends, creating large adverse moves that overwhelm carry. Pure carry may need leverage and long holding periods, increasing liquidation and gap risk.

**Execution cleanliness:** Good. Entries can be at candle close or scheduled funding windows. No intrabar assumptions are needed.

**Data required:** OHLCV and Binance funding history, both already available in the project.

**First feasibility test:** Scan funding carry and reversal separately. Do not mix them. Measure forward returns after extreme funding only when volatility/trend filters are favorable.

**Pass/fail threshold:** Positive net returns after funding, fees, and stress friction; drawdown acceptable; enough events across years; no dependence on one symbol.

**Likely to beat current Panic-Reversal?** Unclear. It is easy to test, but prior evidence is weak.

## 5. Portfolio Rotation / Relative Momentum

**Core market behavior:** Crypto assets show rotating leadership. A portfolio that allocates to stronger assets and avoids weaker ones may capture persistent cross-sectional momentum.

**Why it could be profitable:** It can be robust, simple, and less dependent on rare event timing. It can also be tested on existing OHLCV data quickly.

**Why it could fail:** Rotation can be crowded, whipsaw-prone, and highly correlated during market stress. It may produce modest returns unless scaled or combined with market regime exposure.

**Execution cleanliness:** Very good. Rebalance at fixed times with market orders. No intrabar assumptions.

**Data required:** OHLCV only for the first test. Funding/OI can be added later for risk filters.

**First feasibility test:** Weekly or daily relative momentum rotation across the 9 symbols, with simple long-only top-N allocation, market execution, fees, slippage, and drawdown tracking.

**Pass/fail threshold:** Positive multi-year return, better drawdown-adjusted performance than BTC buy-and-hold or equal-weight basket, low turnover sensitivity, and no single asset dominance.

**Likely to beat current Panic-Reversal?** Unclear. It may be more robust but probably has lower EV per trade.

## 6. Trend-Following / Momentum Breakout On Higher Timeframe

**Core market behavior:** Crypto occasionally trends strongly after breakouts or regime shifts. Higher-timeframe trend following attempts to capture the middle of those moves.

**Why it could be profitable:** Execution is clean, friction sensitivity is low, and it can produce large winners if the market trends.

**Why it could fail:** Earlier OHLCV breakout work was weak. Trade frequency may be low, losing streaks can be long, and capital can sit idle or churn in ranges.

**Execution cleanliness:** Excellent. Signals are candle-close based and entries/exits can be market-only.

**Data required:** OHLCV only at 1h, 4h, and possibly daily.

**First feasibility test:** A conservative higher-timeframe trend-following benchmark with fixed rules, no aggressive parameter sweep, and comparison against buy-and-hold/equal-weight exposure.

**Pass/fail threshold:** Positive return after stress friction, acceptable drawdown, clear improvement over passive exposure, and enough trades to trust results.

**Likely to beat current Panic-Reversal?** Unclear to unlikely without a better regime definition.

## 7. Volatility Compression To Expansion

**Core market behavior:** Low-volatility compression can precede expansion. The system would attempt to enter when price breaks out of a compressed range.

**Why it could be profitable:** Volatility clustering is real, and compression/expansion logic is simple and executable.

**Why it could fail:** The previous volatility expansion/breakout scenario failed Phase 1-2. False breakouts are common, and OHLCV-only filters did not produce enough edge.

**Execution cleanliness:** Good. Candle-close breakout and market entry are straightforward.

**Data required:** OHLCV only.

**First feasibility test:** Only revisit if a materially different event definition is found in event scans. Do not continue optimizing the prior breakout logic.

**Pass/fail threshold:** EV/trade above 0.3R, sufficient trade count, stress positive, and no month/symbol concentration.

**Likely to beat current Panic-Reversal?** No. Existing evidence argues against prioritizing it.

## Top 2 Recommendations

### 1. Liquidation / OI Flush Reversal

This is the strongest next candidate because it attacks the suspected source of the current edge directly: forced positioning unwind. If liquidation/OI data confirms the same mean-reversion behavior with better separation, it can improve EV and capital efficiency rather than merely operationalizing the current imperfect proxy.

### 2. Panic Reversal With Improved Flow/OI Filters

This is the most efficient follow-up because it builds on validated research instead of starting cold. The current Panic-Reversal module is profitable but concentration-heavy. Flow/OI filters are the most plausible way to reduce false positives, improve robustness, and decide whether the module deserves continued implementation work.

## Why These Are Better Than Continuing Panic-Reversal Implementation Now

The current Panic-Reversal module is a valid candidate, but implementation work is premature if the core edge can be materially improved with positioning data. The main unresolved research risks are trade concentration, high concurrent exposure, and whether OHLCV-only panic candles are only a noisy proxy for liquidation/OI flushes.

Continuing bot implementation now would harden operational infrastructure around an incomplete signal. The better path is to spend the next research cycle testing whether OI/liquidation context produces a cleaner, more capital-efficient version of the same edge.

## Final Ranking

1. Liquidation / OI Flush Reversal
2. Panic Reversal with improved flow/OI filters
3. Cross-Asset Dislocation Reversion
4. Funding/carry with risk filters
5. Portfolio rotation / relative momentum
6. Trend-following / momentum breakout on higher timeframe
7. Volatility compression to expansion
