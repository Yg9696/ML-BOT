from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BotConfig


@dataclass(frozen=True)
class SignalCandidate:
    symbol: str
    event_candle_open_time: pd.Timestamp
    signal_close_time: pd.Timestamp
    entry_due_time: pd.Timestamp
    exit_due_time: pd.Timestamp
    event_return: float
    event_return_pct: float
    realized_vol_regime_value: float
    realized_vol_threshold: float
    breadth_count: int
    btc_24h_return: float
    ema50_slope: float
    ema200_slope: float
    close_price: float

    @property
    def abs_drop(self) -> float:
        return abs(self.event_return)


def add_features(frames: dict[str, pd.DataFrame], config: BotConfig) -> dict[str, pd.DataFrame]:
    featured: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        out = frame.copy()
        returns = out["close"].pct_change()
        prior_returns = returns.shift(1)
        out["event_return"] = returns
        out["ema50"] = out["close"].ewm(span=config.ema50_period, adjust=False).mean()
        out["ema200"] = out["close"].ewm(span=config.ema200_period, adjust=False).mean()
        out["ema50_slope"] = out["ema50"] / out["ema50"].shift(config.ema_slope_bars) - 1.0
        out["ema200_slope"] = out["ema200"] / out["ema200"].shift(config.ema_slope_bars) - 1.0
        out["realized_vol"] = prior_returns.rolling(
            config.realized_vol_lookback_bars,
            min_periods=config.realized_vol_min_periods,
        ).std()
        out["realized_vol_threshold"] = out["realized_vol"].rolling(
            config.vol_threshold_lookback_bars,
            min_periods=config.vol_threshold_min_periods,
        ).quantile(config.vol_threshold_quantile).shift(1)
        out["high_vol_regime"] = out["realized_vol"] >= out["realized_vol_threshold"]
        featured[symbol] = out

    btc = featured["BTCUSDT"][["timestamp", "close"]].copy()
    btc["btc_24h_return"] = btc["close"] / btc["close"].shift(config.btc_24h_bars) - 1.0
    breadth = pd.concat(
        [
            frame[["timestamp", "event_return"]].assign(is_down_2=lambda x: x["event_return"] <= config.breadth_return_threshold)
            for frame in featured.values()
        ],
        ignore_index=True,
    ).groupby("timestamp")["is_down_2"].sum().rename("breadth_count").reset_index()

    btc_context = btc[["timestamp", "btc_24h_return"]]
    return {
        symbol: frame.merge(btc_context, on="timestamp", how="left").merge(breadth, on="timestamp", how="left")
        for symbol, frame in featured.items()
    }


def generate_candidates(frames: dict[str, pd.DataFrame], aligned_timestamp: pd.Timestamp, config: BotConfig) -> list[SignalCandidate]:
    featured = add_features(frames, config)
    candle_delta = pd.Timedelta(minutes=config.candle_minutes)
    candidates: list[SignalCandidate] = []
    for symbol, frame in featured.items():
        row = frame[frame["timestamp"] == aligned_timestamp]
        if row.empty:
            continue
        item = row.iloc[0]
        if not bool(item.get("high_vol_regime", False)):
            continue
        if float(item["event_return"]) > config.event_return_threshold:
            continue
        if int(item.get("breadth_count", 0)) < config.min_breadth_count:
            continue
        if float(item.get("btc_24h_return", 0.0)) >= 0:
            continue
        if float(item.get("ema50_slope", 0.0)) >= 0:
            continue
        if float(item.get("ema200_slope", 0.0)) >= 0:
            continue
        signal_close_time = aligned_timestamp + candle_delta
        entry_due_time = signal_close_time + candle_delta
        candidates.append(
            SignalCandidate(
                symbol=symbol,
                event_candle_open_time=aligned_timestamp,
                signal_close_time=signal_close_time,
                entry_due_time=entry_due_time,
                exit_due_time=entry_due_time + candle_delta * config.hold_bars,
                event_return=float(item["event_return"]),
                event_return_pct=float(item["event_return"]) * 100.0,
                realized_vol_regime_value=float(item["realized_vol"]),
                realized_vol_threshold=float(item["realized_vol_threshold"]),
                breadth_count=int(item["breadth_count"]),
                btc_24h_return=float(item["btc_24h_return"]),
                ema50_slope=float(item["ema50_slope"]),
                ema200_slope=float(item["ema200_slope"]),
                close_price=float(item["close"]),
            )
        )
    return candidates
