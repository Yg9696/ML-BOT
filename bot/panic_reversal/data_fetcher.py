from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import BotConfig


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CandleBatch:
    frames: dict[str, pd.DataFrame]
    aligned_timestamp: pd.Timestamp | None
    quality: list[dict[str, object]]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="mixed")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df[REQUIRED_COLUMNS].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.reset_index(drop=True)


def _fetch_binance_klines(symbol: str, config: BotConfig) -> pd.DataFrame:
    response = requests.get(
        f"{config.binance_base_url}{config.binance_klines_path}",
        params={"symbol": symbol, "interval": config.timeframe, "limit": config.binance_request_limit},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[REQUIRED_COLUMNS].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _quality_row(symbol: str, df: pd.DataFrame, aligned_timestamp: pd.Timestamp | None, config: BotConfig) -> dict[str, object]:
    gaps = df["timestamp"].diff().dropna()
    expected_gap = pd.Timedelta(minutes=config.candle_minutes)
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "first_timestamp": df["timestamp"].min().isoformat() if not df.empty else None,
        "last_timestamp": df["timestamp"].max().isoformat() if not df.empty else None,
        "aligned_timestamp": aligned_timestamp.isoformat() if aligned_timestamp is not None else None,
        "has_aligned_timestamp": bool(aligned_timestamp is not None and (df["timestamp"] == aligned_timestamp).any()),
        "duplicate_rows": int(df["timestamp"].duplicated().sum()) if not df.empty else 0,
        "large_gap_count": int((gaps > expected_gap).sum()) if not gaps.empty else 0,
        "zero_volume_rows": int((df["volume"] <= 0).sum()) if not df.empty else 0,
    }


def fetch_latest_completed_candles(config: BotConfig) -> CandleBatch:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        if config.data_source == "binance":
            frame = _fetch_binance_klines(symbol, config)
        elif config.data_source == "local":
            frame = _load_csv(config.data_dir / f"{symbol}.csv")
        else:
            raise ValueError(f"Unsupported data source: {config.data_source}")
        frames[symbol] = frame

    latest_by_symbol = {symbol: frame["timestamp"].max() for symbol, frame in frames.items() if not frame.empty}
    aligned_timestamp = min(latest_by_symbol.values()) if len(latest_by_symbol) == len(config.symbols) else None
    if aligned_timestamp is not None:
        frames = {symbol: frame[frame["timestamp"] <= aligned_timestamp].reset_index(drop=True) for symbol, frame in frames.items()}
    quality = [_quality_row(symbol, frame, aligned_timestamp, config) for symbol, frame in frames.items()]
    return CandleBatch(frames=frames, aligned_timestamp=aligned_timestamp, quality=quality)
