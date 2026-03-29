"""
Data fetching layer — free sources only.
Primary: yfinance (no API key)
Secondary: pandas-datareader / FRED (no key for most series)
Caching: local parquet files to avoid repeated downloads.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import DATA_CONFIG, DATA_CACHE_DIR
from utils.logger import get_logger

logger = get_logger("data.fetcher")


class DataFetcher:
    """
    Unified data access with transparent local caching.

    Usage:
        fetcher = DataFetcher()
        df = fetcher.get_ohlcv("SPY", start="2010-01-01")
        prices = fetcher.get_multi_close(["AAPL", "MSFT", "SPY"])
    """

    def __init__(self, cache_dir: Path = DATA_CACHE_DIR,
                 cache_expiry_hours: int = DATA_CONFIG.cache_expiry_hours):
        self.cache_dir = Path(cache_dir)
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def get_ohlcv(
        self,
        ticker: str,
        start: str = DATA_CONFIG.backtest_start,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a single ticker.
        Returns cleaned DataFrame with columns: open, high, low, close, volume
        """
        cache_key = self._cache_key(ticker, start, end, interval)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Downloading {ticker} [{interval}] {start} → {end or 'today'}")
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            df = self._normalize_yf(raw, ticker)
            df = clean_ohlcv(df)
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def get_multi_close(
        self,
        tickers: List[str],
        start: str = DATA_CONFIG.backtest_start,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for multiple tickers.
        Returns DataFrame with tickers as columns, dates as index.
        """
        cache_key = self._cache_key("multi_close", "_".join(sorted(tickers)), start, end)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Downloading {len(tickers)} tickers for close prices")
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            if raw.empty:
                return pd.DataFrame()

            # Extract close for each ticker
            if len(tickers) == 1:
                closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                closes = pd.DataFrame()
                for t in tickers:
                    try:
                        col = raw[t]["Close"] if t in raw.columns.get_level_values(0) else raw["Close"]
                        closes[t] = col
                    except Exception:
                        pass

            closes.index = pd.to_datetime(closes.index)
            closes = closes.ffill().dropna(how="all")
            self._save_cache(cache_key, closes)
            return closes
        except Exception as e:
            logger.error(f"Failed to fetch multi-close: {e}")
            return pd.DataFrame()

    def get_live_ohlcv(
        self,
        ticker: str,
        lookback_days: int = DATA_CONFIG.live_lookback_days,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV for live signal generation (bypasses cache)."""
        start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
        logger.info(f"Live fetch: {ticker} ({lookback_days}d lookback)")
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if raw.empty:
                return pd.DataFrame()
            df = self._normalize_yf(raw, ticker)
            return clean_ohlcv(df)
        except Exception as e:
            logger.error(f"Live fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_risk_free_rate(self) -> float:
        """
        Fetch current 3-month T-bill rate from FRED (free, no key).
        Returns annualized rate as a decimal (e.g., 0.045 for 4.5%).
        Falls back to 0.04 if unavailable.
        """
        try:
            import pandas_datareader as pdr
            df = pdr.get_data_fred("DTB3", start="2024-01-01")
            rate = df.iloc[-1, 0] / 100.0
            logger.info(f"Risk-free rate (3m T-bill): {rate:.4f}")
            return float(rate)
        except Exception:
            logger.warning("Could not fetch risk-free rate — using 4.0% default")
            return 0.04

    def get_pairs_universe(
        self,
        tickers: Optional[List[str]] = None,
        start: str = DATA_CONFIG.backtest_start,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch close prices for the entire pairs universe.
        Used by the Cointegration Pairs strategy for pair selection.
        """
        universe = tickers or DATA_CONFIG.pairs_universe
        return self.get_multi_close(universe, start=start, end=end)

    # ──────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────

    def _normalize_yf(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Flatten yfinance multi-level columns into standard OHLCV."""
        # yfinance auto_adjust=True returns Open/High/Low/Close/Volume
        rename = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        # Handle both single and multi-ticker frames
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else raw
        df = raw.rename(columns=rename)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]

    @staticmethod
    def _cache_key(*args) -> str:
        raw = "_".join(str(a) for a in args if a is not None)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self.cache_expiry:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# ──────────────────────────────────────────────
# Standalone cleaning function (also used by fetcher)
# ──────────────────────────────────────────────

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize OHLCV data.
    - Drop rows with NaN close
    - Remove zero-volume rows
    - Sort by date
    - Forward-fill minor gaps
    """
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Drop rows missing close price
    df = df.dropna(subset=["close"] if "close" in df.columns else [df.columns[0]])
    # Remove zero-volume anomaly days (if volume column present)
    if "volume" in df.columns:
        df = df[df["volume"] > 0]
    # Forward-fill remaining small gaps (weekends/holidays already excluded by yfinance)
    df = df.ffill()
    return df
