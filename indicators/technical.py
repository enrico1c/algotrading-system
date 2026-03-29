"""
Indicator engine — pure pandas/numpy implementations.
No external TA library required (avoids version conflicts).
All functions are look-ahead-bias free (use only past data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI using Exponential Moving Average of gains/losses.
    Returns RSI series (0–100), NaN for first `period` rows.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename(f"rsi_{period}")


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean().rename(f"sma_{period}")


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().rename(f"atr_{period}")


def compute_spread_zscore(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    asymptotic_mean: float,
    asymptotic_std: float,
) -> pd.Series:
    """
    Compute z-score of the VECM spread relative to its asymptotic mean.
    spread = price_a - hedge_ratio * price_b
    z = (spread - M) / sqrt(Gamma0)

    Source: Cointegration-Pairs-VECM vault note
    """
    spread = price_a - hedge_ratio * price_b
    z = (spread - asymptotic_mean) / (asymptotic_std + 1e-12)
    return z.rename("spread_zscore")


def rsi_declining(rsi: pd.Series, days: int = 3) -> pd.Series:
    """
    Returns True where RSI has been declining for `days` consecutive days.
    Used by Triple-RSI strategy.
    """
    declining = pd.Series(True, index=rsi.index)
    for lag in range(1, days + 1):
        declining = declining & (rsi.shift(lag - 1) < rsi.shift(lag))
    return declining
