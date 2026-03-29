"""
Performance metrics for backtesting.
Source: Finance/Algo-Trading-Guide/03-Backtesting.md (vault targets)

Metrics computed:
  - Sharpe Ratio        (target > 1.0)
  - Sortino Ratio       (target > 1.5)
  - CAGR                (beats S&P 500)
  - Max Drawdown        (target < 25%)
  - Win Rate            (target > 55%)
  - Profit Factor       (target > 1.5)
  - Calmar Ratio        (CAGR / Max Drawdown)
  - Total trades
  - Avg return per trade
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute full performance metrics from equity curve and trade log.

    Args:
        equity: Daily portfolio value series
        trades: DataFrame with columns: pnl, return_pct, win
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (decimal)
        periods_per_year: Trading days per year (252 for daily)
    """
    metrics: Dict[str, float] = {}

    if equity.empty or len(equity) < 2:
        return _empty_metrics()

    # ── RETURNS ──────────────────────────────
    daily_returns = equity.pct_change().dropna()

    # ── CAGR ─────────────────────────────────
    n_days = len(equity)
    n_years = n_days / periods_per_year
    final_value = float(equity.iloc[-1])
    if initial_capital > 0 and n_years > 0:
        cagr = (final_value / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0
    metrics["cagr"] = cagr
    metrics["total_return"] = (final_value / initial_capital - 1) if initial_capital > 0 else 0.0

    # ── SHARPE ───────────────────────────────
    rf_daily = risk_free_rate / periods_per_year
    excess = daily_returns - rf_daily
    if daily_returns.std() > 0:
        sharpe = (excess.mean() / daily_returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    metrics["sharpe"] = float(sharpe)

    # ── SORTINO ──────────────────────────────
    downside = daily_returns[daily_returns < rf_daily]
    downside_std = downside.std() if len(downside) > 1 else 1e-8
    sortino = ((daily_returns.mean() - rf_daily) / downside_std) * np.sqrt(periods_per_year)
    metrics["sortino"] = float(sortino)

    # ── MAX DRAWDOWN ─────────────────────────
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())
    metrics["max_drawdown"] = max_dd

    # ── CALMAR ───────────────────────────────
    metrics["calmar"] = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # ── VOLATILITY ───────────────────────────
    metrics["annual_volatility"] = float(daily_returns.std() * np.sqrt(periods_per_year))

    # ── TRADE-LEVEL METRICS ──────────────────
    if trades.empty or len(trades) == 0:
        metrics.update({
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_return_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
        })
    else:
        wins = trades[trades["win"] == True] if "win" in trades.columns else pd.DataFrame()
        losses = trades[trades["win"] == False] if "win" in trades.columns else pd.DataFrame()

        metrics["n_trades"] = len(trades)
        metrics["win_rate"] = len(wins) / len(trades) if len(trades) > 0 else 0.0

        if "pnl" in trades.columns:
            gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
            gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
            metrics["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        else:
            metrics["profit_factor"] = 0.0

        if "return_pct" in trades.columns:
            metrics["avg_return_pct"] = float(trades["return_pct"].mean())
            metrics["avg_win_pct"] = float(wins["return_pct"].mean()) if len(wins) > 0 else 0.0
            metrics["avg_loss_pct"] = float(losses["return_pct"].mean()) if len(losses) > 0 else 0.0
        else:
            metrics["avg_return_pct"] = 0.0
            metrics["avg_win_pct"] = 0.0
            metrics["avg_loss_pct"] = 0.0

    # ── TIME IN MARKET ───────────────────────
    metrics["time_in_market"] = float((equity.diff() != 0).mean())

    return metrics


def compute_rolling_sharpe(
    equity: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling Sharpe ratio — used by portfolio rebalancer."""
    rf_daily = risk_free_rate / periods_per_year
    returns = equity.pct_change()
    excess = returns - rf_daily
    roll_sharpe = (excess.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(periods_per_year)
    return roll_sharpe.rename("rolling_sharpe")


def _empty_metrics() -> Dict[str, float]:
    return {
        "cagr": 0.0, "total_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_drawdown": 0.0, "calmar": 0.0, "annual_volatility": 0.0,
        "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "avg_return_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
        "time_in_market": 0.0,
    }
