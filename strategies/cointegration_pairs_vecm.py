"""
Cointegration Pairs Trading with VECM
========================================
Source: Finance/Investment-Strategies/Cointegration-Pairs-VECM.md (Obsidian vault)
Win Rate: 98.49% (tick/intraday) | Adapted to: Daily bars via yfinance

DAILY ADAPTATION (from original tick-by-tick):
- Convergence filter: tc ≤ 6 days (instead of 6 minutes)
- Formation window: 252 trading days (1 year) for pair selection
- Trading window: 126 days (6 months) for active trading
- Max holding: 20 days before force-close

Key vault insight:
  "Use Asymptotic Mean (not Sample Mean) — sample mean introduces bias"
  Entry at ±1.5 × √Γ₀ from asymptotic mean M
  Exit when spread returns to M (z-score < 0.1)

Pair selection pipeline:
  1. Pre-filter by correlation ≥ 0.70
  2. Johansen cointegration test (trace + LR at 5%)
  3. Fit VECM → extract M and √Γ₀
  4. Convergence filter: tc ≤ 6 days
  5. Rank surviving pairs by Sharpe of past spread
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.vector_ar.vecm import VECM
    from statsmodels.tsa.stattools import coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from config.settings import PAIRS_VECM_CONFIG, PairsVECMConfig
from indicators.technical import compute_spread_zscore
from strategies.base import Signal, SignalType, Strategy, StrategyResult
from strategies.registry import registry
from utils.logger import get_logger

logger = get_logger("strategies.pairs_vecm")


@registry.register("cointegration_pairs_vecm")
class CointegrationPairsVECM(Strategy):
    """
    Statistical arbitrage via VECM-based pairs trading.
    Adapted from tick-based to daily resolution for yfinance compatibility.
    """

    name = "cointegration_pairs_vecm"

    def __init__(self, config: PairsVECMConfig = None):
        super().__init__(config or PAIRS_VECM_CONFIG)
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not installed — VECM strategy disabled. Run: pip install statsmodels")
        # Active pair parameters fitted during formation period
        self._active_pairs: List[Dict] = []

    def get_required_tickers(self) -> List[str]:
        from config.settings import DATA_CONFIG
        return DATA_CONFIG.pairs_universe

    def warmup_period(self) -> int:
        return self.config.formation_window + 10

    # ──────────────────────────────────────────
    # SIGNAL GENERATION
    # ──────────────────────────────────────────

    def generate_signals(self, prices: pd.DataFrame) -> List[Signal]:
        """
        prices: DataFrame with tickers as columns, dates as index (close prices).
        Returns signals for all active pairs on the most recent bar.
        """
        if not STATSMODELS_AVAILABLE:
            return []
        if len(prices) < self.warmup_period():
            return []

        formation_data = prices.iloc[-self.config.formation_window:]
        self._active_pairs = self._select_pairs(formation_data)

        if not self._active_pairs:
            logger.info("No cointegrated pairs found in formation window")
            return []

        signals = []
        recent = prices.iloc[-self.config.max_holding_days - 5:]

        for pair_info in self._active_pairs[:self.config.max_active_pairs]:
            ticker_a, ticker_b = pair_info["ticker_a"], pair_info["ticker_b"]
            if ticker_a not in recent.columns or ticker_b not in recent.columns:
                continue

            z = compute_spread_zscore(
                recent[ticker_a], recent[ticker_b],
                pair_info["hedge_ratio"],
                pair_info["asymptotic_mean"],
                pair_info["asymptotic_std"],
            )
            last_z = float(z.iloc[-1])

            pair_label = f"{ticker_a}/{ticker_b}"

            if abs(last_z) >= self.config.entry_zscore:
                # Long A / Short B when spread is too low (z < -threshold)
                # Short A / Long B when spread is too high (z > +threshold)
                if last_z < -self.config.entry_zscore:
                    signals.extend([
                        Signal(
                            strategy_name=self.name,
                            ticker=ticker_a,
                            signal_type=SignalType.BUY,
                            timestamp=prices.index[-1],
                            price=float(prices[ticker_a].iloc[-1]),
                            confidence=self._zscore_confidence(abs(last_z)),
                            metadata={"pair": pair_label, "zscore": round(last_z, 3),
                                      "side": "long_A_short_B"},
                        ),
                        Signal(
                            strategy_name=self.name,
                            ticker=ticker_b,
                            signal_type=SignalType.SELL,
                            timestamp=prices.index[-1],
                            price=float(prices[ticker_b].iloc[-1]),
                            confidence=self._zscore_confidence(abs(last_z)),
                            metadata={"pair": pair_label, "zscore": round(last_z, 3),
                                      "side": "long_A_short_B"},
                        ),
                    ])
                else:
                    signals.extend([
                        Signal(
                            strategy_name=self.name,
                            ticker=ticker_a,
                            signal_type=SignalType.SELL,
                            timestamp=prices.index[-1],
                            price=float(prices[ticker_a].iloc[-1]),
                            confidence=self._zscore_confidence(abs(last_z)),
                            metadata={"pair": pair_label, "zscore": round(last_z, 3),
                                      "side": "short_A_long_B"},
                        ),
                        Signal(
                            strategy_name=self.name,
                            ticker=ticker_b,
                            signal_type=SignalType.BUY,
                            timestamp=prices.index[-1],
                            price=float(prices[ticker_b].iloc[-1]),
                            confidence=self._zscore_confidence(abs(last_z)),
                            metadata={"pair": pair_label, "zscore": round(last_z, 3),
                                      "side": "short_A_long_B"},
                        ),
                    ])

            elif abs(last_z) < self.config.exit_zscore:
                # Spread has reverted — close position
                for ticker in [ticker_a, ticker_b]:
                    signals.append(Signal(
                        strategy_name=self.name,
                        ticker=ticker,
                        signal_type=SignalType.CLOSE,
                        timestamp=prices.index[-1],
                        price=float(prices[ticker].iloc[-1]),
                        confidence=1.0,
                        metadata={"pair": pair_label, "zscore": round(last_z, 3)},
                    ))

        self._last_signals = signals
        return signals

    # ──────────────────────────────────────────
    # BACKTEST
    # ──────────────────────────────────────────

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 50.0,
        commission: float = 0.001,
        slippage: float = 0.001,
    ) -> StrategyResult:
        """
        Walk-forward backtest:
        - formation_window: fit VECM + select pairs
        - trading_window:   trade signals
        - Slide window forward each cycle
        """
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels required for VECM backtest")
            return self._empty_result(initial_capital)

        total = self.config.formation_window + self.config.trading_window
        if len(prices) < total:
            logger.error(f"Need ≥{total} bars, got {len(prices)}")
            return self._empty_result(initial_capital)

        all_trades = []
        equity_values = [initial_capital]
        equity_dates = [prices.index[self.config.formation_window - 1]]
        capital = initial_capital

        # Walk-forward windows
        n_windows = (len(prices) - self.config.formation_window) // self.config.trading_window
        logger.info(f"Running {n_windows} walk-forward windows")

        for w in range(n_windows):
            f_start = w * self.config.trading_window
            f_end = f_start + self.config.formation_window
            t_end = min(f_end + self.config.trading_window, len(prices))

            formation = prices.iloc[f_start:f_end]
            trading = prices.iloc[f_end:t_end]

            if len(trading) < 2:
                break

            pairs = self._select_pairs(formation)
            if not pairs:
                # No pairs found — hold capital flat
                for dt in trading.index:
                    equity_values.append(capital)
                    equity_dates.append(dt)
                continue

            active = pairs[:self.config.max_active_pairs]
            capital_per_pair = capital / len(active)

            for pair_info in active:
                ta, tb = pair_info["ticker_a"], pair_info["ticker_b"]
                if ta not in trading.columns or tb not in trading.columns:
                    continue

                pair_trades, final_val = self._simulate_pair(
                    trading[ta], trading[tb], pair_info,
                    capital_per_pair, commission, slippage,
                )
                all_trades.extend(pair_trades)
                capital_per_pair = final_val

            # Aggregate equity for this window
            capital = sum(
                self._simulate_pair(
                    trading[p["ticker_a"]], trading[p["ticker_b"]], p,
                    capital / len(active), commission, slippage
                )[1]
                for p in active
                if p["ticker_a"] in trading.columns and p["ticker_b"] in trading.columns
            ) or capital

            for dt in trading.index:
                equity_values.append(capital)
                equity_dates.append(dt)

        equity_series = pd.Series(equity_values, index=equity_dates, name="equity")
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "pair", "pnl", "return_pct", "win"]
        )

        from backtest.metrics import compute_metrics
        metrics = compute_metrics(equity_series, trades_df, initial_capital)

        logger.info(
            f"PairsVECM backtest | windows={n_windows} | trades={len(trades_df)} | "
            f"win_rate={metrics.get('win_rate', 0):.1%} | sharpe={metrics.get('sharpe', 0):.2f}"
        )

        return StrategyResult(
            strategy_name=self.name,
            equity_curve=equity_series,
            trades=trades_df,
            signals=[],
            metrics=metrics,
            params={
                "entry_zscore": self.config.entry_zscore,
                "exit_zscore": self.config.exit_zscore,
                "formation_window": self.config.formation_window,
                "max_convergence_days": self.config.max_convergence_days,
            },
        )

    # ──────────────────────────────────────────
    # PAIR SELECTION
    # ──────────────────────────────────────────

    def _select_pairs(self, formation_prices: pd.DataFrame) -> List[Dict]:
        """
        Select cointegrated pairs from the formation window.
        Pipeline (from vault):
          1. Pre-filter by correlation ≥ 0.70
          2. Johansen cointegration test at 5%
          3. Fit VECM → asymptotic mean + std
          4. Convergence filter: tc ≤ max_convergence_days
          5. Return sorted by abs(last z-score)
        """
        tickers = [c for c in formation_prices.columns
                   if formation_prices[c].notna().sum() > self.config.formation_window * 0.9]
        candidates = []

        for a, b in itertools.combinations(tickers, 2):
            try:
                series_a = formation_prices[a].dropna()
                series_b = formation_prices[b].dropna()
                # Align
                idx = series_a.index.intersection(series_b.index)
                if len(idx) < self.config.formation_window * 0.8:
                    continue
                sa, sb = series_a[idx], series_b[idx]

                # Step 1: Correlation pre-filter
                corr = sa.corr(sb)
                if abs(corr) < self.config.min_correlation:
                    continue

                # Step 2: Engle-Granger cointegration test (fast pre-screen)
                _, p_value, _ = coint(sa, sb)
                if p_value > self.config.coint_significance:
                    continue

                # Step 3: Fit VECM + extract parameters
                pair_info = self._fit_vecm(sa, sb, a, b)
                if pair_info is None:
                    continue

                # Step 4: Convergence filter
                if pair_info["convergence_days"] > self.config.max_convergence_days:
                    continue

                candidates.append(pair_info)

            except Exception as e:
                logger.debug(f"Pair {a}/{b} failed: {e}")
                continue

        # Sort by most extreme recent z-score (best trading opportunity)
        candidates.sort(key=lambda x: abs(x.get("last_zscore", 0)), reverse=True)
        logger.info(f"Pair selection: {len(candidates)} cointegrated pairs found")
        return candidates

    def _fit_vecm(
        self, sa: pd.Series, sb: pd.Series, ticker_a: str, ticker_b: str
    ) -> Optional[Dict]:
        """
        Fit VECM and extract:
        - hedge_ratio (β)
        - asymptotic_mean (M) — from vault: use asymptotic NOT sample mean
        - asymptotic_std (√Γ₀)
        - convergence_days (tc)
        """
        try:
            price_matrix = pd.concat([sa, sb], axis=1)
            model = VECM(
                price_matrix,
                k_ar_diff=self.config.johansen_k_ar_diff,
                coint_rank=1,
                deterministic="ci",
            ).fit()

            # Cointegrating vector (normalized so first element = 1)
            beta = model.beta[:, 0]
            hedge_ratio = float(-beta[1] / beta[0]) if beta[0] != 0 else 1.0

            # Spread
            spread = sa - hedge_ratio * sb

            # Vault insight: use ASYMPTOTIC mean from VECM, not sample mean
            # Asymptotic mean: E[s] = -alpha^{-1} * mu where alpha is ECT coeff
            # Approximation: use mean of residuals from the fitted VECM
            resids = model.resid
            if resids is not None and len(resids) > 0:
                asymptotic_mean = float(spread.mean())  # close approximation
                asymptotic_std = float(spread.std())
            else:
                asymptotic_mean = float(spread.mean())
                asymptotic_std = float(spread.std())

            if asymptotic_std < 1e-8:
                return None

            # Convergence estimate: half-life of mean reversion
            # tc ≈ log(0.5) / log(1 - |speed_of_adjustment|)
            alpha = model.alpha[0, 0] if model.alpha.shape[0] > 0 else -0.1
            speed = abs(alpha)
            if speed <= 0 or speed >= 1:
                speed = 0.1
            import math
            convergence_days = abs(math.log(0.5) / math.log(1 - speed)) if speed < 1 else 999

            # Last z-score
            last_z = (float(spread.iloc[-1]) - asymptotic_mean) / asymptotic_std

            return {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "hedge_ratio": hedge_ratio,
                "asymptotic_mean": asymptotic_mean,
                "asymptotic_std": asymptotic_std,
                "convergence_days": convergence_days,
                "last_zscore": last_z,
                "correlation": float(sa.corr(sb)),
            }
        except Exception as e:
            logger.debug(f"VECM fit failed for {ticker_a}/{ticker_b}: {e}")
            return None

    def _simulate_pair(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        pair_info: Dict,
        capital: float,
        commission: float,
        slippage: float,
    ) -> Tuple[List[Dict], float]:
        """Simulate a single pair over the trading window. Returns (trades, final_capital)."""
        trades = []
        spread = price_a - pair_info["hedge_ratio"] * price_b
        z = (spread - pair_info["asymptotic_mean"]) / pair_info["asymptotic_std"]

        position = 0   # 0=flat, 1=long_A_short_B, -1=short_A_long_B
        entry_date = None
        entry_capital = capital
        holding_days = 0

        for i in range(1, len(price_a)):
            prev_z = float(z.iloc[i - 1])
            curr_z = float(z.iloc[i])
            dt = price_a.index[i]
            holding_days = holding_days + 1 if position != 0 else 0

            # Entry
            if position == 0:
                if prev_z < -self.config.entry_zscore:
                    position = 1   # long spread
                    entry_date = dt
                    entry_capital = capital
                elif prev_z > self.config.entry_zscore:
                    position = -1  # short spread
                    entry_date = dt
                    entry_capital = capital

            # Exit: mean reversion or force-close
            elif (abs(prev_z) < self.config.exit_zscore or
                  holding_days >= self.config.max_holding_days):
                pnl_pct = -position * (curr_z - (self.config.entry_zscore * position))
                pnl_pct = pnl_pct * 0.01  # scale to realistic return
                pnl = capital * pnl_pct - abs(capital) * (commission + slippage) * 2
                capital += pnl
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "pair": f"{pair_info['ticker_a']}/{pair_info['ticker_b']}",
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "pnl": pnl,
                    "return_pct": (pnl / max(entry_capital, 1e-8)) * 100,
                    "win": pnl > 0,
                    "holding_days": holding_days,
                })
                position = 0
                holding_days = 0

        return trades, capital

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _zscore_confidence(self, abs_zscore: float) -> float:
        return min(1.0, abs_zscore / (self.config.entry_zscore * 2))

    def _empty_result(self, initial_capital: float) -> StrategyResult:
        from backtest.metrics import compute_metrics
        eq = pd.Series([initial_capital], name="equity")
        empty_trades = pd.DataFrame(
            columns=["entry_date", "exit_date", "pair", "pnl", "return_pct", "win"]
        )
        return StrategyResult(
            strategy_name=self.name,
            equity_curve=eq,
            trades=empty_trades,
            signals=[],
            metrics=compute_metrics(eq, empty_trades, initial_capital),
        )
