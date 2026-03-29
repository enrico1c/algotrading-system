"""
Backtest engine — runs all registered strategies and aggregates results.
Includes walk-forward validation and robustness checks.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config.settings import EXECUTION_CONFIG, Settings
from data.fetcher import DataFetcher
from strategies.base import Strategy, StrategyResult
from strategies.registry import registry
from utils.logger import get_logger

logger = get_logger("backtest.engine")


class BacktestEngine:
    """
    Orchestrates backtests for all active strategies.
    Supports:
      - Individual strategy backtests
      - Portfolio-level combined backtest
      - Walk-forward validation
    """

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.fetcher = DataFetcher()
        self._results: Dict[str, StrategyResult] = {}

    def run_all(
        self,
        start: str = None,
        end: Optional[str] = None,
    ) -> Dict[str, StrategyResult]:
        """
        Run backtests for all registered strategies.
        Returns dict of strategy_name → StrategyResult.
        """
        registry.auto_discover()
        start = start or self.settings.data.backtest_start
        strategies = registry.instantiate_all({
            "rsi2_mean_reversion": self.settings.rsi2,
            "triple_rsi": self.settings.triple_rsi,
            "cointegration_pairs_vecm": self.settings.pairs_vecm,
        })

        per_strategy_capital = (
            self.settings.portfolio.initial_capital
            * self.settings.portfolio.equal_weight_for_n(len(strategies))
        )

        for strat in strategies:
            logger.info(f"Running backtest: {strat.name} (capital={per_strategy_capital:.2f} EUR)")
            result = self._run_single(strat, per_strategy_capital, start, end)
            self._results[strat.name] = result

        return self._results

    def run_single(
        self,
        strategy_name: str,
        capital: Optional[float] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> StrategyResult:
        """Run backtest for one strategy by name."""
        registry.auto_discover()
        strat = registry.instantiate(strategy_name)
        if capital is None:
            n = max(len(registry.names()), 1)
            capital = self.settings.portfolio.initial_capital / n
        result = self._run_single(strat, capital, start or self.settings.data.backtest_start, end)
        self._results[strategy_name] = result
        return result

    def run_walkforward(
        self,
        strategy_name: str,
        n_splits: int = 5,
        capital: Optional[float] = None,
    ) -> List[StrategyResult]:
        """
        Walk-forward validation: split history into N folds,
        train on first 70%, test on last 30% of each fold.
        """
        registry.auto_discover()
        strat = registry.instantiate(strategy_name)
        data = self._load_data(strat)
        if data.empty:
            return []

        capital = capital or self.settings.portfolio.initial_capital / 3
        results = []
        fold_size = len(data) // n_splits

        for i in range(n_splits):
            fold_data = data.iloc[i * fold_size: (i + 1) * fold_size]
            train_end = int(len(fold_data) * 0.70)
            test_data = fold_data.iloc[train_end:]
            if len(test_data) < strat.warmup_period():
                continue
            result = strat.backtest(test_data, initial_capital=capital,
                                    commission=EXECUTION_CONFIG.commission_pct,
                                    slippage=EXECUTION_CONFIG.slippage_pct)
            result.metrics["fold"] = i
            results.append(result)
            logger.info(
                f"Walk-forward fold {i+1}/{n_splits} | {strategy_name} | "
                f"sharpe={result.metrics.get('sharpe', 0):.2f} | "
                f"win_rate={result.metrics.get('win_rate', 0):.1%}"
            )

        return results

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame comparing all strategy results."""
        if not self._results:
            return pd.DataFrame()
        rows = []
        for name, result in self._results.items():
            m = result.metrics
            rows.append({
                "Strategy": name,
                "CAGR": f"{m.get('cagr', 0):.1%}",
                "Sharpe": f"{m.get('sharpe', 0):.2f}",
                "Max DD": f"{m.get('max_drawdown', 0):.1%}",
                "Win Rate": f"{m.get('win_rate', 0):.1%}",
                "Trades": int(m.get("n_trades", 0)),
                "Profit Factor": f"{m.get('profit_factor', 0):.2f}",
                "Calmar": f"{m.get('calmar', 0):.2f}",
            })
        return pd.DataFrame(rows).set_index("Strategy")

    # ──────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────

    def _run_single(
        self,
        strat: Strategy,
        capital: float,
        start: str,
        end: Optional[str],
    ) -> StrategyResult:
        data = self._load_data(strat, start, end)
        if data.empty:
            logger.warning(f"No data for {strat.name} — skipping")
            from backtest.metrics import _empty_metrics
            from strategies.base import StrategyResult
            return StrategyResult(
                strategy_name=strat.name,
                equity_curve=pd.Series([capital]),
                trades=pd.DataFrame(),
                signals=[],
                metrics=_empty_metrics(),
            )
        return strat.backtest(
            data,
            initial_capital=capital,
            commission=EXECUTION_CONFIG.commission_pct,
            slippage=EXECUTION_CONFIG.slippage_pct,
        )

    def _load_data(
        self,
        strat: Strategy,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load appropriate data for the strategy type."""
        start = start or self.settings.data.backtest_start
        tickers = strat.get_required_tickers()

        # Pairs strategy needs multi-ticker close DataFrame
        if strat.name == "cointegration_pairs_vecm":
            return self.fetcher.get_multi_close(tickers, start=start, end=end)

        # Single-ticker strategies
        primary = tickers[0] if tickers else "SPY"
        return self.fetcher.get_ohlcv(primary, start=start, end=end)
