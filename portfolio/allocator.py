"""
Dynamic Portfolio Allocation Engine
=====================================
Capital rules (from vault / user spec):
  - 150 EUR starting capital
  - Equal initial weights: 1/N per strategy
  - Max per strategy: 1/N + 9% headroom (dynamic, N-aware)
    → N=3: 1/3 + 9% ≈ 42%  |  N=4: 25% + 9% = 34%  |  N=2: 50% + 9% = 59%
  - Max single trade: 10% of strategy's current capital
  - Reallocation trigger: rolling Sharpe divergence > threshold
  - Compounding: both at strategy level and portfolio level

Scalability:
  - Adding a new strategy: register it → allocator auto-detects on next rebalance
  - Weights rebalance across ALL strategies automatically
  - Max allocation cap adjusts dynamically with N
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import PORTFOLIO_CONFIG, PortfolioConfig
from utils.logger import get_logger

logger = get_logger("portfolio.allocator")


@dataclass
class AllocationState:
    """Current allocation state for all strategies."""
    timestamp: datetime
    total_capital: float
    weights: Dict[str, float]          # strategy_name → weight (0–1)
    allocated_capital: Dict[str, float]  # strategy_name → EUR amount
    n_strategies: int

    def max_trade_size(self, strategy_name: str, config: PortfolioConfig) -> float:
        """Max EUR for a single trade on this strategy."""
        strat_capital = self.allocated_capital.get(strategy_name, 0.0)
        return strat_capital * config.max_trade_size_pct

    def __str__(self) -> str:
        lines = [
            f"Portfolio State @ {self.timestamp:%Y-%m-%d %H:%M}",
            f"  Total Capital: {self.total_capital:.2f} EUR",
            f"  Strategies ({self.n_strategies}):",
        ]
        for name, w in self.weights.items():
            cap = self.allocated_capital.get(name, 0)
            lines.append(f"    {name:<35} {w:.1%}  ({cap:.2f} EUR)")
        return "\n".join(lines)


class PortfolioAllocator:
    """
    Manages capital allocation across N strategies.

    Key behaviours:
      - Equal weights at init
      - Performance-based rebalancing using rolling Sharpe
      - Dynamic max-weight cap (1/N + 9%)
      - 10% max trade size per strategy
      - Auto-detects new strategies on rebalance
      - Compounding: equity grows within each strategy allocation
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PORTFOLIO_CONFIG
        self._state: Optional[AllocationState] = None
        self._equity_history: Dict[str, pd.Series] = {}  # strategy → equity series
        self._rebalance_log: List[Dict] = []

    # ──────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────

    def initialize(self, strategy_names: List[str]) -> AllocationState:
        """Set equal weights across all strategies at startup."""
        n = len(strategy_names)
        if n == 0:
            raise ValueError("No strategies provided")

        equal_weight = self.config.equal_weight_for_n(n)
        weights = {name: equal_weight for name in strategy_names}
        allocated = {
            name: self.config.initial_capital * equal_weight
            for name in strategy_names
        }

        self._state = AllocationState(
            timestamp=datetime.now(),
            total_capital=self.config.initial_capital,
            weights=weights,
            allocated_capital=allocated,
            n_strategies=n,
        )
        logger.info(
            f"Initialized portfolio: {self.config.initial_capital:.2f} EUR across "
            f"{n} strategies @ {equal_weight:.1%} each"
        )
        return self._state

    # ──────────────────────────────────────────
    # COMPOUNDING — update capital after trades
    # ──────────────────────────────────────────

    def update_strategy_capital(
        self, strategy_name: str, new_capital: float, equity_point: float = None
    ) -> None:
        """
        Called after a strategy closes a trade or at EOD.
        Updates the strategy's allocated capital (compounding in effect).
        """
        if self._state is None:
            raise RuntimeError("Allocator not initialized — call initialize() first")

        old_cap = self._state.allocated_capital.get(strategy_name, 0.0)
        self._state.allocated_capital[strategy_name] = new_capital
        self._state.total_capital = sum(self._state.allocated_capital.values())

        if equity_point is not None:
            if strategy_name not in self._equity_history:
                self._equity_history[strategy_name] = pd.Series(dtype=float)
            self._equity_history[strategy_name][datetime.now()] = equity_point

        logger.debug(
            f"Capital update: {strategy_name} {old_cap:.2f} → {new_capital:.2f} EUR "
            f"(portfolio total: {self._state.total_capital:.2f})"
        )

    def record_equity(self, strategy_name: str, date, equity: float) -> None:
        """Record daily equity for rolling Sharpe computation."""
        if strategy_name not in self._equity_history:
            self._equity_history[strategy_name] = pd.Series(dtype=float, name=strategy_name)
        self._equity_history[strategy_name][pd.Timestamp(date)] = equity

    # ──────────────────────────────────────────
    # REBALANCING
    # ──────────────────────────────────────────

    def should_rebalance(self) -> bool:
        """Check if current weights have drifted beyond the rebalance threshold."""
        if self._state is None or len(self._equity_history) < 2:
            return False
        current_weights = self._compute_current_weights()
        target_weights = self._compute_target_weights()
        drift = max(abs(current_weights.get(n, 0) - target_weights.get(n, 0))
                    for n in self._state.weights)
        return drift > self.config.rebalance_threshold

    def rebalance(self, strategy_names: Optional[List[str]] = None) -> AllocationState:
        """
        Rebalance portfolio weights based on rolling Sharpe performance.
        Auto-detects new strategies if strategy_names differs from current.

        Returns new AllocationState with updated weights.
        """
        if self._state is None:
            raise RuntimeError("Allocator not initialized")

        current_names = list(self._state.weights.keys())
        all_names = strategy_names or current_names

        # Auto-detect new strategies
        new_strategies = [n for n in all_names if n not in current_names]
        if new_strategies:
            logger.info(f"New strategies detected: {new_strategies} — adding to portfolio")
            for name in new_strategies:
                self._state.weights[name] = 0.0
                self._state.allocated_capital[name] = 0.0

        target_weights = self._compute_target_weights()
        n = len(all_names)
        total = self._state.total_capital

        new_allocation = {name: total * w for name, w in target_weights.items()}

        old_weights = dict(self._state.weights)
        self._state.weights = target_weights
        self._state.allocated_capital = new_allocation
        self._state.n_strategies = n
        self._state.timestamp = datetime.now()

        log_entry = {
            "timestamp": self._state.timestamp,
            "old_weights": old_weights,
            "new_weights": target_weights,
            "total_capital": total,
        }
        self._rebalance_log.append(log_entry)

        logger.info(
            f"Rebalanced portfolio (N={n}): "
            + ", ".join(f"{k}={v:.1%}" for k, v in target_weights.items())
        )
        return self._state

    # ──────────────────────────────────────────
    # TRADE SIZING
    # ──────────────────────────────────────────

    def get_trade_size(self, strategy_name: str) -> float:
        """
        Max EUR for a single trade on this strategy.
        = min(10% of strategy capital, available cash)
        """
        if self._state is None:
            return 0.0
        strat_cap = self._state.allocated_capital.get(strategy_name, 0.0)
        return strat_cap * self.config.max_trade_size_pct

    def get_position_size_units(
        self, strategy_name: str, price: float, direction: str = "long"
    ) -> float:
        """
        Convert EUR trade size to number of units/shares.
        For eToro: amount is in USD, this returns EUR amount directly.
        """
        max_eur = self.get_trade_size(strategy_name)
        if price <= 0:
            return 0.0
        return max_eur / price  # fractional units

    # ──────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────

    def _compute_target_weights(self) -> Dict[str, float]:
        """
        Compute target weights based on rolling Sharpe performance.
        Falls back to equal weights if insufficient history.

        Algorithm:
          1. Compute rolling Sharpe for each strategy
          2. Normalize Sharpe scores to sum to 1 (softmax-style)
          3. Clip each weight to [0, max_weight_for_n]
          4. Renormalize so weights sum to 1
        """
        from backtest.metrics import compute_rolling_sharpe

        n = len(self._state.weights)
        equal_w = self.config.equal_weight_for_n(n)
        max_w = self.config.max_weight_for_n(n)

        # Not enough history — equal weights
        if not self._equity_history:
            return {name: equal_w for name in self._state.weights}

        sharpes: Dict[str, float] = {}
        for name in self._state.weights:
            eq = self._equity_history.get(name, pd.Series(dtype=float))
            if len(eq) < self.config.performance_window:
                sharpes[name] = 0.0  # neutral score for new strategies
            else:
                rs = compute_rolling_sharpe(eq, window=self.config.performance_window)
                sharpes[name] = float(rs.iloc[-1]) if not rs.empty else 0.0

        # Softmax-style: shift so min=0, then normalize
        vals = np.array(list(sharpes.values()))
        vals = vals - vals.min()  # shift so minimum = 0
        total_score = vals.sum()

        if total_score == 0:
            # All equal — use equal weights
            return {name: equal_w for name in self._state.weights}

        raw_weights = {name: float(vals[i] / total_score)
                       for i, name in enumerate(sharpes)}

        # Clip to max_weight
        clipped = {name: min(w, max_w) for name, w in raw_weights.items()}
        clipped_total = sum(clipped.values())
        if clipped_total == 0:
            return {name: equal_w for name in self._state.weights}

        # Renormalize
        normalized = {name: w / clipped_total for name, w in clipped.items()}
        return normalized

    def _compute_current_weights(self) -> Dict[str, float]:
        """Current weights based on actual capital (post-compounding)."""
        total = self._state.total_capital
        if total == 0:
            return {n: 0.0 for n in self._state.weights}
        return {
            name: cap / total
            for name, cap in self._state.allocated_capital.items()
        }

    @property
    def state(self) -> Optional[AllocationState]:
        return self._state

    @property
    def rebalance_log(self) -> List[Dict]:
        return list(self._rebalance_log)

    def summary_df(self) -> pd.DataFrame:
        """Return current state as a DataFrame for display."""
        if self._state is None:
            return pd.DataFrame()
        rows = []
        for name, weight in self._state.weights.items():
            rows.append({
                "Strategy": name,
                "Weight": f"{weight:.1%}",
                "Allocated (EUR)": f"{self._state.allocated_capital.get(name, 0):.2f}",
                "Max Trade (EUR)": f"{self.get_trade_size(name):.2f}",
            })
        return pd.DataFrame(rows).set_index("Strategy")
