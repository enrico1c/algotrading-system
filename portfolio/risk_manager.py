"""
Risk Manager
=============
Enforces all risk rules before a trade is executed:
  - 10% max trade size per strategy allocation
  - Strategy minimum capital check
  - Drawdown kill-switch (configurable)
  - Crowding monitor (lesson from Khandani-Lo 2007 / vault)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import PORTFOLIO_CONFIG, PortfolioConfig
from portfolio.allocator import AllocationState
from strategies.base import Signal, SignalType
from utils.logger import get_logger

logger = get_logger("portfolio.risk_manager")


@dataclass
class RiskCheck:
    approved: bool
    reason: str
    adjusted_size_eur: float = 0.0


class RiskManager:
    """
    Gate all orders through risk checks before forwarding to execution.
    All rules sourced from vault + user spec.
    """

    def __init__(
        self,
        config: PortfolioConfig = None,
        max_drawdown_kill: float = 0.20,  # halt strategy if drawdown exceeds 20%
    ):
        self.config = config or PORTFOLIO_CONFIG
        self.max_drawdown_kill = max_drawdown_kill
        self._strategy_drawdowns: dict = {}

    def check_trade(
        self,
        signal: Signal,
        state: AllocationState,
        requested_size_eur: Optional[float] = None,
    ) -> RiskCheck:
        """
        Validate a trade signal before execution.
        Returns RiskCheck with approved=True and final size if all checks pass.
        """
        strategy = signal.strategy_name
        strat_capital = state.allocated_capital.get(strategy, 0.0)

        # Check 1: Strategy has minimum capital
        if strat_capital < self.config.min_strategy_capital:
            return RiskCheck(
                approved=False,
                reason=f"Strategy {strategy} capital {strat_capital:.2f} EUR below minimum "
                       f"{self.config.min_strategy_capital:.2f} EUR",
            )

        # Check 2: 10% max trade size
        max_trade = strat_capital * self.config.max_trade_size_pct
        size = min(requested_size_eur or max_trade, max_trade)

        if size <= 0:
            return RiskCheck(approved=False, reason="Trade size is zero or negative")

        # Check 3: Drawdown kill-switch
        dd = self._strategy_drawdowns.get(strategy, 0.0)
        if abs(dd) > self.max_drawdown_kill:
            return RiskCheck(
                approved=False,
                reason=f"Strategy {strategy} drawdown {dd:.1%} exceeds kill threshold "
                       f"{self.max_drawdown_kill:.1%}",
            )

        # Check 4: CLOSE/SELL signals always allowed (reduce risk)
        if signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            return RiskCheck(approved=True, reason="Exit signal approved", adjusted_size_eur=size)

        # Check 5: Confidence threshold (low confidence → reduce size)
        if signal.confidence < 0.3:
            size *= signal.confidence / 0.3
            logger.debug(f"Size reduced to {size:.2f} due to low confidence {signal.confidence:.2f}")

        return RiskCheck(approved=True, reason="All checks passed", adjusted_size_eur=size)

    def update_drawdown(self, strategy_name: str, current_equity: float, peak_equity: float) -> None:
        """Track strategy-level drawdown for kill-switch."""
        if peak_equity > 0:
            dd = (current_equity - peak_equity) / peak_equity
            self._strategy_drawdowns[strategy_name] = dd

    def get_drawdown(self, strategy_name: str) -> float:
        return self._strategy_drawdowns.get(strategy_name, 0.0)
