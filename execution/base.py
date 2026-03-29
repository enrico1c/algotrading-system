"""Abstract execution layer — decouples signal generation from order routing."""
from __future__ import annotations
from abc import ABC, abstractmethod
from strategies.base import Signal


class ExecutionBase(ABC):
    """
    Abstract base for all execution backends.
    Signal generation is fully independent of execution — swap backends
    without touching any strategy code.
    """

    @abstractmethod
    def execute(self, signal: Signal, size_eur: float) -> dict:
        """
        Route an approved signal to the broker/execution layer.
        Returns a dict with execution details (order_id, status, fill_price, etc.)
        """
        ...

    @abstractmethod
    def get_open_positions(self) -> list:
        """Return list of currently open positions."""
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return current account balance in EUR."""
        ...
