"""
Strategy registry — auto-discovers all registered strategies.
Adding a new strategy requires only:
  @registry.register("my_strategy")
  class MyStrategy(Strategy): ...

The portfolio allocator calls registry.all() to get the full list.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from utils.logger import get_logger

logger = get_logger("strategies.registry")


class StrategyRegistry:
    """
    Central registry for all trading strategies.

    Strategies self-register via the @registry.register decorator.
    The registry is used by the portfolio allocator to enumerate active strategies.
    """

    def __init__(self):
        self._strategies: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register a Strategy subclass under a given name."""
        def decorator(cls):
            cls.name = name
            self._strategies[name] = cls
            logger.debug(f"Registered strategy: {name} ({cls.__name__})")
            return cls
        return decorator

    def get(self, name: str) -> Optional[Type]:
        return self._strategies.get(name)

    def all(self) -> Dict[str, Type]:
        return dict(self._strategies)

    def names(self) -> List[str]:
        return list(self._strategies.keys())

    def instantiate(self, name: str, config=None):
        """Create and return an instance of the named strategy."""
        cls = self._strategies.get(name)
        if cls is None:
            raise KeyError(f"Strategy '{name}' not found in registry. "
                           f"Available: {self.names()}")
        return cls(config=config)

    def instantiate_all(self, configs: dict = None) -> List:
        """
        Instantiate all registered strategies.
        configs: optional dict mapping strategy name → config object
        """
        configs = configs or {}
        return [cls(config=configs.get(name)) for name, cls in self._strategies.items()]

    def auto_discover(self, package_path: Optional[Path] = None) -> None:
        """
        Auto-import all modules in the strategies package so that
        @registry.register decorators execute and populate the registry.
        """
        if package_path is None:
            package_path = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            if module_name in ("base", "registry", "__init__"):
                continue
            full_name = f"strategies.{module_name}"
            try:
                importlib.import_module(full_name)
                logger.debug(f"Auto-discovered module: {full_name}")
            except Exception as e:
                logger.warning(f"Could not import {full_name}: {e}")


# Global singleton — import this in strategy files
registry = StrategyRegistry()
