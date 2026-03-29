"""
Central configuration for the AlgoTrading system.
All capital rules, strategy parameters, and system settings live here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (DATA_CACHE_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# PORTFOLIO RULES (from vault: 150 EUR starting capital)
# ─────────────────────────────────────────────
@dataclass
class PortfolioConfig:
    # Starting capital in EUR
    initial_capital: float = 150.0

    # Initial equal-weight allocation across all strategies
    # (auto-computed: 1 / N_strategies)
    initial_weight_per_strategy: float = 1 / 3  # 33.3% for 3 strategies

    # Maximum allocation cap per strategy (dynamic formula below)
    # For N strategies: max_weight = min(1/N + 0.09, 0.60)
    # → N=3: min(0.424, 0.60) ≈ 0.42  (matches vault spec)
    # → N=4: min(0.34, 0.60)  ≈ 0.34
    # → N=2: min(0.59, 0.60)  ≈ 0.59
    max_weight_headroom: float = 0.09  # added on top of equal weight

    # Maximum single trade size as % of strategy's allocated capital
    max_trade_size_pct: float = 0.10  # 10%

    # Rebalance trigger: how much performance divergence (in Sharpe) before rebalancing
    rebalance_threshold: float = 0.10  # 10% relative weight drift

    # Rolling window (in trading days) to evaluate strategy performance for rebalancing
    performance_window: int = 63  # ~3 months

    # Minimum capital per strategy before it is shut down
    min_strategy_capital: float = 5.0  # EUR

    def max_weight_for_n(self, n: int) -> float:
        """Dynamically computed max weight per strategy given N active strategies."""
        return min(1.0 / n + self.max_weight_headroom, 0.60)

    def equal_weight_for_n(self, n: int) -> float:
        return 1.0 / n


# ─────────────────────────────────────────────
# DATA CONFIG
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    # Primary source: yfinance (free, no key required)
    primary_source: str = "yfinance"

    # Default history for backtesting
    backtest_start: str = "2010-01-01"
    backtest_end: Optional[str] = None  # None = today

    # Forward test / live signal lookback window
    live_lookback_days: int = 252  # 1 year to warm up indicators

    # Intraday interval (only used for pairs-VECM in daily mode)
    daily_interval: str = "1d"

    # Cache settings
    cache_dir: Path = DATA_CACHE_DIR
    cache_expiry_hours: int = 4  # re-fetch if data is older than this

    # Pairs universe for Cointegration strategy (sector peers)
    pairs_universe: list = field(default_factory=lambda: [
        # Technology
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "QCOM",
        # Financials
        "JPM", "BAC", "WFC", "GS", "MS", "C",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        # Energy
        "XOM", "CVX", "COP", "SLB",
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "ABBV",
        # ETFs (for RSI strategies)
        "SPY", "QQQ", "IWM",
    ])

    # RSI strategy primary instruments
    rsi_instruments: list = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])


# ─────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class RSI2Config:
    instrument: str = "SPY"
    rsi_period: int = 2
    entry_threshold: float = 15.0   # RSI below this → buy
    exit_threshold: float = 85.0    # RSI above this → sell
    # Optional: only trade when SPY > 200-day MA (disabled by default for more signals)
    trend_filter: bool = False
    trend_ma_period: int = 200


@dataclass
class TripleRSIConfig:
    instrument: str = "SPY"
    rsi_period: int = 5
    oversold_threshold: float = 30.0   # Condition 1: RSI < 30
    exit_threshold: float = 50.0       # Exit: RSI > 50
    decline_days: int = 3              # Condition 2: declining for N days
    lookback_threshold: float = 60.0   # Condition 3: RSI < 60 three days ago
    ma_period: int = 200               # Condition 4: close > 200 MA


@dataclass
class PairsVECMConfig:
    # Cointegration test parameters
    coint_significance: float = 0.05   # 5% significance level
    johansen_det_order: int = 0        # 0 = no trend in cointegrating eq.
    johansen_k_ar_diff: int = 1        # lag order

    # Signal thresholds (from vault: ±1.5 × √Γ₀)
    entry_zscore: float = 1.5
    exit_zscore: float = 0.1

    # Convergence filter: max days to revert (adapted from 6 minutes to 6 days for daily)
    max_convergence_days: int = 6

    # Rolling window for pair selection (in trading days)
    formation_window: int = 252        # 1 year
    trading_window: int = 126          # 6 months

    # Min correlation for pair candidate pre-filtering
    min_correlation: float = 0.70

    # Max pairs to trade simultaneously
    max_active_pairs: int = 3

    # Max holding period (days) before force-close
    max_holding_days: int = 20


# ─────────────────────────────────────────────
# EXECUTION CONFIG
# ─────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    # Mode: "signal_only" | "etoro" | "paper"
    mode: str = "signal_only"

    # eToro credentials (from environment variables — never hardcode)
    etoro_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ETORO_API_KEY")
    )
    etoro_host: str = "https://api.etoro.com"

    # Transaction cost assumptions for backtesting
    commission_pct: float = 0.001     # 0.1% per trade
    slippage_pct: float = 0.001       # 0.1% slippage

    # eToro-specific: minimum order amount in USD (approx)
    etoro_min_order_usd: float = 50.0

    # Signal dashboard refresh interval (seconds) for forward test
    dashboard_refresh_seconds: int = 60


# ─────────────────────────────────────────────
# SINGLETON INSTANCES
# ─────────────────────────────────────────────
PORTFOLIO_CONFIG = PortfolioConfig()
DATA_CONFIG = DataConfig()
EXECUTION_CONFIG = ExecutionConfig()

RSI2_CONFIG = RSI2Config()
TRIPLE_RSI_CONFIG = TripleRSIConfig()
PAIRS_VECM_CONFIG = PairsVECMConfig()


@dataclass
class Settings:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    rsi2: RSI2Config = field(default_factory=RSI2Config)
    triple_rsi: TripleRSIConfig = field(default_factory=TripleRSIConfig)
    pairs_vecm: PairsVECMConfig = field(default_factory=PairsVECMConfig)
