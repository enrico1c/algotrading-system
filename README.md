# AlgoTrading System

A modular, scalable algorithmic trading framework designed for **eToro** execution with **150 EUR** starting capital. Uses only **free data sources** (no paid APIs required).

## Strategies

| Strategy | Win Rate | Timeframe | Type |
|---|---|---|---|
| RSI(2) Mean Reversion | 91% | Daily | Long-only, mean reversion |
| Triple RSI | 90% | Daily | Long-only, 4-condition filter |
| Cointegration Pairs (VECM) | 98.49%* | Daily | Statistical arbitrage |

*Original win rate on tick data. Adapted to daily bars in this implementation.

## Portfolio Rules

| Rule | Value |
|---|---|
| Starting capital | 150 EUR |
| Initial weight per strategy | 33.3% (equal split) |
| Max weight per strategy | `1/N + 9%` → 42% for 3 strategies |
| Max single trade size | 10% of strategy allocation |
| Rebalancing | Automatic, rolling Sharpe-based |

## Architecture

```
config/         Settings, capital rules, strategy parameters
data/           DataFetcher (yfinance + FRED, free, no API key)
indicators/     RSI, SMA, EMA, ATR, spread z-score
strategies/     Base class + auto-discovery registry
  rsi2_mean_reversion.py
  triple_rsi.py
  cointegration_pairs_vecm.py
portfolio/      Dynamic allocator + risk manager
backtest/       Vectorized engine + performance metrics
forward_test/   Live signal loop + eToro console dashboard
execution/      Signal-only (default) + eToro REST client
reporting/      Console tables + equity curve charts + HTML report
main.py         CLI entry point
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/enrico1c/algotrading-system.git
cd algotrading-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run backtests (downloads real SPY data via yfinance — no API key needed)
python main.py backtest

# 4. Walk-forward validation
python main.py walkforward --strategy rsi2_mean_reversion --splits 5

# 5. Generate live signals once
python main.py signal

# 6. Start continuous signal loop with eToro dashboard
python main.py forward
```

## CLI Reference

```
python main.py backtest                          Run all strategy backtests
python main.py backtest --strategy NAME          Single strategy
python main.py backtest --start 1993-01-01       Custom date range
python main.py backtest --no-plots               Skip chart generation

python main.py walkforward                       Walk-forward validation
python main.py walkforward --splits 10           Custom number of folds

python main.py signal                            Generate signals once (for cron)
python main.py forward                           Continuous signal loop (Ctrl+C)

python main.py portfolio                         Show current allocation state
python main.py strategies                        List registered strategies + rules
```

## Adding a New Strategy

1. Create `strategies/my_strategy.py`
2. Subclass `Strategy` and decorate with `@registry.register("name")`
3. Implement `generate_signals()` and `backtest()`
4. Done — portfolio rebalances automatically across N+1 strategies

```python
from strategies.base import Strategy, Signal, StrategyResult
from strategies.registry import registry

@registry.register("my_strategy")
class MyStrategy(Strategy):
    name = "my_strategy"

    def generate_signals(self, data):
        # ... return List[Signal]

    def backtest(self, data, initial_capital, commission, slippage):
        # ... return StrategyResult
```

## eToro Connection

The system runs in **signal-only mode** by default — signals are displayed with
step-by-step manual execution instructions for eToro.

To enable automated execution:

1. Apply for eToro developer API key at [api-portal.etoro.com](https://api-portal.etoro.com)
2. Set environment variable: `export ETORO_API_KEY=your_key_here`
3. Change execution mode in `config/settings.py`: `mode = "etoro"`

See [ETORO_SETUP.md](ETORO_SETUP.md) for full connection guide.

## Data Sources (all free)

| Source | Used for | API Key? |
|---|---|---|
| Yahoo Finance (yfinance) | All OHLCV data | No |
| FRED (pandas-datareader) | Risk-free rate | No |

## Performance Metrics Tracked

Sharpe Ratio · Sortino Ratio · CAGR · Max Drawdown · Win Rate · Profit Factor · Calmar Ratio

Targets from research: Sharpe > 1.0 · Sortino > 1.5 · Max DD < 25% · Win Rate > 55%

## License

MIT
