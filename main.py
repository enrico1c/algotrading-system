"""
AlgoTrading System — Main Entry Point
========================================
Usage:
  python main.py backtest              # Run all strategy backtests
  python main.py backtest --strategy rsi2_mean_reversion
  python main.py walkforward           # Walk-forward validation
  python main.py forward               # Start live signal generation loop
  python main.py signal                # Generate signals once and exit
  python main.py portfolio             # Show current portfolio state
  python main.py strategies            # List all registered strategies

Capital: 150 EUR | Mode: SIGNAL_ONLY (eToro manual execution)
Data:    yfinance (free, no API key)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger("main")


def cmd_backtest(args, settings: Settings) -> None:
    from backtest.engine import BacktestEngine
    from reporting.reporter import PerformanceReporter
    from portfolio.allocator import PortfolioAllocator

    engine = BacktestEngine(settings)
    reporter = PerformanceReporter()

    if args.strategy:
        results = {args.strategy: engine.run_single(args.strategy)}
    else:
        results = engine.run_all(start=args.start, end=args.end)

    reporter.print_summary(results)

    # Portfolio allocation display
    from strategies.registry import registry
    registry.auto_discover()
    allocator = PortfolioAllocator(settings.portfolio)
    allocator.initialize(list(results.keys()))
    reporter.print_allocation(allocator)

    if not args.no_plots:
        reporter.plot_equity_curves(results)
        reporter.plot_drawdowns(results)
        reporter.save_html_report(results)
        print(f"\nReports saved to: {reporter.output_dir}")


def cmd_walkforward(args, settings: Settings) -> None:
    from backtest.engine import BacktestEngine
    from reporting.reporter import PerformanceReporter

    engine = BacktestEngine(settings)
    reporter = PerformanceReporter()
    strategy = args.strategy or "rsi2_mean_reversion"

    print(f"\nWalk-forward validation: {strategy} ({args.splits} splits)")
    results = engine.run_walkforward(strategy, n_splits=args.splits)

    if not results:
        print("No results — insufficient data")
        return

    import pandas as pd
    fold_rows = []
    for r in results:
        m = r.metrics
        fold_rows.append({
            "Fold": int(m.get("fold", 0)) + 1,
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "CAGR": f"{m.get('cagr', 0):.1%}",
            "Max DD": f"{m.get('max_drawdown', 0):.1%}",
            "Win Rate": f"{m.get('win_rate', 0):.1%}",
            "Trades": int(m.get("n_trades", 0)),
        })
    df = pd.DataFrame(fold_rows)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        print(df.to_string(index=False))

    sharpes = [r.metrics.get("sharpe", 0) for r in results]
    import numpy as np
    print(f"\nRobustness: mean Sharpe={np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")


def cmd_forward(args, settings: Settings) -> None:
    from forward_test.runner import ForwardTestRunner

    runner = ForwardTestRunner(settings)
    runner.initialize()
    print(f"\nStarting forward-test loop (interval={settings.execution.dashboard_refresh_seconds}s)")
    print("Signals will be printed with eToro execution instructions.")
    print("Press Ctrl+C to stop.\n")
    runner.run_loop()


def cmd_signal(args, settings: Settings) -> None:
    """Generate signals once and exit — useful for cron-based automation."""
    from forward_test.runner import ForwardTestRunner

    runner = ForwardTestRunner(settings)
    runner.initialize()
    signals = runner.run_once()

    if not signals:
        print("No signals generated at this time.")
    else:
        print(f"\n{len(signals)} signal(s) generated:")
        for sig in signals:
            print(f"  {sig}")


def cmd_portfolio(args, settings: Settings) -> None:
    from portfolio.allocator import PortfolioAllocator
    from strategies.registry import registry
    from reporting.reporter import PerformanceReporter

    registry.auto_discover()
    allocator = PortfolioAllocator(settings.portfolio)
    allocator.initialize(registry.names())
    reporter = PerformanceReporter()
    reporter.print_allocation(allocator)
    print(str(allocator.state))


def cmd_strategies(args, settings: Settings) -> None:
    from strategies.registry import registry
    registry.auto_discover()
    names = registry.names()
    print(f"\nRegistered strategies ({len(names)}):")
    for name in names:
        cls = registry.get(name)
        print(f"  • {name}  ({cls.__name__})")

    n = len(names)
    cfg = settings.portfolio
    print(f"\nPortfolio rules for N={n} strategies:")
    print(f"  Starting capital : {cfg.initial_capital:.2f} EUR")
    print(f"  Equal weight     : {cfg.equal_weight_for_n(n):.1%}")
    print(f"  Max weight       : {cfg.max_weight_for_n(n):.1%}")
    print(f"  Max trade size   : {cfg.max_trade_size_pct:.0%} of strategy allocation")


# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlgoTrading System — 150 EUR portfolio | 3 strategies | yfinance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # backtest
    bt = sub.add_parser("backtest", help="Run strategy backtests")
    bt.add_argument("--strategy", type=str, help="Single strategy name (default: all)")
    bt.add_argument("--start", type=str, default="2010-01-01")
    bt.add_argument("--end", type=str, default=None)
    bt.add_argument("--no-plots", action="store_true", help="Skip chart generation")

    # walk-forward
    wf = sub.add_parser("walkforward", help="Walk-forward validation")
    wf.add_argument("--strategy", type=str, default="rsi2_mean_reversion")
    wf.add_argument("--splits", type=int, default=5)

    # forward test
    sub.add_parser("forward", help="Start live signal loop (Ctrl+C to stop)")

    # single signal
    sub.add_parser("signal", help="Generate signals once and exit")

    # portfolio
    sub.add_parser("portfolio", help="Show portfolio allocation state")

    # strategies
    sub.add_parser("strategies", help="List registered strategies and rules")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    commands = {
        "backtest": cmd_backtest,
        "walkforward": cmd_walkforward,
        "forward": cmd_forward,
        "signal": cmd_signal,
        "portfolio": cmd_portfolio,
        "strategies": cmd_strategies,
    }

    if args.command is None:
        parser.print_help()
        return

    fn = commands.get(args.command)
    if fn:
        fn(args, settings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
