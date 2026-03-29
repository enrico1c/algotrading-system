"""
Performance Reporter — generates console + HTML reports from backtest results.
Metrics targets from vault (Finance/Algo-Trading-Guide/03-Backtesting.md):
  Sharpe > 1.0 | Sortino > 1.5 | Max DD < 25% | Win Rate > 55% | Profit Factor > 1.5
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config.settings import REPORTS_DIR
from strategies.base import StrategyResult
from utils.logger import get_logger

logger = get_logger("reporting.reporter")

# Metric targets from vault
TARGETS = {
    "sharpe": ("Sharpe Ratio", 1.0, ">"),
    "sortino": ("Sortino Ratio", 1.5, ">"),
    "max_drawdown": ("Max Drawdown", -0.25, ">"),   # less negative is better
    "win_rate": ("Win Rate", 0.55, ">"),
    "profit_factor": ("Profit Factor", 1.5, ">"),
    "calmar": ("Calmar Ratio", 1.0, ">"),
}


class PerformanceReporter:
    """
    Generates backtest performance reports:
      - Console table summary
      - Per-strategy equity curve plots
      - Portfolio-level combined report
      - HTML summary file
    """

    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self, results: Dict[str, StrategyResult]) -> None:
        """Print formatted performance table to console."""
        print("\n" + "=" * 80)
        print("  BACKTEST RESULTS SUMMARY")
        print("=" * 80)

        rows = []
        for name, result in results.items():
            m = result.metrics
            rows.append({
                "Strategy": name,
                "CAGR": f"{m.get('cagr', 0):.1%}",
                "Sharpe": self._fmt_metric(m.get("sharpe", 0), "sharpe"),
                "Sortino": self._fmt_metric(m.get("sortino", 0), "sortino"),
                "Max DD": self._fmt_metric(m.get("max_drawdown", 0), "max_drawdown"),
                "Win Rate": self._fmt_metric(m.get("win_rate", 0), "win_rate"),
                "Pft Factor": self._fmt_metric(m.get("profit_factor", 0), "profit_factor"),
                "Trades": int(m.get("n_trades", 0)),
                "Calmar": f"{m.get('calmar', 0):.2f}",
            })

        df = pd.DataFrame(rows)
        try:
            from tabulate import tabulate
            print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
        except ImportError:
            print(df.to_string(index=False))

        print("=" * 80)
        print("  Targets: Sharpe>1.0 | Sortino>1.5 | MaxDD<25% | WinRate>55% | ProfitFactor>1.5")
        print("=" * 80 + "\n")

    def print_allocation(self, allocator) -> None:
        """Print current portfolio allocation table."""
        if allocator.state is None:
            return
        print("\n" + "─" * 60)
        print("  PORTFOLIO ALLOCATION")
        print("─" * 60)
        df = allocator.summary_df()
        try:
            from tabulate import tabulate
            print(tabulate(df, headers="keys", tablefmt="simple"))
        except ImportError:
            print(df.to_string())
        print("─" * 60 + "\n")

    def plot_equity_curves(
        self,
        results: Dict[str, StrategyResult],
        filename: str = "equity_curves.png",
    ) -> Path:
        """Plot all strategy equity curves on one figure."""
        n = len(results)
        if n == 0:
            return None

        fig, axes = plt.subplots(n + 1, 1, figsize=(14, 4 * (n + 1)))
        if n == 1:
            axes = [axes, axes]  # ensure iterable

        fig.suptitle("Strategy Equity Curves", fontsize=14, fontweight="bold")
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

        combined_equity = None
        for i, (name, result) in enumerate(results.items()):
            eq = result.equity_curve
            ax = axes[i]
            ax.plot(eq.index, eq.values, color=colors[i % len(colors)], linewidth=1.5, label=name)
            ax.fill_between(eq.index, eq.values, eq.values.min(), alpha=0.1, color=colors[i % len(colors)])
            m = result.metrics
            ax.set_title(
                f"{name}  |  CAGR={m.get('cagr', 0):.1%}  "
                f"Sharpe={m.get('sharpe', 0):.2f}  "
                f"MaxDD={m.get('max_drawdown', 0):.1%}  "
                f"WinRate={m.get('win_rate', 0):.1%}",
                fontsize=9,
            )
            ax.set_ylabel("Portfolio Value (EUR)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

            if combined_equity is None:
                combined_equity = eq.copy()
            else:
                combined_equity = combined_equity.add(eq, fill_value=0)

        # Combined portfolio equity
        if combined_equity is not None:
            ax_comb = axes[-1]
            ax_comb.plot(combined_equity.index, combined_equity.values,
                         color="#E91E63", linewidth=2, label="Combined Portfolio")
            ax_comb.fill_between(combined_equity.index, combined_equity.values,
                                  combined_equity.values.min(), alpha=0.1, color="#E91E63")
            ax_comb.set_title("Combined Portfolio", fontsize=9)
            ax_comb.set_ylabel("Total Value (EUR)")
            ax_comb.grid(True, alpha=0.3)
            ax_comb.legend(loc="upper left")

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Equity curves saved: {output_path}")
        return output_path

    def plot_drawdowns(
        self,
        results: Dict[str, StrategyResult],
        filename: str = "drawdowns.png",
    ) -> Path:
        """Plot drawdown profiles for each strategy."""
        if not results:
            return None

        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        for i, (name, result) in enumerate(results.items()):
            eq = result.equity_curve
            peak = eq.cummax()
            dd = (eq - peak) / peak * 100
            ax.fill_between(dd.index, dd.values, 0, alpha=0.4,
                            color=colors[i % len(colors)], label=name)

        ax.axhline(-25, color="red", linestyle="--", alpha=0.6, label="25% threshold")
        ax.set_title("Strategy Drawdowns (%)", fontsize=12)
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Drawdown chart saved: {output_path}")
        return output_path

    def save_html_report(
        self,
        results: Dict[str, StrategyResult],
        filename: str = "backtest_report.html",
    ) -> Path:
        """Save a complete HTML performance report."""
        rows_html = ""
        for name, result in results.items():
            m = result.metrics
            rows_html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{m.get('cagr', 0):.1%}</td>
                <td>{m.get('sharpe', 0):.2f}</td>
                <td>{m.get('sortino', 0):.2f}</td>
                <td>{m.get('max_drawdown', 0):.1%}</td>
                <td>{m.get('win_rate', 0):.1%}</td>
                <td>{m.get('profit_factor', 0):.2f}</td>
                <td>{int(m.get('n_trades', 0))}</td>
                <td>{m.get('calmar', 0):.2f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
<title>AlgoTrading Backtest Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
  th {{ background: #1565C0; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f0f4ff; }}
  .target {{ font-size: 12px; color: #666; margin-top: 10px; }}
  img {{ max-width: 100%; margin-top: 20px; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>AlgoTrading Backtest Report</h1>
<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Strategy Performance</h2>
<table>
  <tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th>
      <th>Max DD</th><th>Win Rate</th><th>Profit Factor</th><th>Trades</th><th>Calmar</th></tr>
  {rows_html}
</table>
<p class="target">Targets: Sharpe &gt;1.0 | Sortino &gt;1.5 | MaxDD &lt;25% | Win Rate &gt;55% | Profit Factor &gt;1.5</p>
<img src="equity_curves.png" alt="Equity Curves">
<img src="drawdowns.png" alt="Drawdowns">
</body></html>"""

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved: {output_path}")
        return output_path

    # ──────────────────────────────────────────

    def _fmt_metric(self, value: float, key: str) -> str:
        """Format metric value with pass/fail indicator."""
        _, target, direction = TARGETS.get(key, ("", 0, ">"))
        passes = (value > target) if direction == ">" else (value < target)
        if key in ("max_drawdown",):
            fmt = f"{value:.1%}"
        elif key == "win_rate":
            fmt = f"{value:.1%}"
        else:
            fmt = f"{value:.2f}"
        return f"{fmt} {'✓' if passes else '✗'}"
