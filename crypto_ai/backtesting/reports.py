"""
Human-readable backtest reports and charts (Section 16).
"""

from __future__ import annotations

from pathlib import Path

from crypto_ai.backtesting.engine import BacktestReport


def format_text_report(report: BacktestReport) -> str:
    d = report.to_dict()
    metrics = d["metrics"]
    lines = [
        "BACKTEST REPORT",
        f"Symbol: {report.symbol}   Timeframe: {report.timeframe}",
        "",
        f"Initial balance:        ${d['initial_balance']:,.2f}",
        f"Final balance:           ${d['final_balance']:,.2f}",
        f"Net profit:              ${d['net_profit']:,.2f}",
        f"Return:                  {d['return_pct']:.2f}%",
        f"Buy-and-hold return:     {d['buy_and_hold_return_pct']:.2f}%",
        "",
        f"Number of trades:        {metrics['n_trades']}",
        f"Win rate:                {metrics['win_rate_pct']:.1f}%",
        f"Average win:             ${metrics['avg_win']:,.2f}",
        f"Average loss:            ${metrics['avg_loss']:,.2f}",
        f"Profit factor:           {metrics['profit_factor']:.2f}",
        f"Maximum drawdown:        {metrics['max_drawdown_pct']:.2f}%",
        f"Sharpe ratio:            {metrics['sharpe_ratio']:.2f}",
        f"Sortino ratio:           {metrics['sortino_ratio']:.2f}",
        f"Fees paid:               ${d['fees_paid']:,.2f}",
    ]
    if metrics.get("suspicious_flags"):
        lines.append("")
        lines.append("WARNINGS:")
        for flag in metrics["suspicious_flags"]:
            lines.append(f"  - {flag}")
    lines.append("")
    lines.append("Past performance does not guarantee future results.")
    return "\n".join(lines)


def save_equity_curve_chart(report: BacktestReport, output_path: str | Path) -> Path:
    """
    Saves a PNG chart of the equity curve vs. a buy-and-hold reference
    line. Matplotlib is used in a headless-safe way (Agg backend) so
    this works on a server with no display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(report.equity_curve.index, report.equity_curve.values, label="Strategy equity")

    if len(report.equity_curve) > 1:
        bh_start = report.initial_balance
        bh_end = report.buy_and_hold_final_balance
        bh_line = [
            bh_start + (bh_end - bh_start) * i / (len(report.equity_curve) - 1)
            for i in range(len(report.equity_curve))
        ]
        ax.plot(report.equity_curve.index, bh_line, label="Buy & hold (approx.)", linestyle="--")

    ax.set_title(f"Backtest equity curve — {report.symbol} ({report.timeframe})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USDT)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
