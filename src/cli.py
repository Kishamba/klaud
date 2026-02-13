"""
CLI entry point.
Usage:
  python -m src.cli download-data [--config] [--symbols] [--start] [--end]
  python -m src.cli backtest [--config] [--strategy] [--symbol] [--tf]
  python -m src.cli optimize [--config] [--strategy] [--symbol] [--tf] [--trials]
  python -m src.cli live [--config] [--symbol] [--tf] [--mode]
  python -m src.cli sanity-check [--config]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.utils.helpers import setup_logging, parse_date, ensure_dirs

app = typer.Typer(
    name="crypto-bot",
    help="Bybit Crypto Trading Bot — data / backtest / optimize / live",
    no_args_is_help=True,
)
console = Console()


def _load_cfg(config: str):
    from src.config import load_config
    return load_config(config)


# ─────────────────────────────────────────────────────────────────────────────
# download-data
# ─────────────────────────────────────────────────────────────────────────────

@app.command("download-data")
def download_data(
    config:    str           = typer.Option("configs/config.yaml", "--config", "-c"),
    symbols:   Optional[str] = typer.Option(None, "--symbols", help="Comma-separated, e.g. BTCUSDT,ETHUSDT"),
    timeframes:Optional[str] = typer.Option(None, "--timeframes", help="Comma-separated, e.g. 15,60"),
    start:     Optional[str] = typer.Option(None, "--start", help="ISO date e.g. 2022-01-01"),
    end:       Optional[str] = typer.Option(None, "--end",   help="ISO date e.g. 2024-01-01"),
    log_level: str           = typer.Option("INFO", "--log-level"),
):
    """Download OHLCV candles from Bybit and store as Parquet."""
    setup_logging(log_level)
    cfg = _load_cfg(config)
    ensure_dirs("data/candles", "logs")

    from src.exchange.client import ExchangeClient
    from src.data.downloader import DataDownloader

    sym_list = [s.strip() for s in symbols.split(",")] if symbols else cfg.data.symbols
    tf_list  = [t.strip() for t in timeframes.split(",")] if timeframes else cfg.data.timeframes
    start_dt = parse_date(start) if start else None
    end_dt   = parse_date(end)   if end   else None

    rprint(f"[bold cyan]Downloading data for:[/bold cyan] {sym_list} | TFs: {tf_list}")

    client     = ExchangeClient(cfg.exchange.testnet, cfg.exchange.max_retries, cfg.exchange.retry_delay)
    downloader = DataDownloader(client, cfg.data)

    results = downloader.download_all(sym_list, tf_list, start=start_dt, end=end_dt)

    # Summary table
    table = Table(title="Download Summary")
    table.add_column("Symbol", style="cyan")
    table.add_column("TF")
    table.add_column("Rows", justify="right")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Status")

    for sym, tfs in results.items():
        for tf, df in tfs.items():
            if df.empty:
                table.add_row(sym, tf, "0", "-", "-", "[red]FAILED[/red]")
            else:
                table.add_row(
                    sym, tf, str(len(df)),
                    str(df.index[0].date()), str(df.index[-1].date()),
                    "[green]OK[/green]"
                )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# backtest
# ─────────────────────────────────────────────────────────────────────────────

@app.command("backtest")
def backtest(
    config:   str           = typer.Option("configs/config.yaml", "--config", "-c"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="TrendStrategy|MeanReversionStrategy|RegimeStrategy"),
    symbol:   str           = typer.Option("BTCUSDT",  "--symbol"),
    tf:       str           = typer.Option("60",       "--tf",       help="Bybit interval: 15 or 60"),
    start:    Optional[str] = typer.Option(None,       "--start"),
    end:      Optional[str] = typer.Option(None,       "--end"),
    output:   str           = typer.Option("results",  "--output",   help="Output directory"),
    log_level:str           = typer.Option("INFO",     "--log-level"),
):
    """Run backtest on stored candle data."""
    setup_logging(log_level)
    cfg = _load_cfg(config)
    ensure_dirs(output, "logs")

    from src.data.downloader import DataDownloader
    from src.exchange.client import ExchangeClient
    from src.strategies import get_strategy
    from src.backtest.engine import BacktestEngine
    from datetime import datetime, timezone

    strategy_name = strategy or cfg.strategy.name
    client     = ExchangeClient(cfg.exchange.testnet, cfg.exchange.max_retries, cfg.exchange.retry_delay)
    downloader = DataDownloader(client, cfg.data)

    df = downloader.load(symbol, tf)
    if df is None or df.empty:
        rprint(f"[red]No data for {symbol}/{tf}. Run download-data first.[/red]")
        raise typer.Exit(1)

    # Optional date filtering
    if start:
        df = df[df.index >= parse_date(start)]
    if end:
        df = df[df.index <= parse_date(end)]

    rprint(f"[bold cyan]Backtest:[/bold cyan] {strategy_name} on {symbol}/{tf} | {len(df)} bars")

    strat  = get_strategy(strategy_name, cfg.strategy.params)
    engine = BacktestEngine(cfg.backtest, cfg.risk)
    result = engine.run(df, strat, symbol, tf)

    # Save results
    from datetime import datetime, timezone
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output) / f"backtest_{strategy_name}_{symbol}_{ts}"
    result.save(str(out))
    html_path = result.html_report(str(out))

    # Print metrics
    m = result.metrics
    table = Table(title=f"Backtest Results: {strategy_name} {symbol}/{tf}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    rows = [
        ("Total Return", f"{m.get('total_return_pct', 0):.2f}%"),
        ("CAGR",         f"{m.get('cagr_pct', 0):.2f}%"),
        ("Sharpe",       f"{m.get('sharpe', 0):.3f}"),
        ("Sortino",      f"{m.get('sortino', 0):.3f}"),
        ("Calmar",       f"{m.get('calmar', 0):.3f}"),
        ("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.2f}%"),
        ("Trades",       str(m.get('n_trades', 0))),
        ("Win Rate",     f"{m.get('win_rate_pct', 0):.1f}%"),
        ("Profit Factor",f"{m.get('profit_factor', 0):.2f}"),
        ("Final Equity", f"${m.get('final_equity', 0):.2f}"),
    ]
    for k, v in rows:
        table.add_row(k, v)

    console.print(table)
    rprint(f"[green]HTML report:[/green] {html_path}")
    rprint(f"[green]Results saved to:[/green] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# optimize
# ─────────────────────────────────────────────────────────────────────────────

@app.command("optimize")
def optimize(
    config:   str           = typer.Option("configs/config.yaml", "--config", "-c"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s"),
    symbol:   str           = typer.Option("BTCUSDT",  "--symbol"),
    tf:       str           = typer.Option("60",       "--tf"),
    trials:   Optional[int] = typer.Option(None,       "--trials",   help="Override n_trials from config"),
    output:   str           = typer.Option("results",  "--output"),
    log_level:str           = typer.Option("INFO",     "--log-level"),
):
    """Optimize strategy parameters with Optuna (walk-forward)."""
    setup_logging(log_level)
    cfg = _load_cfg(config)
    ensure_dirs(output, "logs")

    from src.data.downloader import DataDownloader
    from src.exchange.client import ExchangeClient
    from src.optimize.optimizer import Optimizer

    strategy_name = strategy or cfg.strategy.name
    client     = ExchangeClient(cfg.exchange.testnet, cfg.exchange.max_retries, cfg.exchange.retry_delay)
    downloader = DataDownloader(client, cfg.data)

    df = downloader.load(symbol, tf)
    if df is None or df.empty:
        rprint(f"[red]No data for {symbol}/{tf}. Run download-data first.[/red]")
        raise typer.Exit(1)

    rprint(f"[bold cyan]Optimizing:[/bold cyan] {strategy_name} on {symbol}/{tf} | {len(df)} bars")

    optimizer = Optimizer(cfg)
    summary   = optimizer.optimize(
        df, strategy_name, symbol, tf,
        n_trials=trials or cfg.optimize.n_trials,
        output_dir=output,
    )

    if not summary:
        rprint("[red]Optimization failed (no param_space?)[/red]")
        raise typer.Exit(1)

    table = Table(title="Optimization Results")
    table.add_column("Metric",   style="cyan")
    table.add_column("Value",    justify="right")
    table.add_column("OOS",      justify="right")

    tm = summary.get("train_metrics", {})
    om = summary.get("oos_metrics", {})

    for key, label in [
        ("total_return_pct", "Total Return"),
        ("sharpe",           "Sharpe"),
        ("max_drawdown_pct", "Max Drawdown"),
        ("n_trades",         "Trades"),
        ("win_rate_pct",     "Win Rate"),
    ]:
        tv = tm.get(key, 0)
        ov = om.get(key, 0)
        if isinstance(tv, float):
            table.add_row(label, f"{tv:.2f}", f"{ov:.2f}")
        else:
            table.add_row(label, str(tv), str(ov))

    console.print(table)

    rprint("[bold]Best params:[/bold]")
    for k, v in summary.get("best_params", {}).items():
        rprint(f"  {k}: {v}")

    if summary.get("overfit_warning"):
        rprint(f"[yellow]{summary['overfit_warning']}[/yellow]")

    rprint(f"[green]Best config saved to:[/green] configs/best.json")


# ─────────────────────────────────────────────────────────────────────────────
# live
# ─────────────────────────────────────────────────────────────────────────────

@app.command("live")
def live_trade(
    config:    str  = typer.Option("configs/config.yaml", "--config", "-c"),
    symbol:    str  = typer.Option("BTCUSDT", "--symbol"),
    tf:        str  = typer.Option("60",      "--tf"),
    mode:      str  = typer.Option("paper",   "--mode", help="paper | testnet | live"),
    log_level: str  = typer.Option("INFO",    "--log-level"),
):
    """Run live/paper/testnet trading loop."""
    setup_logging(log_level)
    cfg = _load_cfg(config)
    ensure_dirs("logs")

    from src.live.trader import LiveTrader

    rprint(f"[bold {'red' if mode == 'live' else 'yellow'}]Starting {mode.upper()} trading[/bold {'red' if mode == 'live' else 'yellow'}]")
    rprint(f"Strategy: {cfg.strategy.name} | Symbol: {symbol}/{tf}")

    if mode == "live":
        rprint("[bold red]WARNING: REAL MONEY TRADING. Ensure LIVE_TRADING=YES is set.[/bold red]")
        confirm = typer.confirm("Are you sure you want to start LIVE trading?")
        if not confirm:
            raise typer.Exit(0)

    rprint("[dim]Press Ctrl+C to stop[/dim]")

    trader = LiveTrader(cfg, mode=mode)
    try:
        trader.run(symbol=symbol, tf=tf)
    except KeyboardInterrupt:
        rprint("[yellow]Stopped by user[/yellow]")
    except Exception as e:
        rprint(f"[red]Fatal error: {e}[/red]")
        logger.exception(e)
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# sanity-check
# ─────────────────────────────────────────────────────────────────────────────

@app.command("sanity-check")
def sanity_check(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
):
    """Check data availability, config validity, and API connectivity."""
    setup_logging("INFO")
    from src.config import load_config
    from src.exchange.client import ExchangeClient
    from src.data.downloader import DataDownloader

    ok = True
    results = []

    # 1. Config loads
    try:
        cfg = load_config(config)
        results.append(("[green]✓[/green]", "Config", f"Loaded from {config}"))
    except Exception as e:
        results.append(("[red]✗[/red]", "Config", str(e)))
        ok = False
        cfg = None

    if cfg is None:
        for r in results:
            rprint(f"{r[0]} {r[1]:20s} {r[2]}")
        raise typer.Exit(1)

    # 2. Local data files
    from pathlib import Path
    for sym in cfg.data.symbols:
        for tf in cfg.data.timeframes:
            p = Path(cfg.data.data_dir) / sym / f"{tf}.parquet"
            if p.exists():
                import pandas as pd
                df = pd.read_parquet(p)
                results.append(("[green]✓[/green]", f"Data {sym}/{tf}", f"{len(df)} rows"))
            else:
                results.append(("[yellow]![/yellow]", f"Data {sym}/{tf}", "Missing — run download-data"))

    # 3. API keys not in code (just check they are in .env and not hardcoded)
    import os
    api_key = os.getenv("BYBIT_API_KEY", "")
    if api_key and len(api_key) > 5:
        results.append(("[green]✓[/green]", "API Key", "Present in environment"))
    else:
        results.append(("[yellow]![/yellow]", "API Key", "Not set — only needed for live trading"))

    # 4. Exchange connectivity (no auth needed for klines)
    try:
        client = ExchangeClient(testnet=cfg.exchange.testnet)
        ts = client.get_server_time()
        results.append(("[green]✓[/green]", "Exchange API", f"Connected, server_time={ts}"))
    except Exception as e:
        results.append(("[red]✗[/red]", "Exchange API", str(e)))
        ok = False

    # 5. Strategy import
    try:
        from src.strategies import get_strategy, STRATEGY_REGISTRY
        results.append(("[green]✓[/green]", "Strategies", f"Available: {list(STRATEGY_REGISTRY)}"))
    except Exception as e:
        results.append(("[red]✗[/red]", "Strategies", str(e)))
        ok = False

    # 6. LIVE_TRADING guard
    import os
    live_flag = os.getenv("LIVE_TRADING", "NO").upper()
    color = "[red]!" if live_flag == "YES" else "[green]✓"
    results.append((f"{color}[/{'red' if live_flag == 'YES' else 'green'}]",
                    "LIVE_TRADING", live_flag))

    table = Table(title="Sanity Check")
    table.add_column("", width=3)
    table.add_column("Check", style="cyan")
    table.add_column("Detail")
    for icon, name, detail in results:
        table.add_row(icon, name, detail)

    console.print(table)

    if ok:
        rprint("[bold green]All critical checks passed.[/bold green]")
    else:
        rprint("[bold red]Some checks failed. See above.[/bold red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
