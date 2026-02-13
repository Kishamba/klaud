"""
Streamlit UI for Crypto Bot.
Tabs: Data | Backtest | Optimize | Live
Run: streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # Ensure relative paths work

from src.config import load_config, AppConfig
from src.strategies import STRATEGY_REGISTRY
from src.backtest.metrics import drawdown_series


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state init ───────────────────────────────────────────────────────

for key, default in [
    ("backtest_result", None),
    ("optimize_result", None),
    ("live_trader", None),
    ("live_events", []),
    ("live_running", False),
    ("config_path", "configs/config.yaml"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar: Config ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Config")
    config_path = st.text_input("Config file", value=st.session_state.config_path)
    st.session_state.config_path = config_path

    try:
        cfg: AppConfig = load_config(config_path)
        st.success("Config loaded")
    except Exception as e:
        st.error(f"Config error: {e}")
        st.stop()

    st.markdown("---")
    st.markdown("**Risk Settings**")
    st.info(
        f"Capital: ${cfg.risk.initial_capital:,.0f}\n"
        f"Risk/trade: {cfg.risk.risk_per_trade*100:.1f}%\n"
        f"Max leverage: {cfg.risk.max_leverage}x\n"
        f"Max daily loss: {cfg.risk.max_daily_loss_pct}%"
    )
    st.markdown("---")
    st.caption("⚠️ Past performance ≠ future results. Always test on paper first.")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_data, tab_bt, tab_opt, tab_live = st.tabs(["📥 Data", "📈 Backtest", "🔬 Optimize", "🚀 Live"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DATA
# ═══════════════════════════════════════════════════════════════════════════

with tab_data:
    st.header("📥 Data Management")

    col1, col2 = st.columns(2)
    with col1:
        symbols_input = st.text_input("Symbols (comma-separated)", value=",".join(cfg.data.symbols))
        tf_input = st.text_input("Timeframes (comma-separated)", value=",".join(cfg.data.timeframes))
    with col2:
        start_date = st.date_input("Start date (optional)", value=None)
        end_date   = st.date_input("End date (optional)",   value=None)

    # Data status table
    st.subheader("Local Data Status")
    data_rows = []
    for sym in cfg.data.symbols:
        for tf in cfg.data.timeframes:
            p = ROOT / cfg.data.data_dir / sym / f"{tf}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    meta_p = ROOT / cfg.data.data_dir / sym / f"{tf}_meta.json"
                    updated = "-"
                    if meta_p.exists():
                        meta = json.loads(meta_p.read_text())
                        updated = meta.get("updated_at", "-")[:19]
                    data_rows.append({
                        "Symbol": sym, "TF": tf, "Rows": len(df),
                        "Start": str(df.index[0].date()),
                        "End": str(df.index[-1].date()),
                        "Updated": updated, "Status": "✅ OK"
                    })
                except Exception as e:
                    data_rows.append({"Symbol": sym, "TF": tf, "Rows": 0, "Start": "-", "End": "-", "Updated": "-", "Status": f"❌ {e}"})
            else:
                data_rows.append({"Symbol": sym, "TF": tf, "Rows": 0, "Start": "-", "End": "-", "Updated": "-", "Status": "⚠️ Missing"})

    if data_rows:
        st.dataframe(pd.DataFrame(data_rows), use_container_width=True)

    if st.button("⬇️ Download Data", type="primary"):
        args = [
            sys.executable, "-m", "src.cli", "download-data",
            "--config", config_path,
            "--symbols", symbols_input,
            "--timeframes", tf_input,
        ]
        if start_date:
            args += ["--start", str(start_date)]
        if end_date:
            args += ["--end", str(end_date)]

        with st.spinner("Downloading data..."):
            result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Download complete!")
        else:
            st.error("Download failed")

        if result.stdout:
            st.text(result.stdout[-3000:])
        if result.stderr:
            st.warning(result.stderr[-1000:])
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

with tab_bt:
    st.header("📈 Backtest")

    col_cfg, col_run = st.columns([1, 1])

    with col_cfg:
        bt_strategy  = st.selectbox("Strategy", list(STRATEGY_REGISTRY), key="bt_strat")
        bt_symbol    = st.selectbox("Symbol",    cfg.data.symbols, key="bt_sym")
        bt_tf        = st.selectbox("Timeframe", cfg.data.timeframes, key="bt_tf")

        # Strategy-specific params
        st.markdown("**Strategy Parameters**")
        bt_params = {}
        strat_cls = STRATEGY_REGISTRY[bt_strategy]
        dummy = strat_cls()
        for k, v in dummy.params.items():
            if isinstance(v, bool):
                bt_params[k] = st.checkbox(k, value=v, key=f"bt_p_{k}")
            elif isinstance(v, int):
                bt_params[k] = st.number_input(k, value=v, step=1, key=f"bt_p_{k}")
            elif isinstance(v, float):
                bt_params[k] = st.number_input(k, value=v, format="%.4f", key=f"bt_p_{k}")

    with col_run:
        bt_start = st.date_input("From (optional)", value=None, key="bt_start")
        bt_end   = st.date_input("To (optional)",   value=None, key="bt_end")

        if st.button("▶️ Run Backtest", type="primary"):
            # Load data
            data_path = ROOT / cfg.data.data_dir / bt_symbol / f"{bt_tf}.parquet"
            if not data_path.exists():
                st.error(f"No data for {bt_symbol}/{bt_tf}. Run Download Data first.")
            else:
                with st.spinner(f"Running {bt_strategy} on {bt_symbol}/{bt_tf}..."):
                    try:
                        df = pd.read_parquet(data_path)
                        if bt_start:
                            df = df[df.index >= pd.Timestamp(bt_start, tz="UTC")]
                        if bt_end:
                            df = df[df.index <= pd.Timestamp(bt_end, tz="UTC")]

                        from src.backtest.engine import BacktestEngine
                        from src.strategies import get_strategy

                        strat  = get_strategy(bt_strategy, bt_params)
                        engine = BacktestEngine(cfg.backtest, cfg.risk)
                        result = engine.run(df, strat, bt_symbol, bt_tf)
                        st.session_state.backtest_result = result
                        st.success(f"Backtest complete: {len(result.trades)} trades")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────
    result = st.session_state.backtest_result
    if result is not None:
        m = result.metrics

        # KPI tiles
        kpi_cols = st.columns(5)
        kpis = [
            ("Return", f"{m.get('total_return_pct', 0):.1f}%"),
            ("Sharpe", f"{m.get('sharpe', 0):.2f}"),
            ("MaxDD",  f"{m.get('max_drawdown_pct', 0):.1f}%"),
            ("Trades", str(m.get('n_trades', 0))),
            ("Win%",   f"{m.get('win_rate_pct', 0):.1f}%"),
        ]
        for col, (label, val) in zip(kpi_cols, kpis):
            col.metric(label, val)

        # Equity + Drawdown chart
        eq = result.equity_curve
        dd = drawdown_series(eq) * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Equity", "Drawdown %"],
                            row_heights=[0.65, 0.35], vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity",
                                 line=dict(color="royalblue", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                 fill="tozeroy", line=dict(color="crimson")), row=2, col=1)
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Trades table
        trades_df = result.trades_df()
        if not trades_df.empty:
            with st.expander("Trade List", expanded=False):
                show_cols = ["trade_id", "entry_time", "exit_time", "direction",
                             "entry_price", "exit_price", "pnl", "exit_reason"]
                show_cols = [c for c in show_cols if c in trades_df.columns]
                st.dataframe(trades_df[show_cols], use_container_width=True)

        # Export
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = f"results/backtest_{result.strategy_name}_{result.symbol}_{ts}"
        if st.button("💾 Export Report"):
            result.save(out_dir)
            html_path = result.html_report(out_dir)
            st.success(f"Saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: OPTIMIZE
# ═══════════════════════════════════════════════════════════════════════════

with tab_opt:
    st.header("🔬 Optimize Strategy Parameters")

    col_ocfg, col_ores = st.columns([1, 2])

    with col_ocfg:
        opt_strategy = st.selectbox("Strategy", list(STRATEGY_REGISTRY), key="opt_strat")
        opt_symbol   = st.selectbox("Symbol",    cfg.data.symbols, key="opt_sym")
        opt_tf       = st.selectbox("Timeframe", cfg.data.timeframes, key="opt_tf")
        opt_trials   = st.slider("Number of trials", 10, 200, cfg.optimize.n_trials)
        opt_train_r  = st.slider("Train ratio", 0.5, 0.9, cfg.optimize.train_ratio, 0.05)

        if st.button("🚀 Start Optimization", type="primary"):
            data_path = ROOT / cfg.data.data_dir / opt_symbol / f"{opt_tf}.parquet"
            if not data_path.exists():
                st.error(f"No data for {opt_symbol}/{opt_tf}")
            else:
                with st.spinner(f"Optimizing {opt_strategy} ({opt_trials} trials)... this may take a few minutes"):
                    try:
                        df = pd.read_parquet(data_path)
                        from src.optimize.optimizer import Optimizer
                        optimizer = Optimizer(cfg)
                        cfg.optimize.n_trials = opt_trials
                        cfg.optimize.train_ratio = opt_train_r
                        summary = optimizer.optimize(df, opt_strategy, opt_symbol, opt_tf,
                                                     n_trials=opt_trials, train_ratio=opt_train_r)
                        st.session_state.optimize_result = summary
                        st.success("Optimization complete!")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

    with col_ores:
        summary = st.session_state.optimize_result
        if summary:
            st.subheader("Results")

            # Train vs OOS comparison
            tm = summary.get("train_metrics", {})
            om = summary.get("oos_metrics", {})

            cmp_data = {
                "Metric":       ["Return %", "Sharpe", "MaxDD %", "Trades", "Win %"],
                "Train (IS)":   [
                    f"{tm.get('total_return_pct', 0):.1f}",
                    f"{tm.get('sharpe', 0):.2f}",
                    f"{tm.get('max_drawdown_pct', 0):.1f}",
                    str(tm.get('n_trades', 0)),
                    f"{tm.get('win_rate_pct', 0):.1f}",
                ],
                "Test (OOS)":   [
                    f"{om.get('total_return_pct', 0):.1f}",
                    f"{om.get('sharpe', 0):.2f}",
                    f"{om.get('max_drawdown_pct', 0):.1f}",
                    str(om.get('n_trades', 0)),
                    f"{om.get('win_rate_pct', 0):.1f}",
                ],
            }
            st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

            if summary.get("overfit_warning"):
                st.warning(summary["overfit_warning"])

            st.subheader("Best Parameters")
            st.json(summary.get("best_params", {}))

            # Load trials
            trials_files = list(Path("results").glob(f"optimize_{opt_strategy}_{opt_symbol}_*/trials.csv"))
            if trials_files:
                latest = sorted(trials_files)[-1]
                try:
                    trials_df = pd.read_csv(latest)
                    st.subheader("Top 10 Trials")
                    top10 = trials_df.nlargest(10, "score") if "score" in trials_df.columns else trials_df.head(10)
                    st.dataframe(top10, use_container_width=True)

                    # Score distribution
                    fig = go.Figure(go.Histogram(x=trials_df["score"], nbinsx=30,
                                                  marker_color="steelblue"))
                    fig.update_layout(title="Trial Score Distribution",
                                      xaxis_title="Score", height=250,
                                      margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        else:
            st.info("Run optimization to see results.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: LIVE
# ═══════════════════════════════════════════════════════════════════════════

with tab_live:
    st.header("🚀 Live Trading")

    live_mode_options = ["paper", "testnet"]
    live_ok = os.getenv("LIVE_TRADING", "NO").upper() == "YES"
    if live_ok:
        live_mode_options.append("live")

    col_lcfg, col_lstatus = st.columns([1, 2])

    with col_lcfg:
        live_mode   = st.selectbox("Mode", live_mode_options, key="live_mode")
        live_symbol = st.selectbox("Symbol", cfg.data.symbols, key="live_sym")
        live_tf     = st.selectbox("Timeframe", cfg.data.timeframes, key="live_tf")

        if live_mode == "live" and not live_ok:
            st.error("Set LIVE_TRADING=YES in .env to unlock real trading.")

        if not st.session_state.live_running:
            if st.button("▶️ Start Trading", type="primary", disabled=(live_mode == "live" and not live_ok)):
                st.session_state.live_running = True
                st.info(
                    f"Trading started in **{live_mode}** mode.\n\n"
                    "Note: The UI does not run a background thread. "
                    "Use the CLI command instead for continuous operation:\n\n"
                    f"```\npython -m src.cli live --mode {live_mode} --symbol {live_symbol} --tf {live_tf}\n```"
                )
        else:
            if st.button("⏹️ Stop Trading", type="secondary"):
                st.session_state.live_running = False
                if st.session_state.live_trader:
                    st.session_state.live_trader.stop()
                st.success("Trading stopped.")

        st.markdown("---")
        st.markdown("**Emergency Stop**")
        if st.button("🛑 CREATE STOP_TRADING FILE", type="secondary"):
            Path("STOP_TRADING").touch()
            st.error("STOP_TRADING file created. Bot will halt on next iteration.")

        if Path("STOP_TRADING").exists():
            st.error("⚠️ STOP_TRADING file exists! Bot is halted.")
            if st.button("Remove STOP_TRADING"):
                Path("STOP_TRADING").unlink()
                st.success("File removed.")

    with col_lstatus:
        st.subheader("Status")

        live_state_file = Path("live_state.json")
        if live_state_file.exists():
            try:
                state = json.loads(live_state_file.read_text())
                c1, c2, c3 = st.columns(3)
                c1.metric("Equity", f"${state.get('equity', 0):,.2f}")
                c2.metric("Peak",   f"${state.get('peak_equity', 0):,.2f}")
                dd_pct = (state.get("equity", 0) - state.get("peak_equity", 1)) / max(state.get("peak_equity", 1), 1) * 100
                c3.metric("Drawdown", f"{dd_pct:.1f}%")
                st.caption(f"Last updated: {state.get('updated_at', 'N/A')}")
            except Exception:
                st.info("No live state yet.")
        else:
            st.info("No live state file. Start trading to see status.")

        # Events log
        st.subheader("Recent Events")
        log_file = Path("logs/app.log")
        if log_file.exists():
            lines = log_file.read_text().splitlines()[-50:]
            log_text = "\n".join(lines)
            st.text_area("Log tail", value=log_text, height=300, key="log_area")
        else:
            st.info("No log file yet.")

        # CLI command helper
        st.subheader("CLI Commands")
        st.code(
            f"# Start paper trading\n"
            f"python -m src.cli live --mode paper --symbol {live_symbol} --tf {live_tf}\n\n"
            f"# Stop trading (emergency)\n"
            f"touch STOP_TRADING\n\n"
            f"# Remove stop\n"
            f"rm STOP_TRADING",
            language="bash"
        )
