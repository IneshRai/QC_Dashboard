"""
qc_charts.py
============
Chart builders for the QC dashboard. Each builder takes a BacktestResults (or
two, for comparisons) and returns a matplotlib Figure. Returning figures (not
Streamlit calls) means the SAME function feeds both the live dashboard and the
PDF export -- no duplicated plotting code.

To add a new chart later:
  1. Write a function build_<something>(results, ...) -> matplotlib Figure.
  2. Register it in CHART_REGISTRY with a key, label, and whether it supports
     comparison mode.
The dashboard and PDF pick it up automatically from the registry.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")            # headless backend; safe for Streamlit + PDF
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd

from qc_brand import (PRIMARY_COLOR, COMPARE_COLOR, SERIES_COLORS, SLATE,
                      NAVY, GRID, apply_mpl_theme)

# Apply Castellan styling to every figure built in this module.
apply_mpl_theme()

# Aliases used throughout the builders
PRIMARY = PRIMARY_COLOR
SECOND  = COMPARE_COLOR
ACCENTS = SERIES_COLORS[2:]
BENCHMARK_COLOR = SLATE


def _style(ax):
    ax.grid(True, alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ----------------------------------------------------------------- builders
def build_equity_curve(results, compare=None, log_scale=True, show_benchmark=True):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    runs = [(results, PRIMARY)]
    if compare is not None:
        runs.append((compare, SECOND))
    for r, color in runs:
        eq = r.equity_curve()
        if eq.empty:
            continue
        ax.plot(eq.index, eq / eq.iloc[0] * 100, label=r.name, color=color, lw=1.5)

    # Optional benchmark overlay (e.g. SPY), normalized to the same start = 100.
    # Aligned to the primary run's start so it's comparable to the equity line.
    if show_benchmark:
        bm = results.benchmark()
        eq = results.equity_curve()
        if bm is not None and not bm.empty and not eq.empty:
            bm = bm[bm.index >= eq.index[0]]
            if not bm.empty:
                ax.plot(bm.index, bm / bm.iloc[0] * 100,
                        label="Benchmark", color=BENCHMARK_COLOR,
                        lw=1.2, ls="--", alpha=0.9)

    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Growth of 100" + (" (log)" if log_scale else ""))
    ax.set_title("Equity Curve")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def build_drawdown(results, compare=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    runs = [(results, PRIMARY)]
    if compare is not None:
        runs.append((compare, SECOND))
    for r, color in runs:
        dd = r.drawdown_series() * 100
        if dd.empty:
            continue
        ax.fill_between(dd.index, dd, 0, color=color, alpha=0.4, label=r.name)
    ax.set_ylabel("Drawdown %")
    ax.set_title("Drawdowns")
    ax.legend(loc="lower left")
    _style(ax)
    fig.tight_layout()
    return fig


def build_monthly_returns_hist(results, compare=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    rm = results.returns("ME") * 100
    ax.hist(rm, bins=40, color=PRIMARY, alpha=0.7, label=results.name)
    if compare is not None:
        ax.hist(compare.returns("ME") * 100, bins=40, color=SECOND, alpha=0.5, label=compare.name)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Monthly return %")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Monthly Returns")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def build_annual_returns(results, compare=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    yr = (results.returns("YE") * 100)
    yr.index = yr.index.year
    x = np.arange(len(yr))
    width = 0.4 if compare is not None else 0.7
    ax.bar(x - (width/2 if compare is not None else 0), yr.values, width,
           color=PRIMARY, label=results.name)
    if compare is not None:
        yr2 = (compare.returns("YE") * 100)
        yr2.index = yr2.index.year
        # align years
        allyrs = sorted(set(yr.index) | set(yr2.index))
        ax.clear()
        x = np.arange(len(allyrs))
        v1 = [yr.get(y, np.nan) for y in allyrs]
        v2 = [yr2.get(y, np.nan) for y in allyrs]
        ax.bar(x - width/2, v1, width, color=PRIMARY, label=results.name)
        ax.bar(x + width/2, v2, width, color=SECOND, label=compare.name)
        ax.set_xticks(x); ax.set_xticklabels(allyrs, rotation=45, fontsize=7)
    else:
        ax.set_xticks(x); ax.set_xticklabels(yr.index, rotation=45, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Annual return %")
    ax.set_title("Annual Returns")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def build_rolling_sharpe(results, compare=None, window=12):
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, color in [(results, PRIMARY)] + ([(compare, SECOND)] if compare is not None else []):
        rm = r.returns("ME")
        roll = (rm.rolling(window).mean() * 12) / (rm.rolling(window).std() * np.sqrt(12))
        ax.plot(roll.index, roll.values, color=color, lw=1.3, label=r.name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"Rolling {window}m Sharpe")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def build_underwater(results, compare=None):
    """Time-underwater: same as drawdown but emphasizes recovery periods."""
    return build_drawdown(results, compare)


# Castellan-toned diverging colormap for the heatmap: loss -> white -> gain
_HEAT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "castellan_div", ["#B0563C", "#FFFFFF", "#5E9134"])


def build_monthly_heatmap(results, compare=None):
    """Calendar heatmap of monthly returns (years x months)."""
    import calendar
    rm = results.returns("ME") * 100
    fig, ax = plt.subplots(figsize=(10, max(2.2, 0.5 * 0 + 0.55 *
                           max(1, len(set(rm.index.year)))) + 1.2))
    if rm.empty:
        ax.text(0.5, 0.5, "Not enough data for a monthly heatmap",
                ha="center", va="center", transform=ax.transAxes, color=SLATE)
        ax.set_title("Monthly Returns (%)")
        ax.axis("off")
        fig.tight_layout()
        return fig
    df = rm.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    grid = df.pivot_table(index="year", columns="month", values="ret")
    grid = grid.reindex(columns=range(1, 13))
    years = list(grid.index)
    vals = grid.values
    bound = np.nanmax(np.abs(vals)) if np.isfinite(np.nanmax(np.abs(vals))) else 1
    im = ax.imshow(vals, aspect="auto", cmap=_HEAT_CMAP, vmin=-bound, vmax=bound)
    ax.set_xticks(range(12))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)], fontsize=8)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8)
    # annotate each cell with the value
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            if v == v:  # not NaN
                ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color="#1A3A5C")
    ax.set_title("Monthly Returns (%)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------- closed-trade analytics
def _no_trades(ax, title):
    ax.text(0.5, 0.5, "No closed-trade data in this file",
            ha="center", va="center", transform=ax.transAxes, color=SLATE)
    ax.set_title(title)
    _style(ax)


def build_trade_pnl_hist(results, compare=None):
    """Histogram of per-trade return (%)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results.closed_trades()
    if df.empty or df["return_pct"].dropna().empty:
        _no_trades(ax, "Per-Trade P&L (%)")
        fig.tight_layout(); return fig
    rets = df["return_pct"].dropna() * 100
    ax.hist(rets, bins=30, color=PRIMARY, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Trade return %")
    ax.set_ylabel("Number of trades")
    ax.set_title(f"Per-Trade P&L (%)   n={len(rets)}")
    _style(ax)
    fig.tight_layout(); return fig


def build_holding_period_hist(results, compare=None):
    """Histogram of holding period per trade (days)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results.closed_trades()
    if df.empty or df["duration_days"].dropna().empty:
        _no_trades(ax, "Holding Period per Trade")
        fig.tight_layout(); return fig
    days = df["duration_days"].dropna()
    ax.hist(days, bins=30, color=SERIES_COLORS[2], alpha=0.85)
    ax.set_xlabel("Holding period (days)")
    ax.set_ylabel("Number of trades")
    ax.set_title(f"Holding Period per Trade   median {days.median():.0f}d")
    _style(ax)
    fig.tight_layout(); return fig


def build_trades_per_month(results, compare=None):
    """Bar chart of trades opened vs closed per month."""
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results.closed_trades()
    if df.empty:
        _no_trades(ax, "Trades per Month")
        fig.tight_layout(); return fig
    opens = df["entry_time"].dropna().dt.to_period("M").value_counts()
    closes = df["exit_time"].dropna().dt.to_period("M").value_counts()
    months = sorted(set(opens.index) | set(closes.index))
    x = np.arange(len(months))
    w = 0.4
    ax.bar(x - w/2, [opens.get(m, 0) for m in months], w,
           color=PRIMARY, label="Opened")
    ax.bar(x + w/2, [closes.get(m, 0) for m in months], w,
           color=SECOND, label="Closed")
    ax.set_xticks(x)
    # Thin labels so long histories (hundreds of months) stay readable.
    max_labels = 20
    step = max(1, len(months) // max_labels)
    labels = [str(m) if i % step == 0 else "" for i, m in enumerate(months)]
    ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("Number of trades")
    ax.set_title("Trades Opened vs Closed per Month")
    ax.legend()
    _style(ax)
    fig.tight_layout(); return fig


def build_positions_over_time(results, compare=None):
    """Line (step) chart of concurrent open positions over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results.closed_trades()
    if df.empty:
        _no_trades(ax, "Open Positions Over Time")
        fig.tight_layout(); return fig
    events = []
    for _, r in df.iterrows():
        if pd.notna(r["entry_time"]):
            events.append((r["entry_time"], 1))
        if pd.notna(r["exit_time"]):
            events.append((r["exit_time"], -1))
    if not events:
        _no_trades(ax, "Open Positions Over Time")
        fig.tight_layout(); return fig
    ev = pd.DataFrame(events, columns=["t", "d"]).sort_values("t")
    ev["open"] = ev["d"].cumsum()
    ax.step(ev["t"], ev["open"], where="post", color=PRIMARY, lw=1.4)
    ax.fill_between(ev["t"], ev["open"], step="post", alpha=0.15, color=PRIMARY)
    ax.set_ylabel("Open positions")
    ax.set_title(f"Open Positions Over Time   peak {int(ev['open'].max())}")
    _style(ax)
    fig.tight_layout(); return fig


def build_top_bottom_trades(results, compare=None, n=10):
    """Two tables: top n and bottom n trades by return."""
    df = results.closed_trades()
    fig = plt.figure(figsize=(10, 7))
    if df.empty or df["return_pct"].dropna().empty:
        ax = fig.add_subplot(111)
        _no_trades(ax, "Top / Bottom Trades")
        fig.tight_layout(); return fig

    d = df.dropna(subset=["return_pct"]).copy().sort_values("return_pct", ascending=False)
    top = d.head(n)
    bottom = d.tail(n).sort_values("return_pct")

    def _rows(frame):
        out = []
        for _, r in frame.iterrows():
            out.append([
                str(r["ticker"]),
                r["entry_time"].strftime("%Y-%m-%d") if pd.notna(r["entry_time"]) else "-",
                f"{r['entry_price']:.2f}" if r["entry_price"] is not None else "-",
                r["exit_time"].strftime("%Y-%m-%d") if pd.notna(r["exit_time"]) else "-",
                f"{r['exit_price']:.2f}" if r["exit_price"] is not None else "-",
                f"{r['return_pct']*100:+.1f}%",
            ])
        return out

    cols = ["Ticker", "Open", "Open $", "Close", "Close $", "Return"]

    def _draw_table(ax, title, rows):
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, loc="left")
        tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.3)
        for (rr, cc), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            if rr == 0:
                cell.set_facecolor(NAVY)
                cell.set_text_props(color="white", fontweight="bold")

    ax1 = fig.add_subplot(211)
    _draw_table(ax1, f"Top {len(top)} Trades by Return", _rows(top))
    ax2 = fig.add_subplot(212)
    _draw_table(ax2, f"Bottom {len(bottom)} Trades by Return", _rows(bottom))
    fig.tight_layout(); return fig


# ----------------------------------------------------------------- registry
# key -> (label, builder, supports_compare)
CHART_REGISTRY = {
    "equity_curve":   ("Equity curve",            build_equity_curve,        True),
    "drawdown":       ("Drawdowns",               build_drawdown,            True),
    "annual_returns": ("Annual returns",          build_annual_returns,      True),
    "monthly_hist":   ("Monthly return histogram", build_monthly_returns_hist, True),
    "monthly_heatmap": ("Monthly returns heatmap", build_monthly_heatmap,    False),
    "rolling_sharpe": ("Rolling 12m Sharpe",      build_rolling_sharpe,      True),
    "trade_pnl_hist": ("Trade P&L histogram",     build_trade_pnl_hist,      False),
    "holding_period_hist": ("Holding period histogram", build_holding_period_hist, False),
    "trades_per_month": ("Trades per month",      build_trades_per_month,    False),
    "positions_over_time": ("Positions over time", build_positions_over_time, False),
    "top_bottom_trades": ("Top/bottom 10 trades", build_top_bottom_trades,   False),
}


def build_chart(key, results, compare=None):
    """Build a chart by registry key. Ignores `compare` if unsupported."""
    label, builder, supports_compare = CHART_REGISTRY[key]
    if supports_compare:
        return builder(results, compare=compare)
    return builder(results)