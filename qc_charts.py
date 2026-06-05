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
import numpy as np

from qc_brand import (PRIMARY_COLOR, COMPARE_COLOR, SERIES_COLORS, SLATE,
                      apply_mpl_theme)

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
                        label=f"Benchmark", color=BENCHMARK_COLOR,
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


def build_raw_chart(results, chart_name):
    """Plot any raw chart from the JSON by name (all of its series).

    Used by the 'raw chart' checkboxes so arbitrary QC charts (Exposure,
    Portfolio Turnover, Benchmark, etc.) can be dropped into the PDF as-is.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    series = results.chart_series(chart_name)
    plotted = False
    for i, (sname, s) in enumerate(series.items()):
        if s is None or s.empty:
            continue
        ax.plot(s.index, s.values, label=sname, lw=1.3,
                color=SERIES_COLORS[i % len(SERIES_COLORS)])
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No plottable data in this chart",
                ha="center", va="center", transform=ax.transAxes, color=SLATE)
    else:
        ax.legend()
    ax.set_title(chart_name)
    _style(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- registry
# key -> (label, builder, supports_compare)
CHART_REGISTRY = {
    "equity_curve":   ("Equity curve",            build_equity_curve,        True),
    "drawdown":       ("Drawdowns",               build_drawdown,            True),
    "annual_returns": ("Annual returns",          build_annual_returns,      True),
    "monthly_hist":   ("Monthly return histogram", build_monthly_returns_hist, True),
    "rolling_sharpe": ("Rolling 12m Sharpe",      build_rolling_sharpe,      True),
}


def build_chart(key, results, compare=None):
    """Build a chart by registry key. Ignores `compare` if unsupported."""
    label, builder, supports_compare = CHART_REGISTRY[key]
    if supports_compare:
        return builder(results, compare=compare)
    return builder(results)