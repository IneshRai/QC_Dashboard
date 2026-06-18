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
                      NAVY, GREEN, GREEN_DARK, GRID, apply_mpl_theme)

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
        # Format log ticks as plain numbers (100, 1000) rather than mathtext
        # exponents (10^2). Some matplotlib builds fail to parse the mathtext
        # tick labels, so avoiding mathtext entirely keeps this robust across
        # Python/matplotlib versions.
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        sf = ScalarFormatter()
        sf.set_scientific(False)
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.set_minor_formatter(NullFormatter())
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


def build_annual_returns(results, compare=None, show_benchmark=True):
    """Grouped annual-return bars: strategy, optional comparison run, and the
    benchmark side-by-side for each year (benchmark shown when present).

    Bars are grouped per calendar year so strategy vs benchmark is easy to read.
    Years are computed the same way for every series (year-end resample), so the
    comparison is apples-to-apples.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Assemble the series we actually have, each as {year: return%}.
    groups = [(results.name, results.returns("YE") * 100, PRIMARY)]
    if compare is not None:
        groups.append((compare.name, compare.returns("YE") * 100, SECOND))
    if show_benchmark:
        bm = results.benchmark_returns("YE") * 100
        if not bm.empty:
            groups.append(("Benchmark", bm, BENCHMARK_COLOR))

    # Re-key each series by integer year.
    keyed = []
    for label, s, color in groups:
        s = s.copy()
        if not s.empty:
            s.index = s.index.year
        keyed.append((label, s, color))

    all_years = sorted(set().union(*[set(s.index) for _, s, _ in keyed if not s.empty])) \
        if any(not s.empty for _, s, _ in keyed) else []
    x = np.arange(len(all_years))

    n = max(1, len(keyed))
    total_width = 0.8
    bw = total_width / n
    for i, (label, s, color) in enumerate(keyed):
        offset = (i - (n - 1) / 2) * bw
        vals = [s.get(y, np.nan) for y in all_years]
        ax.bar(x + offset, vals, bw, color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(all_years, rotation=45, fontsize=7)
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
    rm = results.returns("ME") * 100
    return _heatmap_figure(rm, "Monthly Returns (%)")


def build_monthly_excess_heatmap(results, compare=None):
    """Calendar heatmap of monthly EXCESS returns: strategy% - benchmark%.

    Aligns each month's strategy return against the benchmark's return for the
    same month and plots the difference, so green = outperformed the benchmark
    that month, red = lagged it. Needs a benchmark series in the file.
    """
    s = results.returns("ME")
    b = results.benchmark_returns("ME")
    if b.empty:
        return _heatmap_figure(pd.Series(dtype=float),
                               "Monthly Excess Returns vs Benchmark (%)",
                               empty_msg="No benchmark series in this file, so "
                                         "excess returns can't be computed.")
    # Align on shared month-ends; subtraction lines up on the index.
    excess = (s - b).dropna() * 100
    return _heatmap_figure(excess, "Monthly Excess Returns vs Benchmark (%)")


def _heatmap_figure(rm_pct, title, empty_msg="Not enough data for a monthly heatmap"):
    """Build a year x month heatmap figure from a month-end %-returns Series.

    Shared by the plain monthly heatmap and the excess-vs-benchmark version so
    the styling (Castellan diverging colormap, cell annotations, sizing) stays
    in one place.
    """
    import calendar
    n_years = max(1, len(set(rm_pct.index.year))) if not rm_pct.empty else 1
    fig, ax = plt.subplots(figsize=(10, max(2.2, 0.55 * n_years) + 1.2))
    if rm_pct.empty:
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center",
                transform=ax.transAxes, color=SLATE)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        return fig
    df = rm_pct.to_frame("ret")
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
    ax.set_title(title)
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
    """Histogram of per-trade return (%).

    A handful of extreme outliers (e.g. a +900% trade) otherwise stretch the
    x-axis to several hundred percent, collapsing the bulk of trades into one
    or two bars -- and the outlier bars themselves are only ~1 trade tall, so
    they're invisible against a y-axis in the thousands. We focus the view on
    the 1st-99th percentile range with fine bins, and FOLD outliers into
    clearly-labelled overflow bins at each edge so no trade is dropped.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results.closed_trades()
    if df.empty or df["return_pct"].dropna().empty:
        _no_trades(ax, "Per-Trade P&L (%)")
        fig.tight_layout(); return fig
    rets = df["return_pct"].dropna() * 100

    # Robust display window so a few outliers don't dominate the axis.
    lo, hi = np.percentile(rets, [1, 99])
    if hi <= lo:                       # all returns ~equal: fall back to full range
        lo, hi = rets.min(), rets.max()
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad

    n_below = int((rets < lo).sum())
    n_above = int((rets > hi).sum())
    # Fold outliers into the edge bins so every trade is still counted/visible.
    clipped = rets.clip(lo, hi)
    ax.hist(clipped, bins=60, range=(lo, hi), color=PRIMARY, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Trade return %")
    ax.set_ylabel("Number of trades")
    ax.set_title(f"Per-Trade P&L (%)   n={len(rets)}")

    # Mean and median markers (bold, labelled verticals). Computed on the FULL
    # set of returns, not the clipped view, so outliers still pull them
    # honestly. Markers are clamped into the display window if they fall outside.
    mean_v = float(rets.mean())
    median_v = float(rets.median())
    _mark_mean_median(ax, mean_v, median_v,
                      f"Mean {mean_v:+.1f}%", f"Median {median_v:+.1f}%",
                      clamp=(lo, hi))

    # Annotate the overflow bins so the folded edge bars aren't misread as
    # genuine counts at exactly lo/hi.
    ymax = ax.get_ylim()[1]
    if n_above:
        ax.annotate(f"{n_above} trades > {hi:.0f}%\n(max {rets.max():.0f}%)",
                    xy=(hi, 0), xytext=(hi, ymax * 0.6),
                    ha="right", va="center", fontsize=8, color=SLATE,
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))
    if n_below:
        ax.annotate(f"{n_below} trades < {lo:.0f}%\n(min {rets.min():.0f}%)",
                    xy=(lo, 0), xytext=(lo, ymax * 0.6),
                    ha="left", va="center", fontsize=8, color=SLATE,
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))

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
    """Daily count of concurrent open positions over time.

    Each trade contributes a +1 at entry and a -1 at exit. Naively cumulating
    these event-by-event produces deep false 'spikes' on rebalance days, when
    many positions close and reopen at the same instant: the running total
    dives on the closes and leaps back on the opens, an artifact of plotting
    order rather than a real swing. To avoid that we (1) net all events that
    share a timestamp into a single change, then (2) report the END-OF-DAY
    count, which is the meaningful 'how many were held that day' figure.
    """
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

    ev = pd.DataFrame(events, columns=["t", "d"])
    # (1) net simultaneous opens/closes so same-instant churn can't create a
    #     spurious down-then-up spike.
    netted = ev.groupby("t", as_index=False)["d"].sum().sort_values("t")
    netted["open"] = netted["d"].cumsum()
    # (2) collapse to one reading per day (end-of-day level), forward-filling
    #     days with no events so the line stays continuous.
    s = netted.set_index("t")["open"]
    daily = s.resample("D").last().ffill()

    ax.step(daily.index, daily.values, where="post", color=PRIMARY, lw=1.4)
    ax.fill_between(daily.index, daily.values, step="post", alpha=0.15,
                    color=PRIMARY)
    ax.set_ylabel("Open positions (end of day)")
    ax.set_title(f"Open Positions Over Time   peak {int(daily.max())}")
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


# ------------------------------------------------- portfolio composition
def _empty_panel(ax, title, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=SLATE, wrap=True, fontsize=10)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _mark_mean_median(ax, mean_v, median_v, mean_txt, median_txt, clamp=None):
    """Draw bold, clearly-labelled mean & median vertical lines on a histogram.

    Lines use contrasting colors that read against both navy bars and green
    fills, full opacity, and a boxed label near the top so the markers don't get
    lost. Values are clamped into `clamp` (lo, hi) for drawing but the label
    still shows the true number.
    """
    trans = ax.get_xaxis_transform()      # x in data coords, y in axes coords
    lo, hi = (clamp if clamp else (None, None))

    def _x(v):
        if clamp:
            return min(max(v, lo), hi)
        return v

    for value, color, dashes, yfrac, txt in (
        (mean_v, "#B0563C", (6, 3), 0.95, mean_txt),
        (median_v, GREEN, (2, 2), 0.82, median_txt),
    ):
        x = _x(value)
        ax.axvline(x, color=color, lw=2.6, dashes=dashes, alpha=0.95, zorder=6)
        # Put the label on whichever side has room.
        right_edge = hi if clamp else ax.get_xlim()[1]
        left_edge = lo if clamp else ax.get_xlim()[0]
        near_right = (x - left_edge) > 0.7 * (right_edge - left_edge)
        ha = "right" if near_right else "left"
        pad = " " if ha == "left" else "  "
        ax.text(x, yfrac, f"{pad}{txt}", transform=trans, color="white",
                fontsize=9, fontweight="bold", ha=ha, va="top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.28", fc=color, ec="none",
                          alpha=0.95))


def _padded_ylim(arrays, must_include=()):
    """Nice padded y-limits covering the data plus any reference values that
    must be on-screen (e.g. the 100% line, or 0 for cash)."""
    vals = np.concatenate([np.asarray(a, dtype=float) for a in arrays]) \
        if arrays else np.array([0.0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0])
    lo, hi = float(vals.min()), float(vals.max())
    for v in must_include:
        lo, hi = min(lo, v), max(hi, v)
    pad = max(2.0, (hi - lo) * 0.10)
    return lo - pad, hi + pad


def build_invested_level(results, compare=None):
    """How much of the portfolio is in equity vs cash over time.

    For a normal long book this is a clean stacked composition (an equity band
    and a cash band) zoomed to where the data actually sits, with the average
    invested level called out -- so the day-to-day variation and any de-risking
    are visible instead of being crushed into a flat 0-100% wall.

    If the book ever shorts or trades on margin (so cash can go negative or the
    parts no longer sum tidily to 100%), it switches to an analytical view:
    net/gross exposure on top and signed cash/dry-powder below. Built from QC's
    'Exposure' chart.
    """
    exp = results.exposure()
    if exp.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _empty_panel(ax, "Invested Level (Cash vs Equity)",
                     "No 'Exposure' chart in this file. Re-run the backtest on a "
                     "recent LEAN/QuantConnect version to record exposure.")
        fig.tight_layout(); return fig

    short_pct = exp["short"] * 100
    gross_pct = exp["gross"] * 100
    has_short = bool((short_pct < -0.05).any())
    has_margin = bool((gross_pct > 100.5).any())

    if has_short or has_margin:
        return _invested_analytical(exp)
    return _invested_composition(exp)


def _invested_composition(exp):
    """Clean cash-vs-equity composition for a long, un-levered book."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    net_all = (exp["net"] * 100).clip(lower=0)          # equity invested

    # On a LONG backtest, trim the leading/trailing days where the book isn't
    # really deployed yet (a startup ramp at ~0%), so one 0% day doesn't force
    # the y-axis to span 0-100% and crush the data. On a SHORT run (a few weeks)
    # the early cash period is often the whole point, so keep every day.
    TRIM_MIN_DAYS = 90
    span_days = (net_all.index[-1] - net_all.index[0]).days if len(net_all) > 1 else 0
    if span_days >= TRIM_MIN_DAYS:
        med = float(net_all.median())
        active = net_all > max(5.0, 0.5 * med)
        net_pct = net_all.loc[active.idxmax():active[::-1].idxmax()] \
            if active.any() else net_all
    else:
        net_pct = net_all
    idx = net_pct.index
    cash_pct = 100 - net_pct                            # dry powder, >= 0 here
    avg_net = float(net_pct.mean())
    avg_cash = float(cash_pct.mean())

    # Two stacked bands: equity (0 -> net) and cash (net -> 100).
    ax.fill_between(idx, 0, net_pct, color=GREEN, alpha=0.55, lw=0)
    ax.fill_between(idx, net_pct, 100, color=SLATE, alpha=0.22, lw=0)
    # Boundary between the two bands, plus a smoothed average-invested trend.
    ax.plot(idx, net_pct, color=GREEN_DARK, lw=0.8, alpha=0.7)
    roll = net_pct.rolling(21, min_periods=1).mean()
    ax.plot(idx, roll, color=NAVY, lw=1.8, label="Invested (21d avg)")
    ax.axhline(100, color=SLATE, lw=1.0, ls=":", alpha=0.7)
    ax.axhline(avg_net, color=NAVY, lw=1.0, ls="--", alpha=0.6)

    # Auto-zoom to where the data lives. Use a low percentile (not the raw min)
    # for the floor so the occasional deep de-risk day doesn't reset the scale,
    # but never hide the genuine minimum: floor just below whichever is lower.
    p_lo = float(np.percentile(net_pct, 1))
    floor = max(0.0, min(p_lo, float(net_pct.min())) - 4)
    # Guard against a degenerate all-flat series (floor == ceiling).
    if floor >= 99:
        floor = max(0.0, avg_net - 6)
    ax.set_ylim(floor, 101.5)

    # Label the bands directly instead of a legend swatch.
    ax.text(0.012, 0.06, f"Equity invested — avg {avg_net:.1f}%",
            transform=ax.transAxes, color="white", fontsize=9.5,
            fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc=GREEN_DARK, ec="none", alpha=0.9))
    ax.text(0.012, 0.93, f"Cash — avg {avg_cash:.1f}%",
            transform=ax.transAxes, color=NAVY, fontsize=9.5, fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SLATE, alpha=0.9))

    ax.set_ylabel("% of portfolio value")
    ax.set_title(f"Invested Level (Cash vs Equity)   "
                 f"avg {avg_net:.0f}% invested / {avg_cash:.0f}% cash")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85, facecolor="white")
    _style(ax)
    fig.tight_layout(); return fig

    # Label the bands directly instead of a legend swatch.
    ax.text(0.012, 0.06, f"Equity invested — avg {avg_net:.1f}%",
            transform=ax.transAxes, color="white", fontsize=9.5,
            fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc=GREEN_DARK, ec="none", alpha=0.9))
    ax.text(0.012, 0.93, f"Cash — avg {avg_cash:.1f}%",
            transform=ax.transAxes, color=NAVY, fontsize=9.5, fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SLATE, alpha=0.9))

    ax.set_ylabel("% of portfolio value")
    ax.set_title(f"Invested Level (Cash vs Equity)   "
                 f"avg {avg_net:.0f}% invested / {avg_cash:.0f}% cash")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85, facecolor="white")
    _style(ax)
    fig.tight_layout(); return fig


def _invested_analytical(exp):
    """Exposure + cash view for books that short or use margin (cash can go
    negative; parts don't sum tidily to 100%)."""
    idx = exp.index
    long_pct = exp["long"] * 100
    short_pct = exp["short"] * 100
    net_pct = exp["net"] * 100
    gross_pct = exp["gross"] * 100
    cash_pct = exp["cash"] * 100
    has_short = bool((short_pct < -0.05).any())

    fig, (ax_exp, ax_cash) = plt.subplots(
        2, 1, figsize=(10, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    # ---- Top: exposure ----
    ax_exp.fill_between(idx, 0, net_pct, color=GREEN, alpha=0.18, lw=0)
    ax_exp.plot(idx, net_pct, color=NAVY, lw=1.4, label="Net exposure")
    ax_exp.plot(idx, gross_pct, color=ACCENTS[0], lw=1.2, label="Gross exposure")
    if has_short:
        ax_exp.plot(idx, long_pct, color=GREEN_DARK, lw=1.0, alpha=0.8,
                    label="Long")
        ax_exp.plot(idx, short_pct, color="#B0563C", lw=1.0, alpha=0.8,
                    label="Short")
    ax_exp.axhline(100, color=SLATE, lw=1.0, ls=":", alpha=0.7)
    ax_exp.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax_exp.set_ylim(*_padded_ylim(
        [net_pct, gross_pct] + ([short_pct] if has_short else []),
        must_include=(100, 0)))
    ax_exp.set_ylabel("Exposure % of PV")
    avg_net = float(net_pct.mean()); avg_gross = float(gross_pct.mean())
    avg_cash = float(cash_pct.mean())
    ax_exp.set_title(f"Invested Level (Cash vs Equity)   "
                     f"avg net {avg_net:.0f}% / gross {avg_gross:.0f}% / "
                     f"cash {avg_cash:.0f}%")
    ax_exp.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85,
                  facecolor="white")
    _style(ax_exp)

    # ---- Bottom: cash / dry powder ----
    ax_cash.fill_between(idx, 0, cash_pct, color=SLATE, alpha=0.30, lw=0)
    ax_cash.plot(idx, cash_pct, color=SLATE, lw=1.2)
    ax_cash.axhline(0, color="k", lw=0.8)
    ax_cash.set_ylim(*_padded_ylim([cash_pct], must_include=(0,)))
    ax_cash.set_ylabel("Cash % of PV")
    _style(ax_cash)

    fig.tight_layout(); return fig


# ------------------------------------------------- order execution timing
def _pick_time_unit(seconds_ref):
    """Choose a human time unit for a representative latency (in seconds).
    Returns (divisor_to_seconds, unit_label)."""
    if seconds_ref < 90:
        return 1.0, "seconds"
    if seconds_ref < 90 * 60:
        return 60.0, "minutes"
    if seconds_ref < 48 * 3600:
        return 3600.0, "hours"
    return 86400.0, "days"


def build_execution_time_hist(results, compare=None):
    """Distribution of order execution time: the gap between when an order is
    submitted and when it is filled.

    The x-axis unit auto-adjusts (seconds / minutes / hours / days) based on the
    bulk of the data, so it reads well whether fills are near-instant market
    orders or limit orders that rest for days. Mean and median are marked.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    lat = results.execution_latencies()        # seconds, filled orders only
    if lat.empty:
        _empty_panel(ax, "Order Execution Time",
                     "No order fill timing in this file (no orders, or no "
                     "fill timestamps recorded).")
        fig.tight_layout(); return fig

    # When (almost) every order fills in the same instant, a continuous
    # histogram is just one spike. Draw a bucketed bar chart instead so there's
    # always a real graph, with the count in each latency band.
    if float(np.percentile(lat, 99)) <= 0:
        _execution_buckets(ax, lat)
        fig.tight_layout(); return fig

    # Pick a unit from the 95th percentile so a few long-resting orders don't
    # force an awkward scale on the bulk of quick fills.
    ref = float(np.percentile(lat, 95)) or float(lat.max())
    div, unit = _pick_time_unit(ref)
    vals = lat / div

    # Robust display window with folded overflow, mirroring the P&L histogram.
    hi = float(np.percentile(vals, 99))
    if hi <= 0:
        hi = float(vals.max())
    n_above = int((vals > hi).sum())
    clipped = vals.clip(0, hi)
    ax.hist(clipped, bins=40, range=(0, hi), color=PRIMARY, alpha=0.85)

    mean_v = float(vals.mean())
    median_v = float(vals.median())
    _mark_mean_median(ax, mean_v, median_v,
                      f"Mean {mean_v:.2f} {unit}", f"Median {median_v:.2f} {unit}",
                      clamp=(0, hi))

    ax.set_xlim(0, hi)
    ax.set_xlabel(f"Execution time ({unit})")
    ax.set_ylabel("Number of orders")
    ax.set_title(f"Order Execution Time   n={len(lat)}")
    if n_above:
        ymax = ax.get_ylim()[1]
        ax.annotate(f"{n_above} orders > {hi:.1f} {unit}\n(max {vals.max():.1f})",
                    xy=(hi, 0), xytext=(hi, ymax * 0.6),
                    ha="right", va="center", fontsize=8, color=SLATE,
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))
    _style(ax)
    fig.tight_layout(); return fig


# Latency buckets (seconds) for the degenerate / near-instant case.
_LAT_BUCKETS = [
    ("0s\n(same bar)", lambda s: s == 0),
    ("≤1 min",         lambda s: (s > 0) & (s <= 60)),
    ("1–60 min",       lambda s: (s > 60) & (s <= 3600)),
    ("1–24 h",         lambda s: (s > 3600) & (s <= 86400)),
    (">1 day",         lambda s: s > 86400),
]


def _execution_buckets(ax, lat):
    """Bar chart of order count per latency band. Always renders something, even
    when every order fills instantly (one full '0s' bar)."""
    counts = [int(cond(lat).sum()) for _, cond in _LAT_BUCKETS]
    labels = [lbl for lbl, _ in _LAT_BUCKETS]
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, width=0.62, color=PRIMARY, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Number of orders")
    n = len(lat)
    imm = counts[0]
    ax.set_ylim(0, max(counts + [1]) * 1.18)
    for b, c in zip(bars, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=NAVY)
    pct = (imm / n * 100) if n else 0
    ax.set_title(f"Order Execution Time   n={n:,}   "
                 f"{pct:.0f}% filled same bar (0s latency)")
    _style(ax)


# ----------------------------------------------------------------- registry
# key -> (label, builder, supports_compare)
CHART_REGISTRY = {
    "equity_curve":   ("Equity curve",            build_equity_curve,        True),
    "drawdown":       ("Drawdowns",               build_drawdown,            True),
    "annual_returns": ("Annual returns",          build_annual_returns,      True),
    "monthly_hist":   ("Monthly return histogram", build_monthly_returns_hist, True),
    "monthly_heatmap": ("Monthly returns heatmap", build_monthly_heatmap,    False),
    "monthly_excess_heatmap": ("Monthly excess returns heatmap (vs benchmark)",
                               build_monthly_excess_heatmap,                 False),
    "rolling_sharpe": ("Rolling 12m Sharpe",      build_rolling_sharpe,      True),
    "invested_level": ("Invested level (cash vs equity)", build_invested_level, False),
    "trade_pnl_hist": ("Trade P&L histogram",     build_trade_pnl_hist,      False),
    "execution_time_hist": ("Order execution time", build_execution_time_hist, False),
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