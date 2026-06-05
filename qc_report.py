"""
qc_report.py
============
Builds a PDF report from selected charts + a stats summary page.

Uses matplotlib's PdfPages so the only dependency is matplotlib (already needed
for the charts). Returns the PDF as bytes, which Streamlit can hand to a
download button directly.
"""

from __future__ import annotations
import io
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qc_charts import (build_chart, build_equity_curve, build_raw_chart,
                       CHART_REGISTRY)
from qc_brand import NAVY, GREEN, SLATE, apply_mpl_theme

apply_mpl_theme()


def _fmt(metrics):
    """Format the computed-metrics dict into (label, value) rows for the cover."""
    def pct(x):
        return f"{x*100:.2f}%" if x == x else "n/a"   # x==x guards NaN
    rows = [
        ("Start date", str(metrics.get('start'))),
        ("End date", str(metrics.get('end'))),
        ("Years", str(metrics.get('years'))),
        ("Start equity", f"${metrics.get('start_equity', 0):,.0f}"),
        ("End equity", f"${metrics.get('end_equity', 0):,.0f}"),
        ("CAGR", pct(metrics.get('cagr', float('nan')))),
        ("Annualized vol", pct(metrics.get('ann_vol', float('nan')))),
        ("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}"),
        ("Max drawdown", pct(metrics.get('max_drawdown', float('nan')))),
        ("Max DD trough", str(metrics.get('max_dd_trough'))),
        ("Recovery (days)", str(metrics.get('recovery_days'))),
        ("Beta to benchmark", f"{metrics.get('beta_to_benchmark', float('nan')):.2f}"),
        ("Corr to benchmark", f"{metrics.get('corr_to_benchmark', float('nan')):.2f}"),
    ]
    return rows


def _draw_metric_rows(fig, rows, x_label, x_value, top_y, step, fontsize=10,
                      max_value_chars=22):
    """Draw (label, value) rows down a column. Labels are left-aligned at
    x_label; values are RIGHT-aligned at x_value so they can never run off the
    page edge. Over-long values are truncated. Returns the final y reached."""
    y = top_y
    for label, value in rows:
        v = str(value)
        if len(v) > max_value_chars:
            v = v[:max_value_chars - 1] + "\u2026"
        fig.text(x_label, y, str(label), fontsize=fontsize, color=SLATE)
        fig.text(x_value, y, v, fontsize=fontsize, fontweight="bold",
                 color=NAVY, ha="right")
        y -= step
    return y


def _run_summary_page(pdf, run, page_title, subtitle=None, is_cover=False):
    """One portrait page summarizing a single run: computed metrics on the
    left, QuantConnect's full reported statistics on the right."""
    fig = plt.figure(figsize=(8.5, 11))

    if is_cover:
        # Castellan accent bar across the top
        fig.add_artist(plt.Line2D([0.1, 0.9], [0.95, 0.95], color=NAVY, lw=4,
                                  transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.1, 0.5], [0.95, 0.95], color=GREEN, lw=4,
                                  transform=fig.transFigure))
        fig.text(0.5, 0.905, page_title, ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.875, "Castellan Group", ha="center", fontsize=11,
                 color=GREEN, fontweight="bold")
        fig.text(0.5, 0.852, f"Generated {datetime.date.today().isoformat()}",
                 ha="center", fontsize=9, color=SLATE)
    else:
        fig.text(0.1, 0.92, page_title, fontsize=18, fontweight="bold", color=NAVY)
        if subtitle:
            fig.text(0.1, 0.895, subtitle, fontsize=10, color=SLATE)

    top = 0.80

    # ---- Left column: metrics we compute from the equity curve ----
    fig.text(0.1, top + 0.03, run.name + " - computed metrics",
             fontsize=12, fontweight="bold", color=NAVY)
    _draw_metric_rows(fig, _fmt(run.computed_metrics()),
                      x_label=0.11, x_value=0.47, top_y=top, step=0.034,
                      max_value_chars=20)

    # ---- Right column: QuantConnect's own reported statistics (full) ----
    fig.text(0.55, top + 0.03, "QuantConnect reported statistics",
             fontsize=12, fontweight="bold", color=NAVY)
    stats = run.statistics()
    if stats:
        stat_rows = list(stats.items())
        # tighter spacing so the full block (often ~27 rows) fits the column
        step = min(0.0275, (top - 0.06) / max(len(stat_rows), 1))
        _draw_metric_rows(fig, stat_rows,
                          x_label=0.56, x_value=0.9, top_y=top, step=step,
                          fontsize=9, max_value_chars=14)
    else:
        fig.text(0.56, top, "No statistics block found in this file.",
                 fontsize=9, color=SLATE)

    fig.text(0.1, 0.03, "Source: QuantConnect backtest results JSON. Computed "
             "metrics derived from the equity curve. For internal use.",
             fontsize=7, color=SLATE)
    pdf.savefig(fig)
    plt.close(fig)


def _cover_page(pdf, results, compare, title):
    # Primary run is the cover; a comparison run (if any) gets its own page.
    _run_summary_page(pdf, results, title, is_cover=True)
    if compare is not None:
        _run_summary_page(pdf, compare, f"Comparison: {compare.name}",
                          subtitle="Reported alongside the primary backtest.")


def generate_pdf(results, selected_keys, compare=None,
                 title="Backtest Report", include_cover=True,
                 raw_chart_names=None, show_benchmark=True) -> bytes:
    """Render selected charts (by registry key) into a PDF, return bytes.

    selected_keys:   list of CHART_REGISTRY keys, in the order to render.
    raw_chart_names: list of raw JSON chart names to append, one page each.
    show_benchmark:  overlay the benchmark on the equity curve when present.
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        if include_cover:
            _cover_page(pdf, results, compare, title)

        # Curated registry charts
        for key in selected_keys:
            if key not in CHART_REGISTRY:
                continue
            if key == "equity_curve":
                fig = build_equity_curve(results, compare=compare,
                                         show_benchmark=show_benchmark)
            else:
                fig = build_chart(key, results, compare=compare)
            # place the chart on a portrait page with a little margin
            fig.set_size_inches(8.5, 5.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Raw JSON charts, one page each
        for cname in (raw_chart_names or []):
            fig = build_raw_chart(results, cname)
            fig.set_size_inches(8.5, 5.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    buf.seek(0)
    return buf.read()