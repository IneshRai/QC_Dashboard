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

from qc_charts import build_chart, CHART_REGISTRY
from qc_brand import NAVY, GREEN, SLATE, apply_mpl_theme

apply_mpl_theme()


def _fmt(metrics):
    """Format the computed-metrics dict into (label, value) rows for the cover."""
    def pct(x):
        return f"{x*100:.2f}%" if x == x else "n/a"   # x==x guards NaN
    rows = [
        ("Period", f"{metrics.get('start')} to {metrics.get('end')} ({metrics.get('years')} yrs)"),
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


def _cover_page(pdf, results, compare, title):
    fig = plt.figure(figsize=(8.5, 11))
    # Castellan accent bar across the top
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.95, 0.95], color=NAVY, lw=4,
                              transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.1, 0.5], [0.95, 0.95], color=GREEN, lw=4,
                              transform=fig.transFigure))
    fig.text(0.5, 0.90, title, ha="center", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.865, "Castellan Group", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    fig.text(0.5, 0.84, f"Generated {datetime.date.today().isoformat()}",
             ha="center", fontsize=9, color=SLATE)

    # primary stats table
    fig.text(0.1, 0.78, results.name, fontsize=13, fontweight="bold", color=NAVY)
    y = 0.74
    for label, value in _fmt(results.computed_metrics()):
        fig.text(0.12, y, label, fontsize=10, color=SLATE)
        fig.text(0.55, y, value, fontsize=10, fontweight="bold", color=NAVY)
        y -= 0.034

    # comparison stats table
    if compare is not None:
        fig.text(0.1, 0.36, compare.name, fontsize=13, fontweight="bold", color=NAVY)
        y = 0.32
        for label, value in _fmt(compare.computed_metrics()):
            fig.text(0.12, y, label, fontsize=10, color=SLATE)
            fig.text(0.55, y, value, fontsize=10, fontweight="bold", color=NAVY)
            y -= 0.034

    fig.text(0.1, 0.03, "Source: QuantConnect backtest results JSON. Metrics "
             "computed from the equity curve. For internal use.", fontsize=7, color=SLATE)
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf(results, selected_keys, compare=None,
                 title="Backtest Report", include_cover=True) -> bytes:
    """Render selected charts (by registry key) into a PDF, return bytes.

    selected_keys: list of CHART_REGISTRY keys, in the order to render.
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        if include_cover:
            _cover_page(pdf, results, compare, title)
        for key in selected_keys:
            if key not in CHART_REGISTRY:
                continue
            fig = build_chart(key, results, compare=compare)
            # place the chart on a portrait page with a little margin
            fig.set_size_inches(8.5, 5.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    buf.seek(0)
    return buf.read()
