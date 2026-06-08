"""
app.py  --  Castellan Backtest Dashboard
========================================
Streamlit app to analyze QuantConnect backtest results JSON files: renders the
equity curve, drawdowns, and more, and exports a branded PDF report of selected
charts.

Run locally:
    streamlit run app.py

Upload one results JSON to analyze, or two to compare (e.g. before/after a fix).
"""

import streamlit as st

from qc_parser import load_results, InvalidBacktestFile
from qc_charts import CHART_REGISTRY, build_chart, build_equity_curve
from qc_report import generate_pdf
from qc_quantstats import build_quantstats_html, QuantStatsError
from qc_brand import streamlit_css, NAVY

st.set_page_config(page_title="Castellan Backtest Dashboard",
                   page_icon="🌿", layout="wide")

# ---- Castellan branding ----
st.markdown(streamlit_css(), unsafe_allow_html=True)
st.markdown(f"<h1 style='margin-bottom:0'>Castellan Backtest Dashboard</h1>",
            unsafe_allow_html=True)
st.markdown("<hr class='castellan-rule'>", unsafe_allow_html=True)
st.markdown("<p class='castellan-caption'>Analyze QuantConnect backtest "
            "results. Upload the JSON from the backtest Overview tab "
            "(Download Results).</p>", unsafe_allow_html=True)


def safe_load(uploaded, label):
    """Load an uploaded file, showing a clean error instead of a stack trace."""
    if uploaded is None:
        return None
    try:
        return load_results(uploaded, name=label)
    except InvalidBacktestFile as e:
        st.error(f"Could not load '{getattr(uploaded, 'name', label)}': {e}")
        return None


# ---------------- Sidebar: uploads + options ----------------
with st.sidebar:
    st.header("Inputs")
    primary_file = st.file_uploader("Primary backtest JSON", type="json", key="primary")
    compare_file = st.file_uploader("Comparison JSON (optional)", type="json", key="compare")
    primary_label = st.text_input("Primary label", value="Backtest")
    compare_label = st.text_input("Comparison label", value="Comparison")
    log_scale = st.checkbox("Log scale on equity curve", value=True)

if primary_file is None:
    st.info("Upload a results JSON in the sidebar to begin.")
    st.stop()

results = safe_load(primary_file, primary_label)
if results is None:
    st.stop()
compare = safe_load(compare_file, compare_label)

# Whether a benchmark series is available to overlay on the equity curve.
_bm = results.benchmark()
has_benchmark = _bm is not None and not _bm.empty

# ---------------- Headline metrics ----------------
m = results.computed_metrics()
st.subheader("Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAGR", f"{m.get('cagr', float('nan'))*100:.1f}%")
c2.metric("Max drawdown", f"{m.get('max_drawdown', float('nan'))*100:.1f}%")
c3.metric("Sharpe", f"{m.get('sharpe', float('nan')):.2f}")
c4.metric("Ann. vol", f"{m.get('ann_vol', float('nan'))*100:.1f}%")
c5.metric("End equity", f"${m.get('end_equity', 0):,.0f}")
if compare is not None:
    mc = compare.computed_metrics()
    st.caption(
        f"Comparison ({compare.name}): CAGR {mc.get('cagr',0)*100:.1f}%, "
        f"Max DD {mc.get('max_drawdown',0)*100:.1f}%, "
        f"Sharpe {mc.get('sharpe',float('nan')):.2f}"
    )

# ---------------- Chart selection ----------------
st.subheader("Charts")
st.write("Select which charts to display and include in the PDF:")

selected = {}
items = list(CHART_REGISTRY.items())
per_row = 3
for start in range(0, len(items), per_row):
    cols = st.columns(per_row)
    for (key, (label, _builder, _cmp)), col in zip(items[start:start + per_row], cols):
        default_on = key in ("equity_curve", "drawdown")
        selected[key] = col.checkbox(label, value=default_on, key=f"sel_{key}")

# Benchmark toggle (on by default) sits right with the chart controls.
# Only meaningful when a Benchmark series exists in the file.
if has_benchmark:
    show_benchmark = st.checkbox(
        "Show benchmark on equity curve", value=True, key="show_benchmark")
else:
    show_benchmark = False
    st.caption("No benchmark included (no 'Benchmark' series in this file). "
               "Add self.SetBenchmark(\"SPY\") in your algorithm to include one.")

# ---------------- Render selected charts ----------------
for key, (label, _b, _c) in CHART_REGISTRY.items():
    if not selected.get(key):
        continue
    if key == "equity_curve":
        fig = build_equity_curve(results, compare=compare, log_scale=log_scale,
                                 show_benchmark=show_benchmark)
    else:
        fig = build_chart(key, results, compare=compare)
    st.pyplot(fig)

# ---------------- PDF export ----------------
st.subheader("Export")
chosen = [k for k, v in selected.items() if v]
report_title = st.text_input("Report title", value="Castellan Backtest Report")
pdf_filename = st.text_input("PDF file name", value="castellan_backtest_report")
include_cover = st.checkbox("Include cover page with stats", value=True)

if not chosen:
    st.warning("Select at least one chart to enable PDF export.")
else:
    # Build the PDF only when the user clicks Generate, using the CURRENT
    # options. This makes the cover/benchmark/chart choices take effect
    # predictably instead of rebuilding on every keystroke.
    if st.button("Generate PDF report"):
        safe_name = (pdf_filename or "castellan_backtest_report").strip()
        if safe_name.lower().endswith(".pdf"):
            safe_name = safe_name[:-4]
        safe_name = safe_name or "castellan_backtest_report"
        st.session_state["pdf_bytes"] = generate_pdf(
            results, chosen, compare=compare,
            title=report_title, include_cover=include_cover,
            show_benchmark=show_benchmark,
        )
        st.session_state["pdf_name"] = f"{safe_name}.pdf"

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_name", "castellan_backtest_report.pdf"),
            mime="application/pdf",
        )
        st.caption("Change any option above and click Generate again to refresh.")

# ---------------- QuantStats tearsheet (separate HTML) ----------------
st.subheader("QuantStats tearsheet")
st.write("Generate a full QuantStats tearsheet as a separate HTML file "
         "(opens in any browser).")

qs_bm_mode = st.radio(
    "Benchmark for the tearsheet",
    ["Backtest's own benchmark", "Fetch a ticker (yfinance)"],
    horizontal=True, key="qs_bm_mode",
)
qs_ticker = None
if qs_bm_mode == "Fetch a ticker (yfinance)":
    qs_ticker = st.text_input("Benchmark ticker", value="SPY", key="qs_ticker")
elif not has_benchmark:
    st.caption("This file has no benchmark series, so the tearsheet will have "
               "no benchmark unless you fetch a ticker.")
qs_title = st.text_input("Tearsheet title", value="Castellan Strategy Tearsheet",
                         key="qs_title")

if st.button("Generate QuantStats tearsheet"):
    with st.spinner("Building QuantStats tearsheet..."):
        try:
            st.session_state["qs_html"] = build_quantstats_html(
                results, benchmark_ticker=(qs_ticker or None), title=qs_title)
        except QuantStatsError as e:
            st.session_state.pop("qs_html", None)
            st.error(str(e))

if st.session_state.get("qs_html"):
    st.download_button(
        "Download QuantStats tearsheet (HTML)",
        data=st.session_state["qs_html"],
        file_name="quantstats_tearsheet.html",
        mime="text/html",
    )
    st.caption("Open it in a browser; for a PDF use the browser's "
               "Print \u2192 Save as PDF.")

# ---------------- QC's own statistics ----------------
with st.expander("QuantConnect's reported statistics"):
    stats = results.statistics()
    if stats:
        st.dataframe(
            {"Metric": list(stats.keys()), "Value": list(stats.values())},
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No statistics block found in this file.")