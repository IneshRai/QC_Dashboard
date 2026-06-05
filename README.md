# Castellan Backtest Dashboard

A Castellan-branded Streamlit tool for analyzing QuantConnect backtest results
JSON files. Renders the equity curve, drawdowns, and more, and exports a
branded PDF report of selected charts. Built as a reusable library so the
parsing layer can be used independently of the dashboard.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard. |
| `qc_parser.py` | Parses a QC results JSON into clean pandas objects (equity, drawdown, returns, stats). The reusable core. Raises `InvalidBacktestFile` on bad input. |
| `qc_charts.py` | Chart builders; each returns a matplotlib figure. A registry (`CHART_REGISTRY`) drives what the app and PDF offer. |
| `qc_report.py` | Builds the branded PDF (cover + selected charts). |
| `qc_brand.py` | Castellan colors, matplotlib theme, and app CSS. Single place to rebrand. |
| `.streamlit/config.toml` | Streamlit theme (Castellan navy/green). |
| `requirements.txt` | Dependencies. |
| `runtime.txt` | Python version pin for Streamlit Cloud. |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

(On Mac, if `pip`/`streamlit` aren't found, use `python3 -m pip install -r requirements.txt`
and `python3 -m streamlit run app.py`.)

Then in the browser: upload a results JSON in the sidebar (QC backtest ->
Overview -> Download Results). Optionally upload a second to compare. Tick the
charts, then click "Download PDF report".

## Deploy to Streamlit Community Cloud (shareable URL)

1. Put this folder in a **GitHub repository** (private recommended; see note below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Authorize Streamlit to access the repo (for a private repo, grant the extra
   private-repo permission when prompted).
4. Click "New app", select the repo/branch, set the main file to `app.py`,
   optionally choose a custom subdomain, and deploy.
5. The app gets a `*.streamlit.app` URL. For a private app, control who can see
   it via "Share" -> add viewer emails (they sign in with Google or an emailed
   link).

### Important for an RIA
Streamlit Community Cloud is a third-party host. Before deploying anything that
touches real client or live-strategy data, confirm with whoever handles
compliance/IT at the firm where that data is allowed to live. Deploying from a
**private** repo keeps the app private to the workspace by default, and viewer
access is explicit-grant only -- but it is still external hosting. For fully
internal hosting, the same code runs under `streamlit run app.py` on an internal
server or in a container.

## Use the parser as a library (no dashboard)

```python
from qc_parser import load_results
r = load_results("backtest.json", name="My run")
eq = r.equity_curve()          # pd.Series
dd = r.drawdown_series()       # pd.Series, <= 0
metrics = r.computed_metrics() # dict: cagr, sharpe, max_drawdown, beta, ...
```

## Add a new chart (main extension point)

1. In `qc_charts.py`, write `build_<name>(results, compare=None) -> Figure`.
2. Register it: `CHART_REGISTRY["<key>"] = ("Label", build_<name>, True)`.

The checkbox and PDF option appear automatically. New data accessors go in
`qc_parser.py`; any raw chart series is already reachable via
`results.chart_series(name)`.
