"""
qc_quantstats.py
================
Optional QuantStats tearsheet generation, kept separate from the main report.

QuantStats produces a self-contained HTML tearsheet from a series of daily
returns plus an optional benchmark. We feed it the strategy's daily returns
(from the parser) and, for the benchmark, either:
  * the backtest's OWN benchmark (the 'Benchmark' series QC recorded), or
  * a ticker (e.g. "SPY") fetched live via yfinance.

The tearsheet comes out as HTML bytes for a download button -- it is NOT part
of the branded PDF, by design (QuantStats has its own look).
"""

from __future__ import annotations
import os
import tempfile
import warnings
import pandas as pd


class QuantStatsError(Exception):
    """Raised when a tearsheet can't be generated (bad data, fetch failure)."""


def _strategy_returns(results) -> pd.Series:
    """Daily strategy returns as a tz-naive, datetime-indexed Series."""
    r = results.returns("D")
    if r is None or r.empty:
        raise QuantStatsError("No daily returns available from this backtest.")
    r = r.copy()
    r.index = pd.to_datetime(r.index)
    if getattr(r.index, "tz", None) is not None:
        r.index = r.index.tz_localize(None)
    r.name = "Strategy"
    return r


def _backtest_benchmark_returns(results):
    """Daily returns of the backtest's own benchmark series, or None."""
    bm = results.benchmark()
    if bm is None or bm.empty:
        return None
    ret = bm.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    if getattr(ret.index, "tz", None) is not None:
        ret.index = ret.index.tz_localize(None)
    ret.name = "Benchmark"
    return ret


def _fetch_ticker_returns(ticker: str, start, end):
    """Daily returns for a ticker via yfinance, aligned to [start, end]."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise QuantStatsError(
            "yfinance is not installed, so a benchmark ticker can't be "
            "fetched. Add 'yfinance' to requirements.txt."
        ) from e
    try:
        df = yf.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=True)
    except Exception as e:  # network / rate-limit / bad ticker
        raise QuantStatsError(f"Could not fetch '{ticker}' via yfinance: {e}") from e
    if df is None or df.empty:
        raise QuantStatsError(
            f"yfinance returned no data for '{ticker}'. Check the symbol "
            "or try again (Yahoo can rate-limit)."
        )
    close = df["Close"]
    if isinstance(close, pd.DataFrame):      # yfinance can return a frame
        close = close.iloc[:, 0]
    ret = close.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    if getattr(ret.index, "tz", None) is not None:
        ret.index = ret.index.tz_localize(None)
    ret.name = ticker.upper()
    return ret


def build_quantstats_html(results, benchmark_ticker: str | None = None,
                          title: str = "Strategy Tearsheet") -> bytes:
    """Generate a QuantStats HTML tearsheet and return it as bytes.

    benchmark_ticker:
        None  -> use the backtest's own benchmark (if present)
        "SPY" -> fetch that ticker via yfinance and use it as the benchmark
    """
    import quantstats as qs

    returns = _strategy_returns(results)

    if benchmark_ticker:
        benchmark = _fetch_ticker_returns(
            benchmark_ticker.strip(), returns.index.min(), returns.index.max())
    else:
        benchmark = _backtest_benchmark_returns(results)  # may be None

    tmp_path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(returns, benchmark=benchmark, title=title,
                            output=tmp_path)
        with open(tmp_path, "rb") as fh:
            data = fh.read()
    except QuantStatsError:
        raise
    except Exception as e:
        raise QuantStatsError(f"QuantStats failed to build the tearsheet: {e}") from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # QuantStats can leave figures open; close them to free memory.
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    return data