"""
qc_parser.py
============
Reusable parsing layer for QuantConnect backtest results JSON files.

The QC results JSON is large and awkward to work with directly. This module
turns it into clean pandas objects so the dashboard (or any script) can just
ask for "the equity curve" or "the stats" without knowing the JSON layout.

Design goal: every chart/metric the dashboard renders goes through a function
here, so adding new visualizations later means adding one parser function +
one render function, nothing else.

Public API:
    load_results(path_or_buffer) -> BacktestResults
    BacktestResults.equity_curve()      -> pd.Series (daily portfolio value)
    BacktestResults.benchmark()         -> pd.Series or None (SPY etc.)
    BacktestResults.drawdown_series()   -> pd.Series (drawdown %, <= 0)
    BacktestResults.returns(freq)       -> pd.Series of periodic returns
    BacktestResults.statistics()        -> dict of QC's headline stats
    BacktestResults.computed_metrics()  -> dict of metrics we compute ourselves
    BacktestResults.list_charts()       -> list of available chart names
    BacktestResults.chart_series(name)  -> dict[str, pd.Series] for any chart
"""

from __future__ import annotations
import json
import io
import re
import numpy as np
import pandas as pd


class InvalidBacktestFile(Exception):
    """Raised when an uploaded file is not a usable QC results JSON."""
    pass


def _values_to_series(values):
    """Convert a QC 'values' array into a pandas Series indexed by datetime.

    QC rows are either [ts, value] (line/scatter) or [ts, o, h, l, c] (candle).
    For candles we take the close (last element). 'ts' is a unix timestamp.
    """
    if not values:
        return pd.Series(dtype=float)
    rows = []
    for row in values:
        if row is None or len(row) < 2:
            continue
        ts = row[0]
        val = row[-1]          # close for candles, the value for line series
        if val is None:
            continue
        rows.append((ts, val))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["ts", "val"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    return df.set_index("date")["val"].sort_index()


class BacktestResults:
    """Wraps one parsed QC results JSON and exposes clean accessors."""

    def __init__(self, raw: dict, name: str = "backtest"):
        self.raw = raw
        self.name = name
        self._charts = raw.get("charts", {}) or {}

    # ---------------- chart access ----------------
    def list_charts(self):
        """Names of all charts present in this result file."""
        return list(self._charts.keys())

    def chart_series(self, chart_name: str):
        """Return {series_name: pd.Series} for any chart, or {} if absent."""
        chart = self._charts.get(chart_name)
        if not chart or "series" not in chart:
            return {}
        out = {}
        for sname, sdata in chart["series"].items():
            out[sname] = _values_to_series(sdata.get("values", []))
        return out

    # ---------------- core series ----------------
    def equity_curve(self, daily: bool = True) -> pd.Series:
        """Portfolio value over time (daily close by default)."""
        s = self.chart_series("Strategy Equity").get("Equity", pd.Series(dtype=float))
        if daily and not s.empty:
            s = s.resample("D").last().dropna()
        return s

    def benchmark(self, daily: bool = True):
        """Benchmark price series (e.g. SPY), or None if not present."""
        series = self.chart_series("Benchmark")
        if not series:
            return None
        s = list(series.values())[0]
        if daily and not s.empty:
            s = s.resample("D").last().dropna()
        return s

    def drawdown_series(self) -> pd.Series:
        """Drawdown as a fraction (<= 0), computed from the equity curve."""
        eq = self.equity_curve()
        if eq.empty:
            return eq
        return eq / eq.cummax() - 1.0

    def returns(self, freq: str = "ME") -> pd.Series:
        """Periodic returns. freq: 'D','W','ME' (month-end),'QE','YE'."""
        eq = self.equity_curve()
        if eq.empty:
            return eq
        return eq.resample(freq).last().pct_change().dropna()

    # ---------------- closed trades ----------------
    @staticmethod
    def _duration_to_days(s):
        """Parse a .NET TimeSpan string ('89.23:00:00', '5:30:00') to days."""
        if s is None:
            return np.nan
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).strip()
        neg = s.startswith("-")
        s = s.lstrip("-")
        m = re.match(r"^(?:(\d+)\.)?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$", s)
        if not m:
            try:
                return float(s)
            except ValueError:
                return np.nan
        days = int(m.group(1) or 0)
        hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        total = days + hh / 24 + mm / 1440 + ss / 86400
        if m.group(5):
            total += float("0." + m.group(5)) / 86400
        return -total if neg else total

    def closed_trades(self) -> pd.DataFrame:
        """Return QC's closed trades as a tidy DataFrame, or empty if absent.

        Columns: ticker, direction, quantity, entry_time, entry_price,
        exit_time, exit_price, pnl, fees, return_pct, duration_days, is_win.
        return_pct is profit/loss over entry notional, so it is sign-correct
        for both long and short trades.
        """
        ct = (self.raw.get("totalPerformance") or {}).get("closedTrades")
        if not ct:
            return pd.DataFrame()
        rows = []
        for t in ct:
            sym = t.get("symbol") or {}
            if isinstance(sym, dict):
                ticker = sym.get("value") or sym.get("permtick") or str(sym.get("id", "?"))
            else:
                ticker = str(sym)
            ep = t.get("entryPrice")
            xp = t.get("exitPrice")
            qty = t.get("quantity") or 0
            pnl = t.get("profitLoss")
            notional = (ep or 0) * abs(qty)
            ret = (pnl / notional) if (notional and pnl is not None) else np.nan
            rows.append({
                "ticker": ticker,
                "direction": "Short" if t.get("direction") == 1 else "Long",
                "quantity": qty,
                "entry_time": pd.to_datetime(t.get("entryTime"), utc=True, errors="coerce"),
                "entry_price": ep,
                "exit_time": pd.to_datetime(t.get("exitTime"), utc=True, errors="coerce"),
                "exit_price": xp,
                "pnl": pnl,
                "fees": t.get("totalFees"),
                "return_pct": ret,
                "duration_days": self._duration_to_days(t.get("duration")),
                "is_win": t.get("isWin"),
            })
        df = pd.DataFrame(rows)
        # drop timezone so these line up cleanly with other (naive) series
        for c in ("entry_time", "exit_time"):
            if c in df and pd.api.types.is_datetime64tz_dtype(df[c]):
                df[c] = df[c].dt.tz_localize(None)
        return df.sort_values("entry_time").reset_index(drop=True)

    # ---------------- stats ----------------
    def statistics(self) -> dict:
        """QC's own headline statistics block (strings as given)."""
        return self.raw.get("statistics", {}) or {}

    def computed_metrics(self) -> dict:
        """Metrics we compute ourselves from the equity curve, so they are
        consistent regardless of QC version. Includes drawdown dating and
        SPY beta/correlation when a benchmark is present."""
        eq = self.equity_curve()
        if eq.empty:
            return {}
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
        dd = self.drawdown_series()
        max_dd = dd.min()
        trough = dd.idxmin()
        peak = eq[:trough].idxmax()
        rec = eq[trough:][eq[trough:] >= eq.loc[peak]]
        rec_days = int((rec.index[0] - trough).days) if len(rec) > 0 else None

        rm = self.returns("ME")
        vol = rm.std() * np.sqrt(12)
        sharpe = (rm.mean() * 12) / vol if vol and vol > 0 else np.nan

        beta = corr = np.nan
        bm = self.benchmark()
        if bm is not None and not bm.empty:
            bmm = bm.resample("ME").last().pct_change()
            j = pd.concat([rm, bmm], axis=1, keys=["s", "b"]).dropna()
            if len(j) > 2:
                beta = float(np.polyfit(j["b"], j["s"], 1)[0])
                corr = float(j["s"].corr(j["b"]))

        return {
            "start": eq.index[0].date(),
            "end": eq.index[-1].date(),
            "years": round(yrs, 1),
            "start_equity": float(eq.iloc[0]),
            "end_equity": float(eq.iloc[-1]),
            "cagr": float(cagr),
            "ann_vol": float(vol) if vol else np.nan,
            "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
            "max_drawdown": float(max_dd),
            "max_dd_trough": trough.date(),
            "recovery_days": rec_days,
            "beta_to_benchmark": beta,
            "corr_to_benchmark": corr,
        }


def load_results(path_or_buffer, name: str = None) -> BacktestResults:
    """Load a QC results JSON from a file path or a file-like/bytes buffer
    (Streamlit's uploaded_file works directly).

    Raises InvalidBacktestFile with a clear message if the file is not valid
    JSON or does not look like a QuantConnect results file.
    """
    if hasattr(path_or_buffer, "read"):
        data = path_or_buffer.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        nm = name or getattr(path_or_buffer, "name", "backtest")
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as e:
            raise InvalidBacktestFile(f"File is not valid JSON: {e}")
    else:
        nm = name or str(path_or_buffer)
        try:
            with open(path_or_buffer, "r") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidBacktestFile(f"File is not valid JSON: {e}")

    if not isinstance(raw, dict) or "charts" not in raw:
        raise InvalidBacktestFile(
            "This does not look like a QuantConnect results JSON "
            "(no 'charts' section). Use the file from the backtest "
            "Overview tab -> Download Results."
        )
    results = BacktestResults(raw, name=nm)
    if results.equity_curve().empty:
        raise InvalidBacktestFile(
            "No 'Strategy Equity' series found in this file, so there is "
            "nothing to chart. Make sure this is a completed backtest's "
            "results JSON."
        )
    return results