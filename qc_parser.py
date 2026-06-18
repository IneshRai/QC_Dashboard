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
    BacktestResults.benchmark_returns(freq) -> pd.Series (benchmark periodic returns)
    BacktestResults.exposure()          -> pd.DataFrame (long/short/net/gross/cash, daily)
    BacktestResults.orders()            -> pd.DataFrame of orders (with fill latency)
    BacktestResults.execution_latencies() -> pd.Series of submit->fill seconds
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

    def benchmark_returns(self, freq: str = "ME") -> pd.Series:
        """Periodic returns of the benchmark price series, same convention as
        returns() so the two are directly comparable. Empty Series if there is
        no benchmark in the file."""
        bm = self.benchmark()
        if bm is None or bm.empty:
            return pd.Series(dtype=float)
        return bm.resample(freq).last().pct_change().dropna()

    # ---------------- exposure / invested level ----------------
    def exposure(self) -> pd.DataFrame:
        """Daily portfolio composition as fractions of total portfolio value.

        Built from QC's 'Exposure' chart, whose series are named
        '<SecurityType> - Long Ratio' / '<SecurityType> - Short Ratio' and hold
        holdings_value / portfolio_value (shorts stored negative). We sum across
        all security types to get the whole-portfolio picture.

        Returns a DataFrame indexed by day with columns (all fractions of PV):
            long   total long exposure   (>= 0; can exceed 1 on margin)
            short  total short exposure   (<= 0)
            net    long + short           (signed net invested)
            gross  long - short           (total capital at work, >= 0)
            cash   1 - net                (negative when on margin; > 1 net short)

        Empty DataFrame if the file has no 'Exposure' chart (older LEAN, or a
        backtest that never held anything).
        """
        series = self.chart_series("Exposure")
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series).sort_index()
        if df.empty:
            return pd.DataFrame()
        # A type that isn't held at a sampled instant simply has no point there;
        # treat that as 0 exposure rather than a gap to carry forward.
        df = df.fillna(0.0)
        long_cols = [c for c in df.columns if str(c).endswith("Long Ratio")]
        short_cols = [c for c in df.columns if str(c).endswith("Short Ratio")]
        long_sum = df[long_cols].sum(axis=1) if long_cols else pd.Series(0.0, index=df.index)
        short_sum = df[short_cols].sum(axis=1) if short_cols else pd.Series(0.0, index=df.index)
        out = pd.DataFrame({"long": long_sum, "short": short_sum})
        # Collapse to one composition per day and carry it across non-sampled
        # days (weekends/holidays) so the line stays continuous.
        out = out.resample("D").last().ffill().dropna(how="all")
        out["net"] = out["long"] + out["short"]
        out["gross"] = out["long"] - out["short"]
        out["cash"] = 1.0 - out["net"]
        return out

    # ---------------- orders / execution ----------------
    # QC order type ids -> readable labels (Common/Orders/OrderTypes.cs)
    _ORDER_TYPES = {
        0: "Market", 1: "Limit", 2: "StopMarket", 3: "StopLimit",
        4: "MarketOnOpen", 5: "MarketOnClose", 6: "OptionExercise",
        7: "LimitIfTouched", 8: "ComboMarket", 9: "ComboLimit",
        10: "ComboLegLimit", 11: "TrailingStop",
    }

    def orders(self) -> pd.DataFrame:
        """All orders from the results JSON as a tidy DataFrame, or empty.

        Columns: id, ticker, type (int), type_name, status (int), quantity,
        submit_time, fill_time, latency, latency_seconds. submit_time is the
        order's 'time' (when it was created/submitted) and fill_time is
        'lastFillTime'; latency is the gap between them. Times are tz-naive UTC.
        """
        od = self.raw.get("orders") or {}
        if not od:
            return pd.DataFrame()
        # 'orders' is usually a dict keyed by order id, occasionally a list.
        items = od.values() if isinstance(od, dict) else od
        rows = []
        for o in items:
            if not isinstance(o, dict):
                continue
            otype = o.get("type")
            rows.append({
                "id": o.get("id"),
                "ticker": self._ticker_of(o.get("symbol")),
                "type": otype,
                "type_name": self._ORDER_TYPES.get(otype, str(otype)),
                "status": o.get("status"),
                "quantity": o.get("quantity"),
                "submit_time": pd.to_datetime(o.get("time") or o.get("createdTime"),
                                              utc=True, errors="coerce"),
                "fill_time": pd.to_datetime(o.get("lastFillTime"),
                                            utc=True, errors="coerce"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in ("submit_time", "fill_time"):
            if pd.api.types.is_datetime64tz_dtype(df[c]):
                df[c] = df[c].dt.tz_localize(None)
        df["latency"] = df["fill_time"] - df["submit_time"]
        df["latency_seconds"] = df["latency"].dt.total_seconds()
        return df

    def execution_latencies(self) -> pd.Series:
        """Submit->fill latency in SECONDS for every filled order, as a Series.

        Only orders that actually filled (have a fill_time) and whose latency is
        non-negative are included. Empty Series if there are no fills/orders.
        """
        df = self.orders()
        if df.empty or "latency_seconds" not in df:
            return pd.Series(dtype=float)
        s = df["latency_seconds"].dropna()
        s = s[s >= 0]
        return s.reset_index(drop=True)

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

    @staticmethod
    def _ticker_of(sym):
        """Best-effort human ticker from QC's many symbol shapes.

        Handles: dict with value/permtick/ticker, a plain string, or an
        encoded SID string like 'AMR WMX6K8NSR9ET' (take the leading token).
        Returns '?' only if nothing usable is present.
        """
        if sym is None:
            return "?"
        if isinstance(sym, list):
            sym = sym[0] if sym else None
            if sym is None:
                return "?"
        if isinstance(sym, dict):
            for k in ("value", "permtick", "ticker", "symbol", "Value", "Symbol"):
                v = sym.get(k)
                if v:
                    return str(v).split(" ")[0]
            sid = sym.get("id") or sym.get("ID") or sym.get("Id")
            if sid:
                return str(sid).split(" ")[0]
            return "?"
        s = str(sym).strip()
        return s.split(" ")[0] if s else "?"

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
            sym = t.get("symbol")
            if sym is None:
                syms = t.get("symbols")
                if isinstance(syms, list) and syms:
                    sym = syms[0]
            ticker = self._ticker_of(sym)
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