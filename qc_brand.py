"""
qc_brand.py
===========
Single source of truth for Castellan branding: colors, matplotlib styling,
and the CSS injected into the Streamlit app. Keeping it here means the app,
the charts, and the PDF all stay visually consistent and can be rebranded in
one place.
"""

from __future__ import annotations
import matplotlib.pyplot as plt

# ---- Castellan palette (from the logo / castellangroup.com) ----
NAVY        = "#1A3A5C"   # primary text / headers
NAVY_DARK   = "#12283F"
GREEN       = "#7AB648"   # accent / interactive
GREEN_DARK  = "#5E9134"
SLATE       = "#5C6B7A"   # secondary text
LIGHT_BG    = "#F4F6F8"
GRID        = "#D9E0E6"

# Series colors for multi-line / comparison charts
SERIES_COLORS = [NAVY, GREEN, "#3E6E99", "#9CC472", "#8A6D3B", "#B0563C"]

# Primary / comparison roles used across charts
PRIMARY_COLOR = NAVY
COMPARE_COLOR = GREEN


def apply_mpl_theme():
    """Apply Castellan styling to all matplotlib figures created afterward."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": NAVY,
        "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
        "font.size": 10,
        "legend.frameon": False,
    })


def streamlit_css() -> str:
    """CSS to inject into the Streamlit app for a Castellan look."""
    return f"""
    <style>
      /* Headings in Castellan navy */
      h1, h2, h3, h4 {{ color: {NAVY}; }}
      /* Top accent bar */
      .block-container {{ padding-top: 2rem; }}
      /* Metric values in navy, labels in slate */
      [data-testid="stMetricValue"] {{ color: {NAVY}; }}
      [data-testid="stMetricLabel"] {{ color: {SLATE}; }}
      /* Buttons / download buttons in Castellan green */
      .stButton > button, .stDownloadButton > button {{
          background-color: {GREEN};
          color: white;
          border: none;
          border-radius: 6px;
          font-weight: 600;
      }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
          background-color: {GREEN_DARK};
          color: white;
      }}
      /* Sidebar tint */
      [data-testid="stSidebar"] {{ background-color: {LIGHT_BG}; }}
      /* Header rule under the title */
      .castellan-rule {{
          height: 3px;
          background: linear-gradient(90deg, {NAVY} 0%, {GREEN} 100%);
          border: none; margin: 0 0 1rem 0;
      }}
      .castellan-caption {{ color: {SLATE}; font-size: 0.85rem; }}
    </style>
    """
