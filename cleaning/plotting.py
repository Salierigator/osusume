"""Shared matplotlib style of the survey and EDA notebooks."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Categorical hues in fixed order: a series keeps its slot, slots are never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#d03b3b"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "figure.figsize": (8, 4),
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False,
    })


def thousands(ax, axis: str = "y") -> None:
    """Tick labels as 1.2k / 3.4M."""
    def fmt(v, _):
        for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if abs(v) >= size:
                return f"{v / size:g}{unit}"
        return f"{v:g}"
    getattr(ax, f"{axis}axis").set_major_formatter(mpl.ticker.FuncFormatter(fmt))


def note(ax, text: str) -> None:
    """Small caption under the plot area."""
    ax.annotate(text, xy=(0, -0.2), xycoords="axes fraction", fontsize=8.5, color=MUTED, va="top")
