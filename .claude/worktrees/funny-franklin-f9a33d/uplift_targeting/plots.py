"""README figures, each in a light and a dark variant. Reads only results/*, so the
charts can be redrawn without refitting anything: python -m uplift_targeting.plots

Design: thin marks, hairline recessive grid, 2px lines, 4px rounded bar ends, a legend
for two or more series, and text in ink colors (never in a series color). The two
series colors are the first two slots of a palette validated for color-blind
separation in both modes.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PX = 0.72  # points per CSS pixel: figures are laid out at 100 px per inch
DPI = 200  # saved at 2x for sharp rendering

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "quiet": "#c9c8c2",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "blue": "#3987e5",
        "orange": "#d95926",
        "quiet": "#57564f",
    },
}

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)


# --- layout helpers -------------------------------------------------------------------


class Canvas:
    """A figure sized in CSS pixels, with the title block at the top left."""

    def __init__(self, theme: str, width: int, height: int):
        self.t = THEMES[theme]
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=DPI)
        self.fig.patch.set_facecolor(self.t["surface"])

    def title(self, text: str, subtitle: str, left: int = 24, top: int = 20) -> None:
        self.fig.text(
            left / self.w, 1 - top / self.h, text, ha="left", va="top",
            fontsize=16 * PX, fontweight="semibold", color=self.t["ink"],
        )
        self.fig.text(
            left / self.w, 1 - (top + 26) / self.h, subtitle, ha="left", va="top",
            fontsize=12.5 * PX, color=self.t["ink2"], linespacing=1.45,
        )

    def axes(self, left: int, right: int, top: int, bottom: int):
        ax = self.fig.add_axes(
            [left / self.w, bottom / self.h, (self.w - left - right) / self.w,
             (self.h - top - bottom) / self.h]
        )
        ax.set_facecolor(self.t["surface"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=11.5 * PX, labelcolor=self.t["muted"], pad=6 * PX)
        ax.set_axisbelow(True)
        return ax

    def note(self, text: str, left: int = 24, bottom: int = 14) -> None:
        self.fig.text(
            left / self.w, bottom / self.h, text, ha="left", va="bottom",
            fontsize=11 * PX, color=self.t["muted"],
        )

    def save(self, name: str, theme: str, folder=None) -> None:
        suffix = "" if theme == "light" else "-dark"
        folder = config.FIGURES_DIR if folder is None else folder
        self.fig.savefig(
            folder / f"{name}{suffix}.png", dpi=DPI,
            facecolor=self.t["surface"],
        )
        plt.close(self.fig)


def grid(ax, t, axis: str) -> None:
    ax.grid(axis=axis, color=t["grid"], linewidth=1 * PX)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)


def label(ax, t, text: str, axis: str) -> None:
    kwargs = dict(fontsize=12 * PX, color=t["ink2"], labelpad=8 * PX)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(text, **kwargs)


def legend(fig, t, items: list[tuple[str, str, str]], x: float, y: float) -> None:
    """items: (label, color, kind) with kind 'line', 'dot' or 'square'."""
    handles = []
    for text, color, kind in items:
        if kind == "line":
            handles.append(Line2D([], [], color=color, linewidth=2 * PX, label=text))
        elif kind == "dot":
            handles.append(Line2D([], [], color=color, marker="o", linestyle="none",
                                  markersize=8 * PX, label=text))
        else:
            handles.append(Line2D([], [], color=color, marker="s", linestyle="none",
                                  markersize=9 * PX, label=text))
    leg = fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncols=len(items),
        frameon=False, fontsize=11.5 * PX, handlelength=1.6, handletextpad=0.5,
        columnspacing=1.6, borderaxespad=0,
    )
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def _px_in_data(ax) -> tuple[float, float]:
    """Size of one CSS pixel in data units along x and y."""
    bbox = ax.get_window_extent()
    scale = ax.figure.dpi / 100
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return scale * (x1 - x0) / bbox.width, scale * (y1 - y0) / bbox.height


def rounded_bar(ax, center: float, value: float, thickness: float, color: str,
                horizontal: bool = False, radius_px: float = 4) -> None:
    """A bar from 0 to `value`: square at the baseline, 4px rounded at the data end."""
    if value == 0:
        return
    px_x, px_y = _px_in_data(ax)
    k = 0.5523  # cubic Bezier approximation of a quarter circle
    sign = 1 if value > 0 else -1
    # Work in (along, across) coordinates, then map to (x, y).
    r_along = min(radius_px * (px_x if horizontal else px_y), abs(value))
    r_across = min(radius_px * (px_y if horizontal else px_x), thickness / 2)
    lo, hi = center - thickness / 2, center + thickness / 2
    end, near = value, value - sign * r_along
    pts = [
        (0, lo), (near, lo),
        (near + sign * k * r_along, lo), (end, lo + r_across - k * r_across), (end, lo + r_across),
        (end, hi - r_across),
        (end, hi - r_across + k * r_across), (near + sign * k * r_along, hi), (near, hi),
        (0, hi), (0, lo),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
    verts = pts if horizontal else [(b, a) for a, b in pts]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def _load():
    r = config.RESULTS_DIR
    return {
        "summary": json.loads((r / "summary.json").read_text()),
        "experiment": json.loads((r / "experiment.json").read_text()),
        "which": pd.read_csv(r / "which_email.csv"),
        "deciles": pd.read_csv(r / "deciles.csv"),
        "cv": pd.read_csv(r / "cv_comparison.csv"),
        "cv_curves": pd.read_csv(r / "uplift_curves_cv_spend.csv"),
        "test_curves": pd.read_csv(r / "uplift_curves_test_spend.csv"),
    }


# --- figures --------------------------------------------------------------------------


def which_email(d, theme: str) -> None:
    c = Canvas(theme, 820, 400)
    t = c.t
    c.title(
        "The men's email lifts visits in every group. The women's email does nothing for men's-only buyers.",
        "Extra customers visiting the site within two weeks, per 100 emailed, with 95% intervals. "
        "Held-out test set (25,600 customers).",
    )
    ax = c.axes(left=190, right=40, top=118, bottom=58)
    w = d["which"][d["which"]["outcome"] == "visit"]
    rows = [
        ("everyone", "Everyone"),
        ("bought both", "Bought men's and women's"),
        ("bought women's only", "Bought women's only"),
        ("bought men's only", "Bought men's only"),
    ]
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(-2.5, 19)
    grid(ax, t, "x")
    ax.axvline(0, color=t["axis"], linewidth=1 * PX, zorder=1)
    for i, (seg, _) in enumerate(rows):
        for email, color, offset in (("Men's email", t["blue"], -0.14), ("Women's email", t["orange"], 0.14)):
            r = w[(w["segment"] == seg) & (w["email"] == email)].iloc[0]
            y = i + offset
            ax.plot([r["lo"] * 100, r["hi"] * 100], [y, y], color=color, linewidth=2 * PX,
                    solid_capstyle="round", zorder=3)
            ax.plot(r["effect"] * 100, y, "o", color=color, markersize=9 * PX,
                    markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=4)
            key = (seg, email)
            if key in {("everyone", "Men's email"), ("bought both", "Men's email"),
                       ("bought men's only", "Women's email")}:
                ax.text(r["hi"] * 100 + 0.5, y, f"+{r['effect'] * 100:.1f}", va="center",
                        ha="left", fontsize=11.5 * PX, color=t["ink"], fontweight="semibold")
    ax.set_yticks(range(len(rows)), [name for _, name in rows])
    ax.tick_params(axis="y", labelsize=12 * PX, labelcolor=t["ink2"])
    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    label(ax, t, "Extra visitors per 100 customers", "x")
    legend(c.fig, t, [("Men's email", t["blue"], "dot"), ("Women's email", t["orange"], "dot")],
           x=190 / c.w, y=1 - 84 / c.h)
    c.save("which_email", theme)


def deciles(d, theme: str) -> None:
    c = Canvas(theme, 820, 400)
    t = c.t
    dec = d["deciles"]
    everyone = d["which"].query("segment == 'everyone' and email == \"Men's email\" and outcome == 'visit'")
    avg = float(everyone["effect"].iloc[0]) * 100
    top = dec.iloc[0]["observed_visit_uplift"] * 100
    c.title(
        f"The model's top 10% get {top / avg:.1f}x the average visit lift. Below that, the ranking shows no pattern.",
        "Extra visitors per 100 customers from the men's email, by tenth of the model's predicted uplift, "
        "with 95% intervals.\nHeld-out test set; the model never saw these customers.",
    )
    ax = c.axes(left=64, right=120, top=112, bottom=62)
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, 18)
    grid(ax, t, "y")
    px_x, _ = _px_in_data(ax)
    thickness = min(0.62, 24 * px_x)
    for _, r in dec.iterrows():
        color = t["blue"] if r["decile"] == 1 else t["quiet"]
        rounded_bar(ax, r["decile"], r["observed_visit_uplift"] * 100, thickness, color)
        ax.plot([r["decile"]] * 2, [r["visit_lo"] * 100, r["visit_hi"] * 100],
                color=t["ink2"], linewidth=1.5 * PX, solid_capstyle="round", zorder=3)
    ax.axhline(avg, color=t["ink2"], linewidth=1 * PX, zorder=2.5)
    ax.text(10.75, avg, f"Everyone\n+{avg:.1f}", ha="left", va="center", clip_on=False,
            fontsize=11.5 * PX, color=t["ink2"], linespacing=1.3)
    ax.text(1, dec.iloc[0]["visit_hi"] * 100 + 0.35, f"+{top:.1f}", ha="center", va="bottom",
            fontsize=12 * PX, color=t["ink"], fontweight="semibold")
    ax.axhline(0, color=t["axis"], linewidth=1 * PX, zorder=2.6)
    ax.set_xticks(range(1, 11), ["Top\n10%"] + [str(i) for i in range(2, 10)] + ["Bottom\n10%"])
    label(ax, t, "Customers ranked by predicted uplift, in tenths", "x")
    label(ax, t, "Extra visitors per 100", "y")
    c.save("deciles_visits", theme)


def cv_vs_test(d, theme: str) -> None:
    c = Canvas(theme, 880, 430)
    t = c.t
    c.title(
        "Cross-validation suggested targeting pays. The held-out test can't confirm it.",
        "Extra revenue per 1,000 customers when only the top-ranked share of customers gets the men's email. "
        "Shaded: 95% interval.",
    )
    legend(
        c.fig, t,
        [("Uplift model (chosen)", t["blue"], "line"), ("Response model (likely buyers)", t["orange"], "line"),
         ("Random targeting", t["muted"], "line")],
        x=24 / c.w, y=1 - 78 / c.h,
    )
    panels = [
        ("Training set, cross-validated", d["cv_curves"], False),
        ("Held-out test set", d["test_curves"], True),
    ]
    ymin, ymax = -300, 1100
    width = (c.w - 72 - 40 - 48) / 2
    for i, (name, curves, band) in enumerate(panels):
        left = 72 + i * (width + 48)
        ax = c.axes(left=int(left), right=int(c.w - left - width), top=132, bottom=60)
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin, ymax)
        grid(ax, t, "y")
        f = curves["fraction_emailed"].to_numpy()
        end = curves["chosen"].iloc[-1]
        ax.plot([0, 1], [0, end], color=t["muted"], linewidth=2 * PX, zorder=2)
        if band:
            ax.fill_between(f, curves["chosen_lo"], curves["chosen_hi"], color=t["blue"],
                            alpha=0.12, linewidth=0, zorder=1)
        ax.plot(f, curves["best_response"], color=t["orange"], linewidth=2 * PX, zorder=3,
                solid_joinstyle="round")
        ax.plot(f, curves["chosen"], color=t["blue"], linewidth=2 * PX, zorder=4,
                solid_joinstyle="round")
        ax.axhline(0, color=t["axis"], linewidth=1 * PX, zorder=1.5)
        ax.set_title(name, loc="left", fontsize=12.5 * PX, color=t["ink"], pad=10 * PX,
                     fontweight="semibold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        label(ax, t, "Share of customers emailed", "x")
        if i == 0:
            label(ax, t, "Extra revenue per 1,000 customers", "y")
        else:
            ax.tick_params(axis="y", labelleft=True)
        # Mark the 10% budget, where the comparison in the README is made.
        g = float(np.interp(0.1, f, curves["chosen"]))
        ax.plot(0.1, g, "o", color=t["blue"], markersize=8 * PX, markeredgecolor=t["surface"],
                markeredgewidth=2 * PX, zorder=5)
        ax.text(0.13, g, f"${g:,.0f} at 10%", va="center", ha="left", fontsize=11.5 * PX,
                color=t["ink"], fontweight="semibold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=t["surface"], edgecolor="none"))
    c.save("cv_vs_test", theme)


def cv_ranking(d, theme: str) -> None:
    cv = d["cv"].sort_values("area_over_random", ascending=True).reset_index(drop=True)
    chosen = d["summary"]["selection"]["chosen"]
    leaders = 0
    for _, r in cv.iloc[::-1].iterrows():
        if r["trained_on"] != "visits":
            break
        leaders += 1
    words = {1: "first", 2: "top two", 3: "top three", 4: "top four", 5: "top five", 6: "top six"}
    c = Canvas(theme, 820, 560)
    t = c.t
    c.title(
        f"In cross-validation, models trained on visits took the {words.get(leaders, str(leaders))} places",
        "Average extra revenue over random targeting across all budgets, $ per 1,000 customers. "
        "Training set, 5-fold out-of-fold.",
    )
    legend(
        c.fig, t,
        [("Trained on visits", t["blue"], "square"), ("Trained on spend", t["orange"], "square"),
         ("Response model (likely buyers)", t["quiet"], "square")],
        x=24 / c.w, y=1 - 78 / c.h,
    )
    ax = c.axes(left=232, right=96, top=124, bottom=58)
    vals = cv["area_over_random"].to_numpy() * 1000
    ax.set_ylim(-0.7, len(cv) - 0.3)
    ax.set_xlim(min(-60, vals.min() * 1.25), vals.max() * 1.18)
    grid(ax, t, "x")
    _, px_y = _px_in_data(ax)
    thickness = min(0.62, 14 * px_y)
    for i, r in cv.iterrows():
        if r["learner"] == "Response model":
            color = t["quiet"]
        else:
            color = t["blue"] if r["trained_on"] == "visits" else t["orange"]
        rounded_bar(ax, i, vals[i], thickness, color, horizontal=True)
        if r["candidate"] == chosen:
            ax.text(vals[i] + 4, i, f"${vals[i]:.0f}  chosen", va="center", ha="left",
                    fontsize=11.5 * PX, color=t["ink"], fontweight="semibold")
    ax.axvline(0, color=t["axis"], linewidth=1 * PX, zorder=2.5)
    names = [
        f"{r['learner']}" + ("" if r["learner"] == "Response model" else f" on {r['trained_on']}")
        + f" · {r['reg']}"
        for _, r in cv.iterrows()
    ]
    ax.set_yticks(range(len(cv)), names)
    ax.tick_params(axis="y", labelsize=11.5 * PX, labelcolor=t["ink2"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"))
    label(ax, t, "Gain over random targeting, $ per 1,000 customers", "x")
    c.save("cv_ranking", theme)


def make_all() -> None:
    config.FIGURES_DIR.mkdir(exist_ok=True)
    d = _load()
    for theme in THEMES:
        which_email(d, theme)
        deciles(d, theme)
        cv_vs_test(d, theme)
        cv_ranking(d, theme)


if __name__ == "__main__":
    make_all()
