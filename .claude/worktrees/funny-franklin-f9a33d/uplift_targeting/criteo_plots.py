"""Charts for the Criteo replication, each in a light and a dark variant, in the style
of plots.py. Reads only results/criteo/*, so the charts can be redrawn without
refitting anything: python -m uplift_targeting.criteo_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config
from .plots import PX, THEMES, Canvas, _px_in_data, grid, label, legend, rounded_bar

FAMILY_NAMES = {"visits": "visits", "conversion": "conversions"}


def _load(results_dir: Path) -> dict:
    r = Path(results_dir)
    return {
        "summary": json.loads((r / "summary.json").read_text()),
        "selection": json.loads((r / "selection.json").read_text()),
        "cv": pd.read_csv(r / "cv_comparison.csv"),
        "cv_curves": pd.read_csv(r / "uplift_curves_cv.csv"),
        "test_curves": pd.read_csv(r / "uplift_curves_test_conversion.csv"),
        "deciles": pd.read_csv(r / "deciles.csv"),
    }


def _users(n: int) -> str:
    return f"{n / 1e6:.1f} million" if n >= 1e6 else f"{n:,}"


def _signed(v: float) -> str:
    return f"+{v:,.0f}" if v >= 0 else f"-{-v:,.0f}"


def _family_color(t: dict, lbl: str) -> str:
    trained_on = lbl.split("|")[1].strip()
    return t["blue"] if trained_on == "visits" else t["orange"]


# --- figures ---------------------------------------------------------------------------


def deciles(d: dict, theme: str, folder: Path) -> None:
    c = Canvas(theme, 820, 400)
    t = c.t
    dec = d["deciles"]
    share = d["summary"]["conversion"]["as_randomized"]["models"]["chosen"]["10pct"]["share_of_full_gain"]
    top = float(dec.iloc[0]["conversion_uplift_per_100k"])
    c.title(
        f"The model's top 10% account for {share:.0%} of the extra conversions from ads",
        "Extra conversions per 100,000 users from the ads, by tenth of the chosen model's predicted "
        "uplift, with 95% intervals.\n"
        f"Held-out test set ({_users(d['summary']['users']['test'])} users); the model never saw them.",
    )
    ax = c.axes(left=72, right=40, top=112, bottom=62)
    ax.set_xlim(0.4, 10.6)
    lo = min(0.0, float(dec["conversion_uplift_lo_per_100k"].min()))
    hi = float(dec["conversion_uplift_hi_per_100k"].max())
    ax.set_ylim(lo * 1.15, hi * 1.12)
    grid(ax, t, "y")
    px_x, px_y = _px_in_data(ax)
    thickness = min(0.62, 24 * px_x)
    for _, r in dec.iterrows():
        color = t["blue"] if r["decile"] == 1 else t["quiet"]
        rounded_bar(ax, r["decile"], r["conversion_uplift_per_100k"], thickness, color)
        ax.plot([r["decile"]] * 2, [r["conversion_uplift_lo_per_100k"], r["conversion_uplift_hi_per_100k"]],
                color=t["ink2"], linewidth=1.5 * PX, solid_capstyle="round", zorder=3)
    ax.text(1, dec.iloc[0]["conversion_uplift_hi_per_100k"] + 6 * px_y, _signed(top), ha="center",
            va="bottom", fontsize=12 * PX, color=t["ink"], fontweight="semibold")
    ax.axhline(0, color=t["axis"], linewidth=1 * PX, zorder=2.6)
    ax.set_xticks(range(1, 11), ["Top\n10%"] + [str(i) for i in range(2, 10)] + ["Bottom\n10%"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    label(ax, t, "Users ranked by predicted uplift, in tenths", "x")
    label(ax, t, "Extra conversions per 100,000", "y")
    c.save("deciles_conversions", theme, folder)


def cv_vs_test(d: dict, theme: str, folder: Path) -> None:
    c = Canvas(theme, 880, 430)
    t = c.t
    s = d["summary"]
    picks = s["selection"]
    result = s["conversion"]["as_randomized"]
    beats = {role: m["10pct"]["vs_random_ci"][0] > 0 for role, m in result["models"].items()}
    tie = all(diff["area"]["ci"][0] < 0 < diff["area"]["ci"][1] for diff in result["differences"].values())
    every = {2: "both", 3: "all three", 4: "all four"}.get(len(set(picks.values())), "all the")
    if all(beats.values()) and tie:
        headline = f"On held-out users, {every} models beat random targeting by about the same margin"
    elif beats["chosen"]:
        headline = "On held-out users, the chosen model beats random targeting"
    else:
        headline = "On held-out users, the chosen model can't be told from random targeting"
    c.title(
        headline,
        "Extra conversions per 100,000 users when only the top-ranked share of users gets ads. "
        "Shaded: 95% interval for the chosen model.",
    )
    lines = [
        ("best_visits", "Uplift model trained on visits", t["blue"]),
        ("best_conversion", "Uplift model trained on conversions", t["orange"]),
        ("best_response", "Response model (likely converters)", t["quiet"]),
    ]
    lines = [line for line in lines if line[0] in picks]
    legend(c.fig, t, [(name, color, "line") for _, name, color in lines]
           + [("Random targeting", t["muted"], "line")], x=24 / c.w, y=1 - 78 / c.h)
    panels = [
        (f"Training subsample, cross-validated ({_users(s['users']['cross_validation'])} users)",
         d["cv_curves"], False),
        (f"Held-out test set ({_users(s['users']['test'])} users)", d["test_curves"], True),
    ]
    roles = [role for role, _, _ in lines]
    values = np.concatenate([p[role].to_numpy() for _, p, _ in panels for role in roles]
                            + [d["test_curves"]["chosen_hi"].to_numpy()])
    ymin, ymax = min(0.0, float(values.min())) * 1.1, float(values.max()) * 1.12
    width = (c.w - 72 - 40 - 48) / 2
    for i, (name, curves, band) in enumerate(panels):
        left = 72 + i * (width + 48)
        ax = c.axes(left=int(left), right=int(c.w - left - width), top=132, bottom=60)
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin, ymax)
        grid(ax, t, "y")
        f = curves["fraction_targeted"].to_numpy()
        end = curves["chosen"].iloc[-1]
        ax.plot([0, 1], [0, end], color=t["muted"], linewidth=2 * PX, zorder=2)
        if band:
            ax.fill_between(f, curves["chosen_lo"], curves["chosen_hi"],
                            color=_family_color(t, picks["chosen"]), alpha=0.14, linewidth=0, zorder=1)
        for z, (role, _, color) in enumerate(reversed(lines)):
            ax.plot(f, curves[role], color=color, linewidth=2 * PX, zorder=3 + z, solid_joinstyle="round")
        ax.axhline(0, color=t["axis"], linewidth=1 * PX, zorder=1.5)
        ax.set_title(name, loc="left", fontsize=12.5 * PX, color=t["ink"], pad=10 * PX,
                     fontweight="semibold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        label(ax, t, "Share of users targeted", "x")
        if i == 0:
            label(ax, t, "Extra conversions per 100,000 users", "y")
        # Mark the 10% budget, where the comparison in the README is made.
        g = float(np.interp(0.1, f, curves["chosen"]))
        ax.plot(0.1, g, "o", color=_family_color(t, picks["chosen"]), markersize=8 * PX,
                markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=8)
        ax.text(0.13, g, f"{_signed(g)} at 10%", va="center", ha="left", fontsize=11.5 * PX,
                color=t["ink"], fontweight="semibold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=t["surface"], edgecolor="none"))
    c.save("cv_vs_test", theme, folder)


def cv_ranking(d: dict, theme: str, folder: Path) -> None:
    key = "area_over_random_per_100k"
    cv = d["cv"].sort_values(key, ascending=True).reset_index(drop=True)
    chosen = d["selection"]["chosen"]
    # How many of the top places went to models trained on the same outcome (the response
    # model is trained on the outcome that matters, so it counts with that family).
    ranked = cv.iloc[::-1]
    lead_family = ranked.iloc[0]["trained_on"]
    leaders = 0
    for _, r in ranked.iterrows():
        if r["trained_on"] != lead_family:
            break
        leaders += 1
    with_response = (ranked.iloc[:leaders]["learner"] == "Response model").any()
    family = FAMILY_NAMES.get(lead_family, lead_family)
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    if leaders == 1:
        headline = f"In cross-validation, a model trained on {family} ranked first"
    else:
        headline = (f"In cross-validation, models trained on {family}"
                    + (", including the response model," if with_response else "")
                    + f" took the top {words.get(leaders, leaders)} places")
    c = Canvas(theme, 820, 182 + 21 * len(cv))
    t = c.t
    c.title(
        headline,
        "Gain over random targeting averaged across budgets, per 100,000 users. "
        f"{_users(int(d['summary']['users']['cross_validation']))} training users, out-of-fold.",
    )
    legend(c.fig, t,
           [("Trained on visits", t["blue"], "square"), ("Trained on conversions", t["orange"], "square"),
            ("Response model (likely converters)", t["quiet"], "square")],
           x=24 / c.w, y=1 - 78 / c.h)
    ax = c.axes(left=232, right=96, top=124, bottom=58)
    vals = cv[key].to_numpy()
    span = max(vals.max(), 0) - min(vals.min(), 0)
    ax.set_ylim(-0.7, len(cv) - 0.3)
    ax.set_xlim(min(0.0, vals.min()) - 0.08 * span, max(vals.max(), 0) + 0.2 * span)
    grid(ax, t, "x")
    _, px_y = _px_in_data(ax)
    px_x, _ = _px_in_data(ax)
    thickness = min(0.62, 14 * px_y)
    for i, r in cv.iterrows():
        if r["learner"] == "Response model":
            color = t["quiet"]
        else:
            color = t["blue"] if r["trained_on"] == "visits" else t["orange"]
        rounded_bar(ax, i, vals[i], thickness, color, horizontal=True)
        if r["candidate"] == chosen:
            ax.text(max(vals[i], 0) + 6 * px_x, i, f"{vals[i]:,.0f}  chosen", va="center", ha="left",
                    fontsize=11.5 * PX, color=t["ink"], fontweight="semibold")
    ax.axvline(0, color=t["axis"], linewidth=1 * PX, zorder=2.5)
    names = [
        f"{r['learner']}"
        + ("" if r["learner"] == "Response model" else f" on {FAMILY_NAMES.get(r['trained_on'], r['trained_on'])}")
        + f" · {r['reg']}"
        for _, r in cv.iterrows()
    ]
    ax.set_yticks(range(len(cv)), names)
    ax.tick_params(axis="y", labelsize=11.5 * PX, labelcolor=t["ink2"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    label(ax, t, "Gain over random targeting, conversions per 100,000 users", "x")
    c.save("cv_ranking", theme, folder)


def make_all(results_dir: Path = config.CRITEO_RESULTS_DIR,
             figures_dir: Path = config.CRITEO_FIGURES_DIR) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = _load(results_dir)
    for theme in THEMES:
        deciles(d, theme, figures_dir)
        cv_vs_test(d, theme, figures_dir)
        cv_ranking(d, theme, figures_dir)


if __name__ == "__main__":
    make_all()
