"""Export the numbers behind the demo page in docs/ (served by GitHub Pages).

Reads only results/, like plots.py, so the page can be rebuilt without refitting
anything: python -m uplift_targeting.demo

Each dataset gets its own file, docs/data/<name>.json, and its own section of the page,
which reads nothing else: results/ for the email experiment, results/criteo/ for the
replication. To add a dataset, write a build function for it and list it in EXPORTS.

The page draws these numbers and does only simple arithmetic on them: the profit of
emailing everyone at the cost and margin a visitor picks (the formula in
analysis.blanket_economics), and random targeting as the budget times the effect of
treating everyone. The build functions check that arithmetic against the results before
anything is written. No model runs in the browser.

To preview the page, serve docs/ (it loads its data with fetch, so opening the file
directly won't work): python -m http.server 8000 --directory docs
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config

DATA_DIR = config.ROOT / "docs" / "data"
PER_1000 = 1000
TOLERANCE = 0.01  # results are rounded to 4-6 decimals; amounts are per 1,000 customers


def _read_json(name: str, folder: Path = config.RESULTS_DIR) -> dict:
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _read_csv(name: str, folder: Path = config.RESULTS_DIR) -> list[dict[str, str]]:
    with open(folder / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _columns(rows: list[dict[str, str]], names: list[str] | None = None) -> dict[str, list[float]]:
    return {n: [float(r[n]) for r in rows] for n in (names or list(rows[0]))}


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"demo export: {message}")


def _uniform_grid(values: list[float], name: str) -> None:
    steps = [b - a for a, b in zip(values, values[1:])]
    _check(len(values) > 1 and all(abs(s - steps[0]) < 1e-9 and s > 0 for s in steps),
           f"{name} must be an evenly spaced, increasing grid")


def everyone_profit(effect: float, margin: float, cost: float) -> float:
    """Profit per 1,000 customers of emailing everyone, as in analysis.blanket_economics.
    The page uses the same formula in app.js, for the estimate and both interval ends."""
    return (margin * effect - cost) * PER_1000


def _check_everyone_formula(summary: dict, mens: dict) -> None:
    blanket = summary["email_everyone"]
    margin = blanket["margin"]
    _check(abs(margin * mens["spend_effect"] - blanket["breakeven_cost"]) < 1e-5,
           "break-even cost in summary.json doesn't match margin x men's-email spend effect")
    for cost, row in blanket["profit_per_1000"].items():
        page = [everyone_profit(v, margin, float(cost))
                for v in (mens["spend_effect"], *mens["spend_ci"])]
        committed = [row["estimate"], *row["ci"]]
        _check(all(abs(a - b) < TOLERANCE for a, b in zip(page, committed)),
               f"profit of emailing everyone at ${cost} doesn't match summary.json")


def _curves(summary_block: dict, outcome: str) -> dict:
    rows = _read_csv(f"uplift_curves_test_{outcome}.csv")
    roles = ("chosen", "best_response")
    cols = ["fraction_emailed"] + [f"{r}{s}" for r in roles for s in ("", "_lo", "_hi")]
    out = _columns(rows, cols)
    _uniform_grid(out["fraction_emailed"], f"fraction_emailed in {outcome} curves")
    _check(out["fraction_emailed"][0] == 0 and out["fraction_emailed"][-1] == 1,
           f"{outcome} curves must run from 0% to 100% of customers")
    full = summary_block["full_email_gain_per_1000"]
    for role in roles:
        _check(abs(out[role][-1] - full) < TOLERANCE,
               f"{outcome} curve of {role} must end at the effect of emailing everyone")
    chosen = summary_block["budgets"]["chosen"]
    out["full_gain"] = full
    out["full_gain_ci"] = summary_block["full_email_gain_ci_per_1000"]
    # Gain over random targeting with paired bootstrap intervals, only computed at these budgets.
    out["vs_random"] = {
        key: {"budget": int(key.removesuffix("pct")) / 100, "per_1000": v["vs_random_per_1000"],
              "ci": v["vs_random_ci_per_1000"]}
        for key, v in chosen.items() if key.endswith("pct")
    }
    return out


def build_hillstrom() -> dict:
    """The email experiment: everything in results/ (top level)."""
    summary = _read_json("summary.json")
    experiment = _read_json("experiment.json")
    control, mens, womens = (experiment["arms"][label] for label in config.ARM_LABELS)
    _check_everyone_formula(summary, mens)

    sweep = _columns(_read_csv("cost_sweep.csv"))
    _uniform_grid(sweep["cost_per_email"], "cost_per_email in cost_sweep.csv")
    economics = summary["economics"]
    _check(any(abs(c - economics["cost_per_email"]) < 1e-9 for c in sweep["cost_per_email"]),
           "the assumed cost per email must be one of the costs in cost_sweep.csv")
    _check(abs(economics["margin"] - summary["email_everyone"]["margin"]) < 1e-9,
           "cost_sweep.csv and email_everyone must assume the same margin")

    deciles = _columns(_read_csv("deciles.csv"))
    _check(deciles["decile"] == [float(d) for d in range(1, 11)], "deciles.csv must hold tenths 1 to 10")

    which = _read_csv("which_email.csv")
    everyone = {
        r["outcome"]: {"effect": float(r["effect"]), "ci": [float(r["lo"]), float(r["hi"])]}
        for r in which if r["segment"] == "everyone" and r["email"] == config.ARM_LABELS[1]
    }
    _check(set(everyone) == {"visit", "spend"}, "which_email.csv must hold the men's email's effect on everyone")

    with open(config.RESULTS_DIR / "profile.csv", newline="", encoding="utf-8") as f:
        header, *rows = list(csv.reader(f))
    profile = {"columns": header[1:], "rows": [[r[0], *map(float, r[1:])] for r in rows]}
    # The page's text names these rows.
    labels = {r[0] for r in profile["rows"]}
    _check({"customers", "bought men's and women's"} <= labels, "profile.csv is missing a row the page names")

    def arm(a: dict, label: str) -> dict:
        keys = ("n", "visit_rate", "spend_per_customer", "visit_effect", "visit_ci", "spend_effect", "spend_ci")
        return {"label": label, **{k: a[k] for k in keys if k in a}}

    return {
        "note": "Written by python -m uplift_targeting.demo from results/. Don't edit by hand.",
        "experiment": {
            "customers": experiment["n_customers"],
            "purchase_rate": experiment["purchase_rate"],
            "average_order": experiment["average_order"],
            "control": arm(control, config.ARM_LABELS[0]),
            "mens": arm(mens, config.ARM_LABELS[1]),
            "womens": arm(womens, config.ARM_LABELS[2]),
        },
        "split": {"train": summary["train_customers"], "test": summary["test_customers"]},
        "models": {"chosen": summary["selection"]["chosen"], "response": summary["selection"]["best_response"]},
        "economics": {"margin": economics["margin"], "cost_per_email": economics["cost_per_email"]},
        "cost_sweep": sweep,
        "curves": {"spend": _curves(summary, "spend"), "visits": _curves(summary["visits"], "visits")},
        "power": {
            key: {"detectable_per_1000": v["detectable_gain_per_1000"],
                  "cv_suggested_per_1000": v["cv_suggested_gain_per_1000"],
                  "customers_needed": v["test_customers_needed"],
                  "times_test_set": v["times_current_test_set"]}
            for key, v in summary["power"].items()
        },
        "deciles": deciles,
        "everyone": everyone,
        "profile": profile,
    }


def build_criteo() -> dict:
    """The replication on Criteo's ad experiments: everything in results/criteo/.

    Treatment there is slightly predictable from the features, so every test-set estimate
    exists twice: as randomized, and reweighted by each user's estimated chance of
    treatment. Both are exported, and the page can show either.
    """
    folder = config.CRITEO_RESULTS_DIR
    summary = _read_json("summary.json", folder)
    experiment = _read_json("experiment.json", folder)
    selection = summary["selection"]
    # The page calls the chosen model the best one trained on conversions.
    _check(selection["chosen"] == selection["best_conversion"],
           "the chosen Criteo model must be the best one trained on conversions")

    rows = _read_csv("uplift_curves_test_conversion.csv", folder)
    fractions = [float(r["fraction_targeted"]) for r in rows]
    _uniform_grid(fractions, "fraction_targeted in the Criteo curves")
    _check(fractions[0] == 0 and fractions[-1] == 1, "the Criteo curves must run from 0% to 100% of users")
    step = fractions[1] - fractions[0]

    deciles = _columns(_read_csv("deciles.csv", folder))
    _check(deciles["decile"] == [float(d) for d in range(1, 11)], "results/criteo/deciles.csv must hold tenths 1 to 10")

    estimators = {}
    for estimator, suffix in (("as_randomized", ""), ("reweighted", "_reweighted")):
        block = summary["conversion"][estimator]
        full = block["full_gain"]
        curve = {key: [float(r[f"chosen{suffix}{end}"]) for r in rows]
                 for key, end in (("gain", ""), ("lo", "_lo"), ("hi", "_hi"))}
        _check(abs(curve["gain"][-1] - full) < TOLERANCE,
               f"the chosen model's {estimator} curve must end at the effect of treating everyone")
        budgets = {}
        for key, v in block["models"]["chosen"].items():
            if not key.endswith("pct"):
                continue
            budget = int(key.removesuffix("pct")) / 100
            # The page draws random targeting as budget x the effect of treating everyone,
            # so the paired gain over random must agree with the curve minus that line.
            gap = curve["gain"][round(budget / step)] - budget * full
            _check(abs(gap - v["vs_random"]) < TOLERANCE,
                   f"the {estimator} gain over random at {key} doesn't match the curve in summary.json")
            budgets[key] = {"budget": budget, "vs_random": v["vs_random"], "ci": v["vs_random_ci"],
                            "share": v["share_of_full_gain"], "share_ci": v["share_of_full_gain_ci"]}
        power = summary["power"][estimator]
        estimators[estimator] = {
            "full_gain": full,
            "full_gain_ci": block["full_gain_ci"],
            "visit_full_gain": summary["visit"][estimator]["full_gain"],
            "curve": curve,
            "budgets": budgets,
            "models": {
                role: {"area": m["area_over_random"], "area_ci": m["area_ci"],
                       **{key: {"vs_random": v["vs_random"], "ci": v["vs_random_ci"]}
                          for key, v in m.items() if key.endswith("pct")}}
                for role, m in block["models"].items()
            },
            "differences": block["differences"],
            "power": {
                key: {"detectable": v["detectable_gain"], "cv_suggested": v["cv_suggested_gain"],
                      "users_needed": v["test_users_needed"], "times_test_set": v["times_current_test_set"]}
                for key, v in power["chosen_vs_random"].items()
            },
            "detectable_difference": {
                key: v["detectable_difference"] for key, v in power["best_visits_vs_best_conversion"].items()
            },
            "deciles": {key: deciles[f"conversion_uplift{suffix}{end}_per_100k"]
                        for key, end in (("uplift", ""), ("lo", "_lo"), ("hi", "_hi"))},
        }

    check = experiment["propensity_check"]
    quantiles = check["propensity_quantiles"]
    return {
        "note": "Written by python -m uplift_targeting.demo from results/criteo/. Don't edit by hand.",
        "users": experiment["users"],
        "treated_share": experiment["treated_share"],
        "conversion_rate": experiment["conversion_rate"],
        "visit_rate": experiment["visit_rate"],
        "effects": experiment["effects"],  # all users, as randomized
        "max_smd": experiment["max_abs_standardized_mean_difference"],
        "propensity": {"auc": check["auc"], "auc_null_se": check["auc_null_se"], "min": quantiles["0"],
                       "p01": quantiles["0.01"], "p99": quantiles["0.99"], "max": quantiles["1"]},
        "split": summary["users"],
        "candidates": len(_read_csv("cv_comparison.csv", folder)),
        "train_conversions": summary["outcome_counts"]["train"]["conversion"],
        "bootstrap_resamples": summary["bootstrap"]["resamples"],
        "models": selection,
        "fractions": fractions,
        "deciles": {key: deciles[key] for key in ("decile", "users", "treated_share", "control_conversion_rate")},
        "estimators": estimators,
    }


# One data file, and one section of the page (docs/js/<name>.js), per dataset.
EXPORTS = {"hillstrom": build_hillstrom, "criteo": build_criteo}


def output_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in EXPORTS.items():
        path = output_path(name)
        # allow_nan=False: a browser can't parse NaN or Infinity, so fail here instead.
        text = json.dumps(build(), separators=(",", ":"), allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
        print(f"Demo data -> {path.relative_to(config.ROOT).as_posix()} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
