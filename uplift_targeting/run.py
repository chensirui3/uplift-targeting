"""Run the whole project end to end: python -m uplift_targeting.run

1. Download and verify the data; check the randomization; estimate average effects.
2. Split 60/40. Everything below up to step 4 uses the training set only.
3. Cross-validate 18 candidate models and pick one with a rule fixed in advance.
4. Evaluate the chosen model, and the baselines it competes with, once on the test set.
5. Write every number to results/ and every chart to figures/.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import analysis, config, data, modeling
from .learners import scores


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _write_json(name: str, payload) -> None:
    (config.RESULTS_DIR / name).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def main() -> None:
    start = time.time()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)

    df = data.load(data.download())
    experiment = analysis.experiment_summary(df)
    _write_json("experiment.json", experiment)
    print(f"Loaded {len(df):,} customers; max |SMD| = {experiment['max_abs_standardized_mean_difference']:.3f}")

    train, test = data.split(df)

    # --- Model selection on the training set only -----------------------------------
    cv_table, oof = modeling.cross_validate(train)
    cv_table.round(6).to_csv(config.RESULTS_DIR / "cv_comparison.csv", index_label="candidate")
    picks = modeling.select(cv_table)
    _write_json("selection.json", {"rule": modeling.SELECTION_RULE, **picks})
    print("Chosen:", picks["chosen"])

    # --- One-time evaluation on the test set ----------------------------------------
    X_test, _, _ = data.arrays(test)
    fitted = {role: modeling.fit_by_label(lbl, train) for role, lbl in picks.items()}
    test_scores = {role: scores(model, X_test) for role, model in fitted.items()}
    rule = analysis.surrogate_rule(train, fitted["chosen"], test)
    test_scores["rule"] = rule["score"]
    (config.RESULTS_DIR / "rule.txt").write_text(rule["text"])

    budget = analysis.budget_analysis(test, test_scores)
    visits = analysis.budget_analysis(test, test_scores, outcome="visit")
    power = analysis.power_analysis(train, test, cv_table, picks["chosen"], budget)
    costs = np.round(np.arange(0, 0.501, 0.01), 2)
    sweep = analysis.cost_sweep(train, oof[picks["chosen"]], test, test_scores["chosen"], costs)
    sweep.round(4).to_csv(config.RESULTS_DIR / "cost_sweep.csv", index=False)
    dec = analysis.deciles(test, test_scores["chosen"])
    dec.round(5).to_csv(config.RESULTS_DIR / "deciles.csv", index=False)
    prof = analysis.profile(test, test_scores["chosen"])
    prof.round(4).to_csv(config.RESULTS_DIR / "profile.csv")
    choice = analysis.which_email(test, test_scores["chosen"])
    choice["segments"].round(5).to_csv(config.RESULTS_DIR / "which_email.csv", index=False)

    for name, result in (("spend", budget), ("visits", visits)):
        curves = pd.DataFrame({"fraction_emailed": result["fractions"]})
        for role, entry in result["models"].items():
            curves[f"{role}"] = np.array(entry["curve"]) * analysis.PER_1000
            curves[f"{role}_lo"] = np.array(entry["curve_lo"]) * analysis.PER_1000
            curves[f"{role}_hi"] = np.array(entry["curve_hi"]) * analysis.PER_1000
        curves.round(4).to_csv(
            config.RESULTS_DIR / f"uplift_curves_test_{name}.csv", index=False
        )
    # Cross-validated (training set) curves of the same models, for the CV-vs-test chart.
    cv_curves = pd.DataFrame({"fraction_emailed": modeling.FRACTIONS})
    for role, lbl in picks.items():
        cv_curves[role] = modeling.mens_curve(train, oof[lbl]) * analysis.PER_1000
    cv_curves.round(4).to_csv(config.RESULTS_DIR / "uplift_curves_cv_spend.csv", index=False)

    summary = {
        "selection": picks,
        "labels": {role: picks.get(role, "readable rule (depth-2 tree)") for role in test_scores},
        "test_customers": int(len(test)),
        "full_email_gain_per_1000": budget["full_email_gain_per_1000"],
        "full_email_gain_ci_per_1000": budget["full_email_gain_ci_per_1000"],
        "budgets": {
            role: {k: v for k, v in entry.items() if k not in ("curve", "curve_lo", "curve_hi")}
            for role, entry in budget["models"].items()
        },
        "differences": budget["differences"],
        "visits": {
            "full_email_gain_per_1000": visits["full_email_gain_per_1000"],
            "full_email_gain_ci_per_1000": visits["full_email_gain_ci_per_1000"],
            "budgets": {
                role: {k: v for k, v in entry.items() if k not in ("curve", "curve_lo", "curve_hi")}
                for role, entry in visits["models"].items()
            },
            "differences": visits["differences"],
        },
        "power": power,
        "rule_fidelity_corr": rule["fidelity_corr"],
        "model_choice_vs_always_mens": choice["model_choice"],
        "dollars_per_visit": fitted["chosen"].dollars_per_visit_.tolist()
        if hasattr(fitted["chosen"], "dollars_per_visit_")
        else None,
        "economics": {"margin": config.MARGIN, "cost_per_email": config.COST_PER_EMAIL},
        "email_everyone": analysis.blanket_economics(experiment),
        "train_customers": int(len(train)),
        "default_cost_row": sweep[np.isclose(sweep["cost_per_email"], config.COST_PER_EMAIL)]
        .iloc[0]
        .to_dict(),
    }
    _write_json("summary.json", summary)

    try:
        from . import plots
    except ImportError:  # plotting is optional for the numbers
        plots = None
    if plots is not None:
        plots.make_all()
    print(f"Done in {time.time() - start:.0f}s. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
