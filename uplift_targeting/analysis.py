"""Experiment checks and the one-time evaluation on the held-out test set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from . import config
from . import evaluation as ev
from .data import FEATURES, arrays
from .modeling import BUDGETS, FRACTIONS, mens_curve, mens_pair
from .policy import top_fraction_policy

PER_1000 = 1000  # report money per 1,000 customers


def _diff_ci(t: pd.Series, c: pd.Series) -> tuple[float, float, float]:
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    return float(diff), float(diff - 1.96 * se), float(diff + 1.96 * se)


def experiment_summary(df: pd.DataFrame) -> dict:
    """Arm sizes, covariate balance and average effects on all 64,000 customers."""
    control = df[df["arm"] == 0]
    arms = {}
    for a, name in enumerate(config.ARM_LABELS):
        g = df[df["arm"] == a]
        entry = {
            "n": int(len(g)),
            "visit_rate": float(g["visit"].mean()),
            "conversion_rate": float(g["conversion"].mean()),
            "spend_per_customer": float(g["spend"].mean()),
        }
        if a > 0:
            for y in ("visit", "conversion", "spend"):
                diff, lo, hi = _diff_ci(g[y], control[y])
                entry[f"{y}_effect"] = diff
                entry[f"{y}_ci"] = [lo, hi]
                entry[f"{y}_relative"] = diff / float(control[y].mean())
        arms[name] = entry
    smd = []
    for f in FEATURES:
        for a in (1, 2):
            t, c = df.loc[df["arm"] == a, f], control[f]
            pooled = np.sqrt((t.var() + c.var()) / 2)
            smd.append(abs(t.mean() - c.mean()) / pooled)
    buyers = df[df["conversion"] == 1]
    return {
        "n_customers": int(len(df)),
        "arms": arms,
        "max_abs_standardized_mean_difference": float(max(smd)),
        "purchase_rate": float(df["conversion"].mean()),
        "visit_rate": float(df["visit"].mean()),
        "average_order": float(buyers["spend"].mean()),
    }


def blanket_economics(experiment: dict, costs=(0.05, 0.10, 0.20), margin: float = config.MARGIN) -> dict:
    """Profit of emailing everyone the men's email, from the full experiment's effect.
    No model is involved, so all 64,000 customers can be used."""
    mens = experiment["arms"][config.ARM_LABELS[1]]
    effect, (lo, hi) = mens["spend_effect"], mens["spend_ci"]
    return {
        "margin": margin,
        "breakeven_cost": margin * effect,
        "breakeven_cost_ci": [margin * lo, margin * hi],
        "profit_per_1000": {
            f"{c:.2f}": {
                "estimate": (margin * effect - c) * PER_1000,
                "ci": [(margin * lo - c) * PER_1000, (margin * hi - c) * PER_1000],
            }
            for c in costs
        },
    }


def budget_analysis(
    test: pd.DataFrame, test_scores: dict[str, np.ndarray], outcome: str = "spend"
) -> dict:
    """Send the men's email to the top X% by each model. Paired bootstrap intervals.
    Amounts are per 1,000 customers: dollars for spend, extra visits for visits."""
    pair = mens_pair(test)
    y = test[outcome].to_numpy(dtype=float)[pair]
    treated = test["arm"].to_numpy()[pair] == 1
    curves, boots = {}, {}
    for name, s in test_scores.items():
        curves[name] = mens_curve(test, s, outcome)
        boots[name] = ev.bootstrap_uplift_curve(
            y, treated, s[pair, 0], FRACTIONS, config.N_BOOTSTRAP, config.SEED
        )
    ate = float(y[treated].mean() - y[~treated].mean())
    ate_boot = next(iter(boots.values()))[:, -1]  # same resamples for every model
    random_boot = FRACTIONS[None, :] * ate_boot[:, None]

    def at(curve_or_boot, b):
        idx = int(np.argmin(np.abs(FRACTIONS - b)))
        return curve_or_boot[..., idx]

    models = {}
    for name in test_scores:
        lo, hi = ev.interval(boots[name])
        entry = {
            "area_over_random": ev.area_over_random(FRACTIONS, curves[name]),
            "area_ci": list(
                ev.interval(
                    np.array([ev.area_over_random(FRACTIONS, r) for r in boots[name]])
                )
            ),
            "curve": curves[name].tolist(),
            "curve_lo": lo.tolist(),
            "curve_hi": hi.tolist(),
        }
        for b in BUDGETS:
            key = f"{int(b * 100)}pct"
            gain = float(at(curves[name], b))
            vs_random = at(boots[name], b) - at(random_boot, b)
            entry[key] = {
                "gain_per_1000": gain * PER_1000,
                "gain_ci_per_1000": [float(v) * PER_1000 for v in ev.interval(at(boots[name], b))],
                "share_of_full_gain": gain / ate,
                "vs_random_per_1000": float(gain - b * ate) * PER_1000,
                "vs_random_ci_per_1000": [float(v) * PER_1000 for v in ev.interval(vs_random)],
            }
        models[name] = entry
    # Paired differences between models at each budget.
    names = list(test_scores)
    pairs = {}
    for i, a in enumerate(names):
        for b_name in names[i + 1 :]:
            for b in BUDGETS:
                d = at(boots[a], b) - at(boots[b_name], b)
                point = float(at(curves[a], b) - at(curves[b_name], b))
                pairs[f"{a} minus {b_name} @ {int(b * 100)}pct"] = {
                    "per_1000": point * PER_1000,
                    "ci_per_1000": [float(v) * PER_1000 for v in ev.interval(d)],
                    "share_of_resamples_above_zero": float((d > 0).mean()),
                }
    return {
        "full_email_gain_per_1000": ate * PER_1000,
        "full_email_gain_ci_per_1000": [float(v) * PER_1000 for v in ev.interval(ate_boot)],
        "fractions": FRACTIONS.tolist(),
        "models": models,
        "differences": pairs,
    }


def power_analysis(train: pd.DataFrame, test: pd.DataFrame, cv_table: pd.DataFrame,
                   chosen: str, budget: dict) -> dict:
    """How big a gain over random targeting could this test set detect, and how many
    customers would it take to detect the gain cross-validation suggested?

    The standard error comes from the paired bootstrap on the test set. Detectable
    means 80% power with a two-sided 5% test: 2.8 standard errors.
    """
    pair = mens_pair(train)
    spend = train["spend"].to_numpy(dtype=float)[pair]
    treated = train["arm"].to_numpy()[pair] == 1
    train_ate = spend[treated].mean() - spend[~treated].mean()
    out = {}
    for b in BUDGETS:
        key = f"{int(b * 100)}pct"
        lo, hi = budget["models"]["chosen"][key]["vs_random_ci_per_1000"]
        se = (hi - lo) / (2 * 1.96)
        detectable = 2.8 * se
        cv_gain = (cv_table.loc[chosen, f"gain_at_{key}"] - b * train_ate) * PER_1000
        multiplier = (detectable / cv_gain) ** 2 if cv_gain > 0 else float("inf")
        out[key] = {
            "standard_error_per_1000": se,
            "detectable_gain_per_1000": detectable,
            "cv_suggested_gain_per_1000": cv_gain,
            "test_customers_needed": multiplier * len(test),
            "times_current_test_set": multiplier,
        }
    return out


def cost_sweep(
    train: pd.DataFrame,
    train_oof_score: np.ndarray,
    test: pd.DataFrame,
    test_score: np.ndarray,
    costs: np.ndarray,
    margin: float = config.MARGIN,
) -> pd.DataFrame:
    """For each cost per email, pick how many customers to email using the *training*
    out-of-fold curve, then measure the profit of that decision on the test set."""
    train_curve = mens_curve(train, train_oof_score)
    test_curve = mens_curve(test, test_score)
    pair = mens_pair(test)
    y = test["spend"].to_numpy(dtype=float)[pair]
    treated = test["arm"].to_numpy()[pair] == 1
    boot = ev.bootstrap_uplift_curve(
        y, treated, test_score[pair, 0], FRACTIONS, config.N_BOOTSTRAP, config.SEED
    )
    rows = []
    for c in costs:
        idx = int(np.argmax(margin * train_curve - c * FRACTIONS))
        f = FRACTIONS[idx]
        targeted = margin * test_curve[idx] - c * f
        blanket = margin * test_curve[-1] - c
        targeted_boot = margin * boot[:, idx] - c * f
        blanket_boot = margin * boot[:, -1] - c
        rows.append(
            {
                "cost_per_email": float(c),
                "share_emailed": float(f),
                "targeted_per_1000": targeted * PER_1000,
                "targeted_lo": float(np.quantile(targeted_boot, 0.025)) * PER_1000,
                "targeted_hi": float(np.quantile(targeted_boot, 0.975)) * PER_1000,
                "blanket_per_1000": blanket * PER_1000,
                "blanket_lo": float(np.quantile(blanket_boot, 0.025)) * PER_1000,
                "blanket_hi": float(np.quantile(blanket_boot, 0.975)) * PER_1000,
                "best_of_blanket_or_nothing_per_1000": max(blanket, 0.0) * PER_1000,
                "targeted_minus_best_simple_per_1000": (targeted - max(blanket, 0.0)) * PER_1000,
                "diff_lo": float(np.quantile(targeted_boot - np.maximum(blanket_boot, 0), 0.025))
                * PER_1000,
                "diff_hi": float(np.quantile(targeted_boot - np.maximum(blanket_boot, 0), 0.975))
                * PER_1000,
            }
        )
    return pd.DataFrame(rows)


def deciles(test: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Observed men's-email effect within each tenth of customers, ranked by the model."""
    rank = pd.qcut(-score[:, 0], 10, labels=False) + 1  # 1 = highest predicted uplift
    pair = mens_pair(test)
    rows = []
    for d in range(1, 11):
        g = test[(rank == d) & pair]
        t, c = g[g["arm"] == 1], g[g["arm"] == 0]
        spend, lo, hi = _diff_ci(t["spend"], c["spend"])
        visit, vlo, vhi = _diff_ci(t["visit"], c["visit"])
        rows.append(
            {
                "decile": d,
                "customers": int((rank == d).sum()),
                "predicted_spend_uplift": float(score[rank == d, 0].mean()),
                "observed_spend_uplift": spend,
                "spend_lo": lo,
                "spend_hi": hi,
                "observed_visit_uplift": visit,
                "visit_lo": vlo,
                "visit_hi": vhi,
                "baseline_spend": float(c["spend"].mean()),
            }
        )
    return pd.DataFrame(rows)


def profile(test: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Who is in the model's top and bottom deciles?"""
    rank = pd.qcut(-score[:, 0], 10, labels=False) + 1

    def describe(g: pd.DataFrame) -> dict:
        return {
            "customers": len(g),
            "new customer (last 12 months)": g["newbie"].mean(),
            "bought men's and women's": ((g["mens"] == 1) & (g["womens"] == 1)).mean(),
            "bought men's only": ((g["mens"] == 1) & (g["womens"] == 0)).mean(),
            "bought women's only": ((g["mens"] == 0) & (g["womens"] == 1)).mean(),
            "months since last purchase (median)": g["recency"].median(),
            "spent last year (median $)": g["history"].median(),
            "buys on the web": g["channel_web"].mean(),
            "buys by phone": 1 - g["channel_web"].mean() - g["channel_multichannel"].mean(),
            "buys on several channels": g["channel_multichannel"].mean(),
            "urban": g["zip_urban"].mean(),
            "rural": g["zip_rural"].mean(),
        }

    return pd.DataFrame(
        {
            "Top 10% (most persuadable)": describe(test[rank == 1]),
            "Everyone": describe(test),
            "Bottom 10%": describe(test[rank == 10]),
        }
    )


def which_email(test: pd.DataFrame, chosen_score: np.ndarray) -> dict:
    """Is the women's email ever the better choice?"""
    segments = {
        "bought men's only": (test["mens"] == 1) & (test["womens"] == 0),
        "bought women's only": (test["mens"] == 0) & (test["womens"] == 1),
        "bought both": (test["mens"] == 1) & (test["womens"] == 1),
        "everyone": pd.Series(True, index=test.index),
    }
    rows = []
    for seg, mask in segments.items():
        g = test[mask]
        c = g[g["arm"] == 0]
        for a in (1, 2):
            t = g[g["arm"] == a]
            for y in ("spend", "visit"):
                diff, lo, hi = _diff_ci(t[y], c[y])
                rows.append(
                    {"segment": seg, "email": config.ARM_LABELS[a], "outcome": y,
                     "effect": diff, "lo": lo, "hi": hi, "customers": int(mask.sum())}
                )
    # Let the model pick the email for each customer versus always sending the men's.
    _, arm, Y = arrays(test)
    spend = Y["spend"].to_numpy(dtype=float)
    baseline = spend[arm == 0].mean()
    mens_only = np.column_stack([chosen_score[:, 0], np.full(len(test), -np.inf)])
    choice = {}
    for b in BUDGETS:
        picked = top_fraction_policy(chosen_score, b)
        men = top_fraction_policy(mens_only, b)
        choice[f"{int(b * 100)}pct"] = {
            "model_picks_email_per_1000": float(ev.policy_value(picked, arm, spend) - baseline) * PER_1000,
            "always_mens_per_1000": float(ev.policy_value(men, arm, spend) - baseline) * PER_1000,
            "share_given_womens": float((picked == 2).mean() / max(b, 1e-9)),
        }
    return {"segments": pd.DataFrame(rows), "model_choice": choice}


def surrogate_rule(train: pd.DataFrame, chosen_model, test: pd.DataFrame, depth: int = 2) -> dict:
    """Fit a shallow tree to the chosen model's predictions: a rule a marketer can read."""
    X_train, _, _ = arrays(train)
    X_test, _, _ = arrays(test)
    target = chosen_model.predict_uplift(X_train)[:, 0]
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=2000, random_state=config.SEED)
    tree.fit(X_train, target)
    rule_score = np.column_stack([tree.predict(X_test), np.full(len(test), -np.inf)])
    fidelity = float(np.corrcoef(tree.predict(X_test), chosen_model.predict_uplift(X_test)[:, 0])[0, 1])
    return {
        "text": export_text(tree, feature_names=FEATURES, decimals=2),
        "score": rule_score,
        "fidelity_corr": fidelity,
        "leaves": int(tree.get_n_leaves()),
    }
