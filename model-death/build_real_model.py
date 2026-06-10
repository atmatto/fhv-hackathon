"""
Train the death-LEARNED model: weights come from who actually died of CVD in
the linked NHANES mortality data, NOT borrowed from the column model.

It trains three models on the pooled 2007-2018 cohort and prints them side by
side, so you can see what the borrowed weights were worth:

  1. Cox (death-learned)  - fits its own coefficient on every feature.
  2. Cox (importance prior) - the borrowed-weights model, for comparison.
  3. Random Survival Forest - non-linear death-learned model.

It also prints the death-learned hazard ratios next to the column model's
importance rank, so you can see where "what predicts dying of CVD" disagrees
with "what predicts currently having CVD", and a rough calibration check.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

from lifelines.utils import concordance_index

from column_model_spec import FEATURES, IMPORTANCE, cvd_risk_index
from data_loading import FEATURE_COLS
from model import CoxModel, RSFModel, CompetingRisksModel
from mortality_cohort import download, build_pooled_frame, DATA_DIR
from calibration import cif_calibration, ici

# Lipid trio is perfectly collinear (non_hdl = total_chol - hdl); drop one for
# the linear Cox fit. Trees (RSF) handle collinearity, so they keep all 27.
COX_DROP = ["non_hdl"]


def _usable_cox_features(train: pd.DataFrame) -> list[str]:
    """Features with real variance (drop near-constant heavily-imputed labs)."""
    feats = [f for f in FEATURE_COLS if f not in COX_DROP]
    keep = [f for f in feats if train[f].std(skipna=True) > 1e-6]
    dropped = sorted(set(feats) - set(keep))
    if dropped:
        print(f"   (dropped near-constant columns from Cox: {dropped})")
    return keep


def main():
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Downloading NHANES cohort + mortality files (first run)...")
        download()

    print("\nBuilding pooled survival frame:")
    df = build_pooled_frame()
    print(f"\nPOOLED: n={len(df)}, CVD deaths={int(df['event'].sum())} "
          f"({df['event'].mean():.1%}), "
          f"median follow-up={df['time_years'].median():.1f} y\n")

    # Stratified split keeps the death rate equal in train/test.
    test_idx = (df.groupby("event", group_keys=False)
                  .apply(lambda g: g.sample(frac=0.25, random_state=1)).index)
    test = df.loc[test_idx]
    train = df.drop(test_idx)

    # Two baselines: the borrowed weights, and the single strongest factor.
    # Borrowed index is reported by its raw RANKING ability (concordance of the
    # index itself), not a refit single-covariate Cox -- the index is heavy-
    # tailed (uacr) and unstable as a lone covariate, which understates it.
    test_idx_score = np.array([cvd_risk_index(r)
                               for r in test[FEATURE_COLS].to_dict("records")])
    c_prior = concordance_index(test["time_years"], -test_idx_score, test["event"])
    c_age = concordance_index(test["time_years"], -test["age"].values,
                              test["event"])

    # 1. Death-learned Cox -------------------------------------------------
    # penalizer=0.01: enough ridge to stabilize the (cleaned) feature set
    # without the over-shrinkage that flattened high-risk predictions at 0.1.
    cox_feats = _usable_cox_features(train)
    cox = CoxModel(penalizer=0.01, use_importance_index=False,
                   features=cox_feats).fit(train)
    c_cox = cox.concordance(test)

    # 3. Random Survival Forest -------------------------------------------
    rsf = RSFModel(n_estimators=200, min_samples_leaf=50).fit(train)
    c_rsf = rsf.concordance(test)

    # 4. Competing-risks model (the one we ship) --------------------------
    cr = CompetingRisksModel(penalizer=0.01, features=cox_feats).fit(train)
    c_cr = cr.concordance(test)

    print("C-index on held-out test (higher = ranks who-dies-sooner better):")
    print(f"   borrowed importance index (column model)  : {c_prior:.3f}")
    print(f"   age alone (baseline)                       : {c_age:.3f}")
    print(f"   Cox, death-learned ({len(cox_feats)} features)        : {c_cox:.3f}")
    print(f"   Random Survival Forest, death-learned      : {c_rsf:.3f}")
    print(f"   Competing-risks (CIF), death-learned       : {c_cr:.3f}")
    times = [5, 8, 10, 12]
    auc, mean_auc = rsf.time_dependent_auc(train, test, times)
    print("\n   RSF time-dependent AUC: "
          + ", ".join(f"{t}y={a:.2f}" for t, a in zip(times, auc))
          + f" (mean {mean_auc:.2f})")

    # --- What death says matters vs what the column model said -----------
    # Per-SD hazard ratio = exp(coef * SD): the effect of a typical (1 SD)
    # change, comparable across features with different units (years vs 0/1).
    print("\nWhat the DEATH data weights vs the column model's rank")
    print("(death HR is per 1 SD; >1 raises CVD-death risk, <1 lowers it):")
    coef = np.log(cox.hazard_ratios())
    sd = train[cox_feats].std()
    per_sd = {f: float(np.exp(coef.get(f, 0.0) * sd[f])) for f in cox_feats}
    imp_rank = {f: i + 1 for i, f in enumerate(FEATURES)}
    rows = sorted(cox_feats, key=lambda f: abs(np.log(per_sd[f])), reverse=True)
    print(f"   {'feature':<14}{'col-model rank':>15}{'death HR/SD':>13}")
    for f in rows[:12]:
        arrow = "risk+" if per_sd[f] > 1 else "risk-"
        print(f"   {f:<14}{imp_rank[f]:>15}{per_sd[f]:>13.2f}  {arrow}")

    # --- Competing risks corrects the OVERSTATEMENT ----------------------
    # Cause-specific 1-S(t) ignores that you may die of something else first.
    # The CIF accounts for it, so it is lower where competing mortality is high.
    print("\nCause-specific risk (overstates) vs competing-risks CIF, 10y:")
    demo = {"45yo, healthy": dict(age=45, sbp=115, htn_dx=0, diabetes_dx=0),
            "80yo, high-risk": dict(age=80, sbp=170, htn_dx=1, diabetes_dx=1,
                                    creatinine=1.8, bun=35)}
    for label, p in demo.items():
        cs = cox.risk_by_horizon(p, (10,))[10]
        ci = cr.risk_by_horizon(p, (10,))[10]
        print(f"   {label:<16} cause-specific={cs:5.1%}   CIF={ci:5.1%}")

    # --- Calibration of the shipped competing-risks model ----------------
    # Observed = Aalen-Johansen CIF (the correct estimator under competing
    # risks; KM would overstate it, just as cause-specific prediction does).
    print("\nAbsolute-risk calibration of the competing-risks model")
    print("(predicted CIF vs Aalen-Johansen observed; ICI = mean gap):")
    for h in (5, 10):
        rows = cif_calibration(cr, test, h, bins=10)
        print(f"   {h}y ICI = {ici(rows):.3f}")
    print()
    rows = cif_calibration(cr, test, 10, bins=10)
    print("   10y calibration (predicted -> AJ observed):")
    for i, (p, o, n) in enumerate(rows, 1):
        print(f"     bin {i:>2}: predicted {p:5.1%}   observed {o:5.1%}   (n={n})")

    data = {
        'cph_cvd': getattr(cr, 'cph_cvd'),
        'cph_other': getattr(cr, 'cph_other'),
        'features': getattr(cr, 'features'),
        '_grid': getattr(cr, '_grid'),
        '_H1_0': getattr(cr, '_H1_0'),
        '_H2_0': getattr(cr, '_H2_0'),
        '_dH1_0': getattr(cr, '_dH1_0'),
    }

    joblib.dump(data, "cvd_death_model.joblib")
    print("\nSaved competing-risks death model -> cvd_death_model.joblib")


if __name__ == "__main__":
    main()
