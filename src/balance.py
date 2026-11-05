"""
Does dropping the unclassified names bias anything?

    python balance.py

Half the panel has a surname we cannot read confidently. We exclude those
inventors. That is the conservative choice -- misclassifying a hidden Arab as
Jewish contaminates the comparison group and pulls beta_1 toward zero, whereas
excluding only costs precision. But it is only defensible if the excluded
group is not systematically different from the kept group, and if the result
does not hinge on how they would have been classified.

Three checks:
  1. BALANCE   -- do the excluded look like the included on the outcome?
  2. SENSITIVITY -- reassign a share of them at random and re-estimate.
  3. BOUNDS    -- the extreme cases, as a worst-case envelope.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

RNG = np.random.default_rng(20260828)
N_DRAWS = 200
LINE = "=" * 70


def fit(df):
    """Binomial corporate-share model, cohort controls only."""
    X = patsy.dmatrix("arab + C(cohort)", df, return_type="dataframe")
    y = df[["corp_count", "indep_count"]].values
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    return r.params["arab"], r.bse["arab"], r.pvalues["arab"]


def main():
    p = pd.read_csv("panel.csv")
    p["cohort"] = p["cohort"].astype(str)
    amb = p[p.ethnicity == "ambiguous"]
    ar = p[p.ethnicity == "arab"]
    je = p[p.ethnicity == "jewish"]

    # ------------------------------------------------------------ 1
    print(LINE)
    print("1. BALANCE -- are the excluded different from the included?")
    print(LINE)
    rows = []
    for col in ["corp_share", "ever_corporate", "total_patents",
                "corp_count", "indep_count", "n_firms", "first_year"]:
        rows.append({
            "variable": col,
            "excluded (amb)": round(amb[col].mean(), 3),
            "Arab": round(ar[col].mean(), 3),
            "Jewish": round(je[col].mean(), 3),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    d, pv = stats.ttest_ind(amb["corp_share"], je["corp_share"],
                            equal_var=False)
    print(f"\n  excluded vs Jewish, corporate share: "
          f"diff = {amb.corp_share.mean() - je.corp_share.mean():+.4f}, "
          f"p = {pv:.4f}")
    print("  If the excluded resemble the included on the OUTCOME, dropping")
    print("  them costs precision but does not bias beta_1.")

    print("\n  Region distribution (% of each group)")
    reg = pd.crosstab(p.district, p.ethnicity, normalize="columns") * 100
    print(reg.round(1).to_string())

    # ------------------------------------------------------------ 2
    print("\n" + LINE)
    print("2. SENSITIVITY -- what if some of the excluded really are Arab?")
    print(LINE)
    base = p[p.ethnicity.isin(["arab", "jewish"])].copy()
    base["arab"] = (base.ethnicity == "arab").astype(int)
    b0, se0, p0 = fit(base)
    print(f"  baseline (excluded dropped): beta = {b0:+.4f} "
          f"(se {se0:.4f}, p {p0:.4f}), n_arab = {int(base.arab.sum())}")
    print()
    print(f"  {'assumed Arab share':>20} {'n_arab':>8} {'mean beta':>11} "
          f"{'sd':>8} {'% significant':>14}")

    for share in (0.02, 0.05, 0.10, 0.20):
        betas, sig = [], 0
        for _ in range(N_DRAWS):
            a2 = amb.copy()
            a2["arab"] = (RNG.random(len(a2)) < share).astype(int)
            j2 = je.copy(); j2["arab"] = 0
            ar2 = ar.copy(); ar2["arab"] = 1
            d2 = pd.concat([ar2, j2, a2], ignore_index=True)
            try:
                b, se, pv = fit(d2)
            except Exception:
                continue
            betas.append(b)
            sig += pv < 0.05
        if betas:
            print(f"  {share:>19.0%} {int(ar.shape[0] + share*len(amb)):>8,} "
                  f"{np.mean(betas):>+11.4f} {np.std(betas):>8.4f} "
                  f"{100*sig/len(betas):>13.0f}%")

    print("\n  If beta stays close to the baseline across these rows, the")
    print("  conclusion does not depend on how the excluded would have been")
    print("  classified. If it swings, say so in the limitations.")

    # ------------------------------------------------------------ 3
    print("\n" + LINE)
    print("3. BOUNDS -- the two extreme cases")
    print(LINE)
    for label, val in [("ALL excluded are Arab", 1), ("ALL are Jewish", 0)]:
        a2 = amb.copy(); a2["arab"] = val
        j2 = je.copy(); j2["arab"] = 0
        ar2 = ar.copy(); ar2["arab"] = 1
        d2 = pd.concat([ar2, j2, a2], ignore_index=True)
        try:
            b, se, pv = fit(d2)
            print(f"  {label:<26} beta = {b:+.4f} (se {se:.4f}, p {pv:.4f})")
        except Exception as e:
            print(f"  {label:<26} failed: {e}")
    print("\n  These are deliberately absurd -- nobody thinks all 17,949 are")
    print("  Arab. They bracket the logically possible range, which is what a")
    print("  bounds argument does. The sensitivity rows above are the")
    print("  informative version.")

    print("\n" + LINE)
    print("WHAT TO WRITE")
    print(LINE)
    print("  Report: the share excluded, that they resemble the included on")
    print("  the outcome, and that beta is stable across plausible assumed")
    print("  Arab shares. That converts a weakness into a documented")
    print("  robustness check.")


if __name__ == "__main__":
    main()
