"""
Step 3d -- does the conclusion survive a different definition of "Arab"?

    python robustness.py

Runs the main models under three independent measures:

  A. NAME       surname lexicon (current). High precision, ~23% recall.
  B. LOCALITY   lives in an Arab-majority town. Near-perfect precision,
                misses Arab inventors in Haifa/Jerusalem/Tel Aviv.
  C. UNION      name OR locality. Highest recall, slightly lower precision.

Convergence across three measures with different error structures is far
stronger evidence than any one of them. Divergence tells you the result is
being driven by measurement rather than by the world.

Also audits the inventors classified Jewish who live in Arab towns.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

import names as nm
import regions as rg


def sig(p):
    """
    COURSE CONVENTION (see the assignment's table-format handout):
        *  p < 0.01
        ** p < 0.05
    This is inverted relative to the usual economics convention. All three
    analysis scripts must agree, or tables in the same paper will contradict
    each other -- p=0.094 was previously printing "*", which under this
    convention reads as p<0.01.
    """
    if p < 0.01:
        return "*"
    if p < 0.05:
        return "**"
    return ""


LINE = "=" * 70
SPECS = [("M0 cohort only", "C(cohort)"),
         ("M3 full        ", "C(cohort) + C(district) + C(field)")]


def fit(df, rhs, kind):
    X = patsy.dmatrix("arab + " + rhs, df, return_type="dataframe")
    if kind == "share":
        y = df[["corp_count", "indep_count"]].values
        r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    elif kind == "ever":
        r = sm.GLM(df["ever_corporate"], X,
                   family=sm.families.Binomial()).fit(cov_type="HC1")
    else:
        r = sm.GLM(df["corp_count"], X,
                   family=sm.families.Poisson()).fit(cov_type="HC1")
    return r.params["arab"], r.bse["arab"], r.pvalues["arab"]


def main():
    p = pd.read_csv("panel.csv")
    p["surname"] = p["surname"].astype(str)
    p["city"] = p["city"].astype(str)
    p["cohort"] = p["cohort"].astype(str)
    p["arab_town"] = p["city"].map(rg.is_arab_locality)

    # ---------------------------------------------------------------- audit
    print(LINE)
    print("AUDIT: inventors classified JEWISH who live in Arab towns")
    print(LINE)
    odd = p[(p.arab_town) & (p.ethnicity == "jewish")]
    print(f"  count: {len(odd):,}\n")
    print("  Which towns:")
    for c, k in odd["city"].value_counts().head(15).items():
        tot = int((p.city == c).sum())
        print(f"    {c:<26} {k:>4} of {tot:>4} inventors there")
    print("\n  Their most common surnames:")
    for s, k in odd["surname"].value_counts().head(15).items():
        print(f"    {s:<24} {k:>3}")
    print("\n  If these are ordinary Jewish surnames spread thinly, the city")
    print("  field is simply noisy. If one town dominates, that town is")
    print("  probably misclassified as Arab and should be checked.")

    # ------------------------------------------------------------ measures
    defs = {}
    a = p[p.ethnicity.isin(["arab", "jewish"])].copy()
    a["arab"] = (a.ethnicity == "arab").astype(int)
    defs["A. NAME    "] = a

    # Definition B now EXCLUDES localities with unreliable geocoding. The
    # audit found Rahat, Kalanswa, Sachnin and others carry far more
    # inventors than their size supports, with only 3-15% Arab-named,
    # against 60-75% in Nazareth and Taibe. Including them makes B a proxy
    # for peripheral residence rather than for ethnicity.
    b = p[p["city"].map(rg.geocode_reliable)].copy()
    b["arab"] = b["arab_town"].astype(int)
    defs["B. LOCALITY"] = b

    c = p.copy()
    c["arab"] = ((c.ethnicity == "arab")
                 | (c.arab_town & c["city"].map(rg.geocode_reliable))).astype(int)
    c = c[(c["arab"] == 1) | (c.ethnicity == "jewish")]
    defs["C. UNION   "] = c

    print("\n" + LINE)
    print("ROBUSTNESS: beta_arab under three definitions")
    print(LINE)
    rows = []
    for label, d in defs.items():
        for kind, kname in [("share", "corp share"), ("ever", "ever corp"),
                            ("count", "corp count")]:
            for sname, rhs in SPECS:
                try:
                    bb, se, pv = fit(d, rhs, kind)
                    if not np.isfinite(bb) or not np.isfinite(se) or abs(bb) > 20:
                        raise ValueError("separation")
                    rows.append({"definition": label, "outcome": kname,
                                 "spec": sname.strip(), "n": len(d),
                                 "n_arab": int(d["arab"].sum()),
                                 "beta": round(bb, 4), "se": round(se, 4),
                                 "p": round(pv, 4),
                                 "sig": sig(pv)})
                except Exception as e:
                    rows.append({"definition": label, "outcome": kname,
                                 "spec": sname.strip(), "n": len(d),
                                 "n_arab": int(d["arab"].sum()),
                                 "beta": np.nan, "se": np.nan, "p": np.nan,
                                 "sig": "SEPARATION" if "separation" in str(e)
                                        else type(e).__name__})
    t = pd.DataFrame(rows)
    for kname in t["outcome"].unique():
        print(f"\n--- {kname} ---")
        print(t[t.outcome == kname].drop(columns="outcome").to_string(index=False))
    t.to_csv("results_robustness.csv", index=False)

    print("\n" + LINE)
    print("READING THIS TABLE")
    print(LINE)
    print("  Same sign and rough size across A, B and C -> the result is real,")
    print("  because the three measures fail in different ways.")
    print("  Sign flips between definitions -> you are measuring your own")
    print("  classifier, not the world. Say so rather than picking a column.")
    print("\n  Wrote results_robustness.csv")


if __name__ == "__main__":
    main()
