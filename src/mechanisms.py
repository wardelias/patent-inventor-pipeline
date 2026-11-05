"""
Turning H1-H3 into testable predictions.

    python mechanisms.py

Sponsorship, risk and legitimacy are survey constructs. Patent data cannot
observe them directly -- but each implies a prediction that patent data CAN
test. This script runs one test per hypothesis.

  H1 SPONSORSHIP  If the barrier is a shortage of senior champions, the gap
                  should be SMALLER in firms that already employ Arab
                  inventors, where such champions are more likely to exist.
                  Test: interact Arab with the firm's Arab share (excluding
                  the inventor themselves, so the measure is not mechanical).

  H2 RISK         If the barrier is the personal cost of championing an
                  unproven idea, it binds hardest before a person has a track
                  record. Test: the 1 -> 2 patent transition. Among inventors
                  who patent once, who files again?

  H3 LEGITIMACY   If ideas from Arab employees get less organisational
                  support, the gap should appear in CORPORATE output but not
                  in INDEPENDENT output, where no organisation gatekeeps.
                  Test: compare the two counts directly.

Each test states up front what would COUNT AS SUPPORT, so the result cannot be
reinterpreted after the fact.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

import build_panel as bp

LINE = "=" * 70


def sig(p):
    return "*" if p < 0.01 else ("**" if p < 0.05 else "")


def load():
    p = pd.read_csv("panel.csv")
    p["cohort"] = p["cohort"].astype(str)
    d = p[p.ethnicity.isin(["arab", "jewish"])].copy()
    d["arab"] = (d.ethnicity == "arab").astype(int)
    return p, d


# ===================================================== H1 SPONSORSHIP
def h1(panel, d):
    print(LINE)
    print("H1 SPONSORSHIP -- does the gap shrink where Arab inventors already")
    print("                  work?")
    print(LINE)
    print("  SUPPORT = positive, significant interaction (Arab x firm Arab")
    print("  share). Against = null or negative interaction.\n")

    raw = pd.read_csv("raw_records.csv", dtype=str)
    raw["filing_year"] = pd.to_numeric(raw["filing_year"], errors="coerce")
    eth = panel.set_index("inventor_id")["ethnicity"].to_dict()
    fy = panel.set_index("inventor_id")["first_year"].to_dict()
    raw["eth"] = raw["inventor_id"].map(eth)
    raw["first_year"] = raw["inventor_id"].map(fy)
    raw = raw[raw["eth"].isin(["arab", "jewish"])]
    raw = raw[(raw.filing_year >= raw.first_year)
              & (raw.filing_year < raw.first_year + bp.WINDOW)]
    raw["kind"] = raw.apply(bp.classify_assignee, axis=1)
    raw = raw[(raw.kind == "corporate")
              & (raw.assignee_id.astype(str).str.strip() != "")]

    # each inventor's main firm in the window
    link = (raw.groupby(["inventor_id", "assignee_id", "eth"]).size()
               .reset_index(name="k")
               .sort_values(["inventor_id", "k"], ascending=[True, False],
                            kind="mergesort")
               .drop_duplicates("inventor_id"))

    firm = link.groupby("assignee_id")["eth"].agg(
        n="size", n_arab=lambda x: (x == "arab").sum()).reset_index()
    link = link.merge(firm, on="assignee_id")
    # leave-one-out: exclude the inventor from their own firm's share
    link["peers"] = link["n"] - 1
    link["peer_arab"] = link["n_arab"] - (link["eth"] == "arab").astype(int)
    link = link[link["peers"] >= 3]
    link["firm_arab_share"] = 100 * link["peer_arab"] / link["peers"]

    m = d.merge(link[["inventor_id", "firm_arab_share", "peers"]],
                on="inventor_id")
    print(f"  inventors at firms with >=3 other inventors: {len(m):,} "
          f"(Arab: {int(m.arab.sum())})")
    if int(m.arab.sum()) < 30:
        print("  Too few Arab inventors for this test.")
        return
    print(f"  mean firm Arab share -- Arab inventors  : "
          f"{m[m.arab==1].firm_arab_share.mean():.2f}%")
    print(f"  mean firm Arab share -- Jewish inventors: "
          f"{m[m.arab==0].firm_arab_share.mean():.2f}%")

    X = patsy.dmatrix("arab * firm_arab_share + C(cohort)", m,
                      return_type="dataframe")
    y = m[["corp_count", "indep_count"]].values
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    key = [c for c in r.params.index if c.startswith("arab:")]
    print(f"\n  Arab                     {r.params['arab']:+.4f} "
          f"({r.bse['arab']:.4f})  p={r.pvalues['arab']:.3f}")
    for k in key:
        print(f"  Arab x firm Arab share   {r.params[k]:+.4f} "
              f"({r.bse[k]:.4f})  p={r.pvalues[k]:.3f} {sig(r.pvalues[k])}")
        verdict = ("SUPPORTS H1" if r.pvalues[k] < .05 and r.params[k] > 0
                   else "does not support H1")
        print(f"  -> {verdict}")


# =========================================================== H2 RISK
def h2(d):
    print("\n" + LINE)
    print("H2 RISK -- is the gap largest at career entry?")
    print(LINE)
    print("  SUPPORT = Arab inventors significantly less likely to file a")
    print("  SECOND patent, with the gap narrowing at later transitions.\n")

    d = d.copy()
    d["second"] = (d.total_patents >= 2).astype(int)
    d["third"] = (d.total_patents >= 3).astype(int)

    for col, label in [("second", "filed a 2nd patent"),
                       ("third", "filed a 3rd patent")]:
        a, j = d[d.arab == 1][col], d[d.arab == 0][col]
        t, p = stats.ttest_ind(a, j, equal_var=False)
        print(f"  {label:<22} Arab {a.mean():.3f}  Jewish {j.mean():.3f}  "
              f"diff {a.mean()-j.mean():+.4f}{sig(p)}  p={p:.3f}")

    # conditional transition: of those with >=2, who reached >=3
    sub = d[d.total_patents >= 2]
    a, j = sub[sub.arab == 1]["third"], sub[sub.arab == 0]["third"]
    if len(a) > 10:
        t, p = stats.ttest_ind(a, j, equal_var=False)
        print(f"  {'2nd -> 3rd (conditional)':<22} Arab {a.mean():.3f}  "
              f"Jewish {j.mean():.3f}  diff {a.mean()-j.mean():+.4f}"
              f"{sig(p)}  p={p:.3f}")

    X = patsy.dmatrix("arab + C(cohort) + C(district) + C(field)", d,
                      return_type="dataframe")
    r = sm.GLM(d["second"], X, family=sm.families.Binomial()).fit(cov_type="HC1")
    p = r.pvalues["arab"]
    print(f"\n  logit on 'filed a 2nd patent', full controls:")
    print(f"    Arab {r.params['arab']:+.4f} ({r.bse['arab']:.4f})  "
          f"p={p:.3f} {sig(p)}   OR={np.exp(r.params['arab']):.3f}")
    print(f"  -> {'SUPPORTS H2' if p < .05 and r.params['arab'] < 0 else 'does not support H2'}")


# ===================================================== H3 LEGITIMACY
def h3(d):
    print("\n" + LINE)
    print("H3 LEGITIMACY -- does the gap appear in corporate output but not")
    print("                 independent output?")
    print(LINE)
    print("  SUPPORT = negative Arab coefficient on CORPORATE count, null or")
    print("  positive on INDEPENDENT count. No organisation gatekeeps the")
    print("  latter, so a gap confined to corporate output points at the firm.\n")

    for col, label in [("corp_count", "corporate patents"),
                       ("indep_count", "independent patents")]:
        X = patsy.dmatrix("arab + C(cohort) + C(district) + C(field)", d,
                          return_type="dataframe")
        r = sm.GLM(d[col], X, family=sm.families.Poisson()).fit(cov_type="HC1")
        p = r.pvalues["arab"]
        print(f"  {label:<22} Arab {r.params['arab']:+.4f} "
              f"({r.bse['arab']:.4f})  p={p:.3f} {sig(p)}   "
              f"ratio={np.exp(r.params['arab']):.3f}")

    a, j = d[d.arab == 1], d[d.arab == 0]
    print(f"\n  raw means -- corporate  : Arab {a.corp_count.mean():.3f}  "
          f"Jewish {j.corp_count.mean():.3f}")
    print(f"  raw means -- independent: Arab {a.indep_count.mean():.3f}  "
          f"Jewish {j.indep_count.mean():.3f}")
    print("  -> read the two coefficients together; a gap in corporate output")
    print("     ALONE is the pattern H3 predicts.")


def main():
    panel, d = load()
    try:
        h1(panel, d)
    except FileNotFoundError:
        print("  (raw_records.csv missing -- skipping H1)")
    h2(d)
    h3(d)
    print("\n" + LINE)
    print("These three tests give Section IV real content. Report each")
    print("verdict plainly, including the ones that go against the hypothesis.")
    print(LINE)


if __name__ == "__main__":
    main()
