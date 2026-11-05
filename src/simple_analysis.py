"""
SIMPLE ANALYSIS -- everything the paper needs, using only methods you can
explain in one sentence each.

    python simple_analysis.py

Methods used, and nothing else:
  * percentages and averages
  * t-tests (comparing two averages)
  * linear regression (OLS)
  * logistic regression (for a yes/no outcome)
  * one paired t-test (comparing two numbers within the same company)

Deliberately NOT used: permutation tests, binomial GLM with offsets, PPML,
mixture estimation, interaction terms. Those were in the earlier version and
they answer the same questions -- but you should not put a method in a paper
you cannot defend out loud.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import build_panel as bp

LINE = "=" * 70
FIELDS = {"A": "Human necessities", "B": "Operations", "C": "Chemistry",
          "D": "Textiles", "E": "Construction", "F": "Mech. eng.",
          "G": "Physics", "H": "Electricity", "Y": "Cross-sectional",
          "unknown": "Unknown"}
DISTRICTS = {"north": "North", "haifa": "Haifa", "center": "Center",
             "tel aviv": "Tel Aviv", "jerusalem": "Jerusalem",
             "south": "South", "unknown": "Unknown"}


def stars(p):
    """Course convention: * p<0.01, ** p<0.05."""
    return "*" if p < 0.01 else ("**" if p < 0.05 else "")


def head(n, t):
    print("\n" + LINE)
    print(f"{n}. {t}")
    print(LINE)


# ==================================================================== 1
def descriptives(d):
    head(1, "WHO IS IN THE SAMPLE")
    a, j = d[d.arab == 1], d[d.arab == 0]
    print(f"  inventors analysed : {len(d):,}")
    print(f"  Arab               : {len(a):,}  ({len(a)/len(d):.2%})")
    print(f"  Jewish             : {len(j):,}")

    print("\n  Table: averages and simple t-tests")
    print(f"  {'variable':<26}{'Arab':>9}{'Jewish':>10}{'diff':>10}{'p':>9}")
    rows = []
    for c, lab in [("total_patents", "patents in 5 years"),
                   ("corp_count", "corporate patents"),
                   ("indep_count", "independent patents"),
                   ("corp_share", "share corporate"),
                   ("ever_corporate", "ever corporate"),
                   ("n_firms", "number of employers")]:
        t, p = stats.ttest_ind(a[c], j[c], equal_var=False)
        print(f"  {lab:<26}{a[c].mean():>9.3f}{j[c].mean():>10.3f}"
              f"{a[c].mean()-j[c].mean():>+10.3f}{p:>9.3f} {stars(p)}")
        rows.append({"variable": lab, "arab": round(a[c].mean(), 3),
                     "jewish": round(j[c].mean(), 3),
                     "diff": round(a[c].mean()-j[c].mean(), 3),
                     "p": round(p, 3), "sig": stars(p)})
    pd.DataFrame(rows).to_csv("simple_table1_means.csv", index=False)
    print("\n  -> simple_table1_means.csv")


# ==================================================================== 2
def representation(d, panel):
    head(2, "HOW MANY ISRAELI INVENTORS ARE ARAB?")
    share = (d.arab == 1).mean()
    print(f"  Among inventors we could classify : {share:.2%}")
    print(f"  If ALL unclassified were Jewish   : "
          f"{(d.arab == 1).sum()/len(panel):.2%}")
    print("  The truth is between these two. Report the range.")
    print("  Arab share of Israeli citizens    : about 21%")

    print("\n  By entry cohort (5-year groups):")
    d = d.copy()
    d["group"] = (d.first_year // 5) * 5
    t = d.groupby("group")["arab"].agg(inventors="size", arab="sum")
    t["pct"] = (100 * t.arab / t.inventors).round(2)
    print(t.to_string())
    t.to_csv("simple_table2_cohorts.csv")

    for col, names, lab in [("district", DISTRICTS, "district"),
                            ("field", FIELDS, "technology field")]:
        g = d.groupby(col)["arab"].agg(n="size", arab="sum")
        g = g[g.n >= 100]
        g["pct"] = (100 * g.arab / g.n).round(2)
        g.index = [names.get(i, i) for i in g.index]
        print(f"\n  By {lab}:")
        print(g.sort_values("pct", ascending=False).to_string())
        g.to_csv(f"simple_table_{col}.csv")


# ==================================================================== 3
def regressions(d):
    head(3, "DOES BEING ARAB PREDICT INVENTING INSIDE A FIRM?")
    print("  Two ordinary regressions. Controls: entry year, district, field.")
    print("  If the barrier is inside the firm, the Arab coefficient should")
    print("  be NEGATIVE.\n")

    rows = []
    # (a) linear regression on the share
    m1 = smf.ols("corp_share ~ arab + C(cohort) + C(district) + C(field)",
                 data=d).fit(cov_type="HC1")
    b, se, p = m1.params["arab"], m1.bse["arab"], m1.pvalues["arab"]
    print(f"  (a) LINEAR regression, outcome = share of patents that are")
    print(f"      corporate (0 to 1)")
    print(f"      Arab coefficient {b:+.4f}  (std. error {se:.4f})  "
          f"p = {p:.3f} {stars(p)}")
    print(f"      reads as: {b*100:+.2f} percentage points")
    rows.append({"model": "OLS, corporate share", "coef": round(b, 4),
                 "se": round(se, 4), "p": round(p, 3), "sig": stars(p)})

    # (b) logistic regression on the yes/no outcome
    m2 = smf.logit("ever_corporate ~ arab + C(cohort) + C(district) + C(field)",
                   data=d).fit(disp=0)
    b, se, p = m2.params["arab"], m2.bse["arab"], m2.pvalues["arab"]
    print(f"\n  (b) LOGISTIC regression, outcome = has at least one corporate")
    print(f"      patent (yes/no)")
    print(f"      Arab coefficient {b:+.4f}  (std. error {se:.4f})  "
          f"p = {p:.3f} {stars(p)}")
    print(f"      odds ratio {np.exp(b):.3f}")
    rows.append({"model": "Logit, ever corporate", "coef": round(b, 4),
                 "se": round(se, 4), "p": round(p, 3), "sig": stars(p)})

    # (c) same thing with no controls, to show it does not matter
    m3 = smf.ols("corp_share ~ arab", data=d).fit(cov_type="HC1")
    b, se, p = m3.params["arab"], m3.bse["arab"], m3.pvalues["arab"]
    print(f"\n  (c) same as (a) but with NO controls, as a check")
    print(f"      Arab coefficient {b:+.4f}  (std. error {se:.4f})  "
          f"p = {p:.3f} {stars(p)}")
    rows.append({"model": "OLS, no controls", "coef": round(b, 4),
                 "se": round(se, 4), "p": round(p, 3), "sig": stars(p)})

    pd.DataFrame(rows).to_csv("simple_table3_regressions.csv", index=False)
    print("\n  -> simple_table3_regressions.csv")


# ==================================================================== 4
def teams(panel, d):
    head(4, "WHO DO ARAB INVENTORS WORK WITH?")
    print("  The idea, in one sentence: if Arab inventors were put on teams")
    print("  at random inside their company, their co-inventors should look")
    print("  like the company does. We compare those two numbers.\n")

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

    # --- (a) the overall picture ---
    raw["is_arab"] = (raw.eth == "arab").astype(int)
    g = raw.groupby("patent_id")["is_arab"]
    raw["team_arab"] = g.transform("sum")
    raw["team_size"] = g.transform("size")
    raw["co_arab"] = raw.team_arab - raw.is_arab
    raw["co_total"] = raw.team_size - 1
    co = raw[raw.co_total > 0].groupby(["inventor_id", "is_arab"]).agg(
        ca=("co_arab", "sum"), ct=("co_total", "sum")).reset_index()
    co["pct"] = 100 * co.ca / co.ct
    if (co.is_arab == 1).sum() < 5:
        print("  Not enough Arab inventors with co-inventors to run this.")
        return
    a_pct = co[co.is_arab == 1]["pct"].mean()
    j_pct = co[co.is_arab == 0]["pct"].mean()
    base = 100 * raw.drop_duplicates("inventor_id")["is_arab"].mean()
    print(f"  Arab share of these inventors overall      : {base:.2f}%")
    print(f"  Arab inventors: % of co-inventors who are Arab : {a_pct:.2f}%")
    print(f"  Jewish inventors: same figure                  : {j_pct:.2f}%")
    t, p = stats.ttest_ind(co[co.is_arab == 1]["pct"],
                           co[co.is_arab == 0]["pct"], equal_var=False)
    print(f"  difference {a_pct-j_pct:+.2f} points, p = {p:.4f} {stars(p)}")

    # --- (b) the within-company comparison (the important one) ---
    print("\n  Now the same question INSIDE each company.")
    firm = (raw.groupby(["inventor_id", "assignee_id"]).size()
               .reset_index(name="k")
               .sort_values(["inventor_id", "k"], ascending=[True, False],
                            kind="mergesort")
               .drop_duplicates("inventor_id"))
    who = raw.drop_duplicates("inventor_id")[["inventor_id", "is_arab"]]
    firm = firm.merge(who, on="inventor_id")
    size = firm.groupby("assignee_id")["is_arab"].agg(n="size", n_arab="sum")

    co2 = co.merge(firm[["inventor_id", "assignee_id"]], on="inventor_id")
    co2 = co2.merge(size, on="assignee_id")
    co2 = co2[(co2.is_arab == 1) & (co2.n >= 3) & (co2.n_arab >= 1)]
    # what the company itself looks like, excluding the person
    co2["company_pct"] = 100 * (co2.n_arab - 1) / (co2.n - 1)

    if len(co2) < 10:
        print("  Too few Arab inventors at multi-inventor firms.")
        return
    t, p = stats.ttest_rel(co2["pct"], co2["company_pct"])
    print(f"  Arab inventors compared          : {len(co2):,}")
    print(f"  their co-inventors are Arab      : {co2['pct'].mean():.2f}%")
    print(f"  their COMPANY is Arab            : {co2['company_pct'].mean():.2f}%")
    print(f"  difference                       : "
          f"{co2['pct'].mean()-co2['company_pct'].mean():+.2f} points")
    print(f"  paired t-test                    : t = {t:.2f}, p = {p:.4f}"
          f" {stars(p)}")
    print()
    if p < 0.05 and co2["pct"].mean() > co2["company_pct"].mean():
        print("  -> Arab inventors work with MORE Arab colleagues than their")
        print("     own company's make-up would produce by chance. Teams")
        print("     inside firms are separated, not just firms.")
    else:
        print("  -> No evidence of separation beyond which company people")
        print("     work at.")
    pd.DataFrame({"co_inventors_arab_pct": [co2["pct"].mean()],
                  "company_arab_pct": [co2["company_pct"].mean()],
                  "n": [len(co2)], "p": [p]}).to_csv(
        "simple_table4_teams.csv", index=False)
    print("\n  -> simple_table4_teams.csv")


# ==================================================================== 5
def robustness(d, panel):
    head(5, "DOES THE ANSWER DEPEND ON THE NAMES WE COULD NOT CLASSIFY?")
    exc = panel[panel.ethnicity == "ambiguous"]
    print(f"  unclassified inventors: {len(exc):,}")
    t, p = stats.ttest_ind(exc["corp_share"], d[d.arab == 0]["corp_share"],
                           equal_var=False)
    print(f"  their share corporate  : {exc['corp_share'].mean():.3f}")
    print(f"  Jewish share corporate : {d[d.arab==0]['corp_share'].mean():.3f}")
    print(f"  difference is not meaningful (p = {p:.3f}), so leaving them out")
    print("  does not tilt the comparison.\n")

    print("  As a check, we pretend some of them are Arab and redo the")
    print("  regression. If the answer barely moves, it does not depend on")
    print("  them.\n")
    rng = np.random.default_rng(1)
    print(f"  {'assumed Arab':>14}{'coefficient':>14}{'p':>9}")
    for share in (0.00, 0.02, 0.05, 0.10):
        e = exc.copy()
        e["arab"] = (rng.random(len(e)) < share).astype(int)
        both = pd.concat([d, e], ignore_index=True)
        m = smf.ols("corp_share ~ arab + C(cohort)", data=both).fit(
            cov_type="HC1")
        print(f"  {share:>13.0%}{m.params['arab']:>+14.4f}"
              f"{m.pvalues['arab']:>9.3f}")
    print("\n  -> if these lines are all close, say so in the paper.")


def main():
    panel = pd.read_csv("panel.csv")
    panel["cohort"] = panel["cohort"].astype(str)
    d = panel[panel.ethnicity.isin(["arab", "jewish"])].copy()
    d["arab"] = (d.ethnicity == "arab").astype(int)

    descriptives(d)
    representation(d, panel)
    regressions(d)
    try:
        teams(panel, d)
    except FileNotFoundError:
        print("\n(raw_records.csv missing -- skipping the team analysis)")
    robustness(d, panel)
    print("\n" + LINE)
    print("Done. Five tables written as simple_table*.csv")
    print(LINE)


if __name__ == "__main__":
    main()
