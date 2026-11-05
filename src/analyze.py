"""
Step 3 -- descriptives, then the model ladder.

Order matters and follows Lia's instruction (30/6): non-parametric first
(graphs, mean comparisons, descriptives), parametric after.

Stage A is a GO / NO-GO gate. If almost nobody has independent patents, the
corporate-share outcome has no variance and the main model is dead. Check this
before writing a single line of the paper.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


FIELD_NAMES = {
    "A": "Human necessities", "B": "Operations/transport", "C": "Chemistry",
    "D": "Textiles", "E": "Construction", "F": "Mech. eng.",
    "G": "Physics", "H": "Electricity", "Y": "Cross-sectional",
    "unknown": "Unknown",
}


def load(path="panel.csv"):
    df = pd.read_csv(path)
    df = df[df["ethnicity"].isin(["arab", "jewish"])].copy()
    df["arab"] = (df["ethnicity"] == "arab").astype(int)
    df["cohort"] = df["cohort"].astype(str)
    return df


# ---------------------------------------------------------------- Stage A
def variance_check(df):
    print("\n" + "=" * 68)
    print("STAGE A -- GO/NO-GO: does the corporate-share outcome have variance?")
    print("=" * 68)
    out = []
    for eth, g in df.groupby("ethnicity"):
        any_indep = (g["indep_count"] > 0).mean()
        out.append({
            "group": eth, "n": len(g),
            "pct_with_any_independent": round(100 * any_indep, 2),
            "mean_corp_share": round(g["corp_share"].mean(), 4),
            "pct_corp_share_eq_1": round(100 * (g["corp_share"] == 1).mean(), 2),
            "mean_total_patents": round(g["total_patents"].mean(), 2),
        })
    t = pd.DataFrame(out)
    print(t.to_string(index=False))

    pooled = (df["indep_count"] > 0).mean()
    print(f"\n  Pooled: {pooled:.1%} of inventors have >=1 independent patent.")
    if pooled < 0.05:
        print("  >> NO-GO. Under 5% -- corp_share is ~1 for everyone and beta_1")
        print("     has almost nothing to explain. Fall back to the binary and")
        print("     count outcomes, and tell Lia the share model is not viable.")
    elif pooled < 0.15:
        print("  >> MARGINAL. Report the share model but lead with binary/count.")
    else:
        print("  >> GO. The share model is the headline specification.")
    return t


# ---------------------------------------------------------------- Stage B
def descriptives(df, prefix="fig"):
    print("\n" + "=" * 68)
    print("STAGE B -- descriptive statistics")
    print("=" * 68)

    t1 = df.groupby("ethnicity").agg(
        n=("inventor_id", "count"),
        total_patents=("total_patents", "mean"),
        corp_count=("corp_count", "mean"),
        indep_count=("indep_count", "mean"),
        corp_share=("corp_share", "mean"),
        ever_corporate=("ever_corporate", "mean"),
        n_firms=("n_firms", "mean"),
    ).round(3)
    print("\nTable 1 -- means by group")
    print(t1.to_string())

    a = df[df.arab == 1]["corp_share"]
    j = df[df.arab == 0]["corp_share"]
    if len(a) > 1 and len(j) > 1:
        from scipy import stats
        tt = stats.ttest_ind(a, j, equal_var=False)
        print(f"\nDifference in mean corporate share: "
              f"{a.mean() - j.mean():+.4f}  (Welch t={tt.statistic:.2f}, "
              f"p={tt.pvalue:.4f})")

    # Graph 1 -- Arab share of inventors by cohort year
    s = df.groupby("cohort")["arab"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s.index, s.values, marker="o", color="#c0392b")
    ax.set_xlabel("Cohort (year of first patent)")
    ax.set_ylabel("Arab share of inventors (%)")
    ax.set_title("Arab share of Israeli inventors, by entry cohort")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_arab_share_by_cohort.png", dpi=150)
    plt.close(fig)

    # Graph 2 -- Arab share by field
    s2 = (df.groupby("field")["arab"].agg(["mean", "count"])
            .query("count >= 20").sort_values("mean"))
    if len(s2):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([FIELD_NAMES.get(i, i) for i in s2.index],
                s2["mean"] * 100, color="#2c3e50")
        ax.set_xlabel("Arab share of inventors (%)")
        ax.set_title("Arab share by technology field (CPC section)")
        fig.tight_layout(); fig.savefig(f"{prefix}_arab_share_by_field.png", dpi=150)
        plt.close(fig)

    # Graph 3 -- the contrast test, visually
    comp = df.groupby("ethnicity")[["corp_count", "indep_count"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    comp.plot(kind="bar", ax=ax, color=["#2980b9", "#e67e22"], rot=0)
    ax.set_ylabel(f"Mean patents in 5-year window")
    ax.set_title("Corporate vs independent output, by group")
    ax.legend(["Corporate", "Independent"])
    fig.tight_layout(); fig.savefig(f"{prefix}_corp_vs_indep.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 3 figures as {prefix}_*.png")
    return t1


# ---------------------------------------------------------------- Stage C
LADDER = [
    ("M0  cohort only",              "arab + C(cohort)"),
    ("M1  + region",                 "arab + C(cohort) + C(district)"),
    ("M2  + field",                  "arab + C(cohort) + C(field)"),
    ("M3  + region + field (full)",  "arab + C(cohort) + C(district) + C(field)"),
]


def _fit_binomial(df, rhs):
    X = patsy.dmatrix(rhs, df, return_type="dataframe")
    y = df[["corp_count", "indep_count"]].values
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")


def _fit_logit(df, rhs):
    X = patsy.dmatrix(rhs, df, return_type="dataframe")
    return sm.GLM(df["ever_corporate"], X,
                  family=sm.families.Binomial()).fit(cov_type="HC1")


def _fit_ppml(df, rhs):
    """Poisson pseudo-ML with robust SEs. Preferred over negative binomial:
    consistent even under misspecified variance, and stable with fixed effects."""
    X = patsy.dmatrix(rhs, df, return_type="dataframe")
    return sm.GLM(df["corp_count"], X,
                  family=sm.families.Poisson()).fit(cov_type="HC1")


def ladder(df):
    print("\n" + "=" * 68)
    print("STAGE C -- model ladder")
    print("The finding is how beta_1 MOVES across columns, not any one column.")
    print("Region and field are partly downstream of the discrimination, so")
    print("controlling for them absorbs part of the gap we want to measure.")
    print("=" * 68)

    specs = [
        ("BINOMIAL corp share (main)", _fit_binomial),
        ("LOGIT ever-corporate", _fit_logit),
        ("PPML corporate count", _fit_ppml),
    ]
    rows = []
    for label, fitter in specs:
        for name, rhs in LADDER:
            try:
                r = fitter(df, rhs)
                b = r.params["arab"]
                se = r.bse["arab"]
                p = r.pvalues["arab"]
                rows.append({
                    "model": label, "spec": name,
                    "beta_arab": round(b, 4), "se": round(se, 4),
                    "exp(beta)": round(np.exp(b), 3),
                    "p": round(p, 4),
                    "sig": sig(p),
                    "n": int(r.nobs),
                })
            except Exception as e:
                rows.append({"model": label, "spec": name,
                             "beta_arab": np.nan, "se": np.nan,
                             "exp(beta)": np.nan, "p": np.nan,
                             "sig": f"FAILED: {type(e).__name__}", "n": 0})
    t = pd.DataFrame(rows)
    for label in t["model"].unique():
        print(f"\n--- {label} ---")
        print(t[t.model == label].drop(columns="model").to_string(index=False))
    t.to_csv("results_ladder.csv", index=False)
    return t


# ---------------------------------------------------------------- Stage D
def firm_fixed_effects(raw_path="raw_records.csv", panel_path="panel.csv"):
    """
    IMPORTANT -- this answers a DIFFERENT question from the main model.

    Inside a firm, every patent is corporate by construction, so corporate
    SHARE cannot vary within firm. Firm FE therefore tests: among people who
    patent at the same firm, do Arab employees produce fewer patents? That is
    output/credit (H1, H3), not access. Present it as a distinct result, not
    as a robustness check on the share model.

    It is also conditioned on having patented at that firm at all, so it cannot
    see Arab employees who were blocked before any patent.
    """
    print("\n" + "=" * 68)
    print("STAGE D -- firm fixed effects (different question, see docstring)")
    print("=" * 68)
    raw = pd.read_csv(raw_path, dtype=str).fillna("")
    panel = pd.read_csv(panel_path)
    panel = panel[panel.ethnicity.isin(["arab", "jewish"])]
    panel["arab"] = (panel.ethnicity == "arab").astype(int)

    raw["filing_year"] = pd.to_numeric(raw["filing_year"], errors="coerce")
    raw = raw.merge(panel[["inventor_id", "arab", "first_year"]], on="inventor_id")
    raw = raw[(raw.filing_year >= raw.first_year) &
              (raw.filing_year < raw.first_year + 5) &
              (raw.assignee_id != "")]

    spells = (raw.groupby(["inventor_id", "assignee_id", "arab"])["patent_id"]
                 .nunique().rename("n_patents").reset_index())

    mix = spells.groupby("assignee_id")["arab"].agg(["nunique", "count"])
    keep = mix[mix["nunique"] == 2].index
    kept = spells[spells.assignee_id.isin(keep)]

    print(f"  inventor-firm spells total          : {len(spells):,}")
    print(f"  firms total                         : {spells.assignee_id.nunique():,}")
    print(f"  firms with BOTH groups              : {len(keep):,}")
    print(f"  spells surviving firm FE            : {len(kept):,} "
          f"({len(kept)/max(len(spells),1):.1%})")
    if len(kept) < 100 or kept["arab"].sum() < 20:
        print("  >> Sample too thin. Report as a footnote, not a headline.")
        return None

    X = patsy.dmatrix("arab + C(assignee_id)", kept, return_type="dataframe")
    r = sm.GLM(kept["n_patents"], X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": kept["assignee_id"]})
    print(f"\n  beta_arab = {r.params['arab']:+.4f}  "
          f"(se {r.bse['arab']:.4f}, p {r.pvalues['arab']:.4f})  "
          f"exp = {np.exp(r.params['arab']):.3f}")
    return r


if __name__ == "__main__":
    d = load()
    variance_check(d)
    descriptives(d)
    ladder(d)
    try:
        firm_fixed_effects()
    except FileNotFoundError:
        print("\n(skipping firm FE -- raw_records.csv not present)")
