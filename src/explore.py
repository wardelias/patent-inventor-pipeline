"""
Data exploration -- tables and figures for Section V.

    python explore.py

Produces tables in the format the course guidelines specify (summary
statistics; mean comparison with stars; * p<0.01, ** p<0.05) and eight figures.

Axis labels are in English. Hebrew text in matplotlib needs bidi handling and
a Hebrew-capable font, which is fragile across machines; the standard practice
in Israeli theses is English figure labels with Hebrew captions in the text.

NEW HERE: co-inventor team analysis (Section 5). Corporate share asks WHERE a
person invents. Team composition asks WHO THEY INVENT WITH -- which is closer
to what H1 and H3 are actually about, since sponsorship and legitimacy operate
through colleagues.
"""
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "figure.autolayout": True,
})

ARAB_C, JEW_C = "#c0392b", "#2c3e50"
FIELDS = {"A": "Human necessities", "B": "Operations & transport",
          "C": "Chemistry & metallurgy", "D": "Textiles", "E": "Construction",
          "F": "Mechanical engineering", "G": "Physics", "H": "Electricity",
          "Y": "Cross-sectional tech", "unknown": "Unknown"}
DISTRICTS = {"north": "North", "haifa": "Haifa", "center": "Center",
             "tel aviv": "Tel Aviv", "jerusalem": "Jerusalem",
             "south": "South", "unknown": "Unknown"}
LINE = "=" * 70


def sig(p):
    """Course guideline convention: * p<0.01, ** p<0.05."""
    if p < 0.01:
        return "*"
    if p < 0.05:
        return "**"
    return ""


def load():
    p = pd.read_csv("panel.csv")
    p["surname"] = p["surname"].astype(str)
    p["city"] = p["city"].astype(str)
    a = p[p.ethnicity.isin(["arab", "jewish"])].copy()
    a["arab"] = (a.ethnicity == "arab").astype(int)
    return p, a


# =====================================================================  1
def table_summary(d):
    print(LINE)
    print("TABLE A. Summary statistics for main variables")
    print(LINE)
    cols = ["total_patents", "corp_count", "indep_count", "corp_share",
            "ever_corporate", "n_firms", "first_year"]
    rows = []
    for c in cols:
        if c not in d:
            continue
        rows.append({"Variable": c, "Obs.": int(d[c].notna().sum()),
                     "Mean": round(d[c].mean(), 3),
                     "Std. Dev.": round(d[c].std(), 3),
                     "Min": round(d[c].min(), 2),
                     "Max": round(d[c].max(), 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv("table_A_summary.csv", index=False)
    return t


# =====================================================================  2
def table_means(d):
    print("\n" + LINE)
    print("TABLE B. Mean comparison: Arab vs Jewish inventors")
    print(LINE)
    a, j = d[d.arab == 1], d[d.arab == 0]
    rows = []
    for c in ["total_patents", "corp_count", "indep_count", "corp_share",
              "ever_corporate", "n_firms"]:
        t, p = stats.ttest_ind(a[c], j[c], equal_var=False)
        rows.append({
            "Variable": c,
            "(1) Diff in means": f"{a[c].mean()-j[c].mean():+.3f}{sig(p)}",
            "(2) Arab Obs.": len(a), "(3) Arab Mean": round(a[c].mean(), 3),
            "(4) Jewish Obs.": len(j), "(5) Jewish Mean": round(j[c].mean(), 3),
            "t": round(t, 2), "p": round(p, 3)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print("\nNotes: Welch t-tests. * p<0.01, ** p<0.05 (course convention).")
    t.to_csv("table_B_means.csv", index=False)
    return t


# =====================================================================  3
def table_by_group(d, col, label, namemap, minn=30):
    g = d.groupby(col).agg(n=("arab", "size"), arab=("arab", "sum"),
                           corp_share=("corp_share", "mean")).reset_index()
    g = g[g.n >= minn]
    g["arab_pct"] = (100 * g.arab / g.n).round(2)
    g[col] = g[col].map(lambda x: namemap.get(x, x))
    g = g.sort_values("arab_pct", ascending=False)
    g["corp_share"] = g["corp_share"].round(3)
    print("\n" + LINE)
    print(f"TABLE. Arab representation by {label}")
    print(LINE)
    print(g[[col, "n", "arab", "arab_pct", "corp_share"]].to_string(index=False))
    return g


# =====================================================================  4
def teams(panel):
    """Co-inventor composition -- who Arab inventors invent alongside."""
    print("\n" + LINE)
    print("TABLE E. Co-inventor team composition")
    print(LINE)
    try:
        raw = pd.read_csv("raw_records.csv", dtype=str)
    except FileNotFoundError:
        print("  raw_records.csv not found -- skipping")
        return None, None

    raw["filing_year"] = pd.to_numeric(raw["filing_year"], errors="coerce")
    eth = panel.set_index("inventor_id")["ethnicity"].to_dict()
    raw["eth"] = raw["inventor_id"].map(eth)
    raw = raw[raw["eth"].notna()]

    # Restrict to the SAME five-year window the panel uses. Without this the
    # team measures were computed over each inventor's entire history while
    # every other variable used the window -- an inconsistency that would
    # have made team size correlate mechanically with career length.
    fy = panel.set_index("inventor_id")["first_year"].to_dict()
    raw["first_year"] = raw["inventor_id"].map(fy)
    raw = raw[(raw.filing_year >= raw.first_year)
              & (raw.filing_year < raw.first_year + 5)]

    # team size and composition per patent (Israeli inventors only)
    pat = raw.groupby("patent_id")["eth"].agg(
        team_size="size",
        n_arab=lambda x: (x == "arab").sum(),
        n_jew=lambda x: (x == "jewish").sum()).reset_index()
    raw = raw.merge(pat, on="patent_id")

    # per inventor: mean team size, and share of identified co-inventors
    # who are Arab
    raw["co_arab"] = raw["n_arab"] - (raw["eth"] == "arab").astype(int)
    raw["co_jew"] = raw["n_jew"] - (raw["eth"] == "jewish").astype(int)
    inv = raw.groupby("inventor_id").agg(
        mean_team=("team_size", "mean"),
        solo_share=("team_size", lambda x: (x == 1).mean()),
        co_arab=("co_arab", "sum"), co_jew=("co_jew", "sum")).reset_index()
    inv["co_identified"] = inv.co_arab + inv.co_jew
    inv["pct_co_arab"] = np.where(inv.co_identified > 0,
                                  100 * inv.co_arab / inv.co_identified, np.nan)

    d = panel.merge(inv, on="inventor_id")
    d = d[d.ethnicity.isin(["arab", "jewish"])]
    a, j = d[d.ethnicity == "arab"], d[d.ethnicity == "jewish"]

    rows = []
    for c, lab in [("mean_team", "Mean Israeli co-inventors per patent"),
                   ("solo_share", "Share of patents invented solo"),
                   ("pct_co_arab", "% of identified co-inventors who are Arab")]:
        aa, jj = a[c].dropna(), j[c].dropna()
        if len(aa) < 5 or len(jj) < 5:
            continue
        if aa.std() == 0 and jj.std() == 0:
            continue
        t, p = stats.ttest_ind(aa, jj, equal_var=False)
        if np.isnan(p):
            continue
        rows.append({"Variable": lab,
                     "(1) Diff": f"{aa.mean()-jj.mean():+.3f}{sig(p)}",
                     "(2) Arab Obs.": len(aa), "(3) Arab Mean": round(aa.mean(), 3),
                     "(4) Jewish Obs.": len(jj), "(5) Jewish Mean": round(jj.mean(), 3),
                     "p": round(p, 4)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv("table_E_teams.csv", index=False)

    print("\n  SCOPE: team size counts only Israeli co-inventors who are")
    print("  themselves in the panel. Foreign co-inventors and inventors")
    print("  outside the cohort window are not counted, so these are lower")
    print("  bounds on true team size. The COMPOSITION share is unaffected.")
    print("\n  WHY THIS MATTERS")
    print("  Corporate share asks WHERE someone invents. This asks WHO WITH.")
    print("  H1 (sponsorship) and H3 (legitimacy) operate through colleagues,")
    print("  so segregation in co-inventor teams is a more direct test than")
    print("  the corporate/independent split. If Arab inventors work mostly")
    print("  alongside other Arab inventors, that is evidence of the kind of")
    print("  organisational separation the three-gate model describes.")
    if len(t):
        r = t[t.Variable.str.contains("co-inventors who are Arab")]
        if len(r):
            print(f"\n  Arab inventors' co-inventors are "
                  f"{r.iloc[0]['(3) Arab Mean']}% Arab;")
            print(f"  Jewish inventors' co-inventors are "
                  f"{r.iloc[0]['(5) Jewish Mean']}% Arab.")
            print("  Compare both against the Arab share of all inventors "
                  "(~2%).")
    return d, t


# =====================================================================  5
def figures(d, panel, team_df):
    a, j = d[d.arab == 1], d[d.arab == 0]

    # F1 -- Arab share by cohort, with 95% CI
    g = d.groupby("first_year")["arab"].agg(["mean", "count"])
    g = g[g["count"] >= 30]
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(g.index, 100 * g["mean"], marker="o", ms=4, color=ARAB_C, lw=1.8)
    ax.fill_between(g.index, 100 * (g["mean"] - 1.96 * se),
                    100 * (g["mean"] + 1.96 * se), color=ARAB_C, alpha=.15)
    ax.set_xlabel("Year of first patent (entry cohort)")
    ax.set_ylabel("Arab share of new inventors (%)")
    ax.set_title("Figure 1. Arab share of Israeli inventors, by entry cohort")
    ax.axhline(20, ls="--", lw=1, color="grey")
    ax.text(g.index.min(), 20.4, "Arab share of population (~20%)",
            fontsize=8, color="grey")
    ax.set_ylim(0, 22)
    fig.savefig("fig1_arab_share_cohort.png"); plt.close(fig)

    # F2 -- Arab share by field
    g2 = d.groupby("field")["arab"].agg(["mean", "count"])
    g2 = g2[g2["count"] >= 100].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([FIELDS.get(i, i) for i in g2.index], 100 * g2["mean"],
            color=JEW_C, height=.65)
    for y, (v, n) in enumerate(zip(g2["mean"], g2["count"])):
        ax.text(100 * v + .04, y, f"n={n:,}", va="center", fontsize=7.5,
                color="grey")
    ax.set_xlabel("Arab share of inventors (%)")
    ax.set_title("Figure 2. Arab representation by technology field")
    fig.savefig("fig2_arab_share_field.png"); plt.close(fig)

    # F3 -- Arab share by district
    g3 = d.groupby("district")["arab"].agg(["mean", "count"])
    g3 = g3[g3["count"] >= 100].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([DISTRICTS.get(i, i) for i in g3.index], 100 * g3["mean"],
            color="#16a085", height=.6)
    for y, (v, n) in enumerate(zip(g3["mean"], g3["count"])):
        ax.text(100 * v + .06, y, f"n={n:,}", va="center", fontsize=7.5,
                color="grey")
    ax.set_xlabel("Arab share of inventors (%)")
    ax.set_title("Figure 3. Arab representation by district of residence")
    fig.savefig("fig3_arab_share_district.png"); plt.close(fig)

    # F4 -- distribution of corporate share (THE key picture)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, 1, 21)
    # Percent of inventors, not density: a density axis reading "17.5" is
    # meaningless to a reader, and the point of this figure is how much mass
    # sits at 1.0.
    wj = np.ones(len(j)) * 100 / len(j)
    wa = np.ones(len(a)) * 100 / len(a)
    ax.hist([j["corp_share"], a["corp_share"]], bins=bins,
            weights=[wj, wa], color=[JEW_C, ARAB_C],
            label=[f"Jewish (n={len(j):,})", f"Arab (n={len(a):,})"])
    ax.set_xlabel("Share of an inventor's patents that are corporate")
    ax.set_ylabel("% of inventors in group")
    ax.set_title("Figure 4. Distribution of corporate share, by group")
    ax.legend(frameon=False)
    ax.text(0.02, 0.86,
            "Both groups pile up at 1.0 — almost all Israeli\n"
            "invention happens inside a firm. There is little\n"
            "variation left for ethnicity to explain.",
            transform=ax.transAxes, fontsize=8, color="#555", va="top")
    fig.savefig("fig4_corp_share_dist.png"); plt.close(fig)

    # F5 -- corporate vs independent means
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2); w = .35
    ax.bar(x - w/2, [a.corp_count.mean(), a.indep_count.mean()], w,
           color=ARAB_C, label="Arab")
    ax.bar(x + w/2, [j.corp_count.mean(), j.indep_count.mean()], w,
           color=JEW_C, label="Jewish")
    ax.set_xticks(x); ax.set_xticklabels(["Corporate patents",
                                          "Independent patents"])
    ax.set_ylabel("Mean patents in 5-year window")
    ax.set_title("Figure 5. Corporate vs independent output, by group")
    ax.legend(frameon=False)
    fig.savefig("fig5_corp_vs_indep.png"); plt.close(fig)

    # F6 -- absolute counts over time (the entry gap, in levels)
    g6 = d.groupby("first_year")["arab"].agg(total="size", arab="sum")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(g6.index, g6["total"], color="#bdc3c7", label="All inventors")
    ax.bar(g6.index, g6["arab"], color=ARAB_C, label="Arab inventors")
    ax.set_xlabel("Year of first patent")
    ax.set_ylabel("Number of new inventors")
    ax.set_title("Figure 6. New Israeli inventors per year, Arab share in red")
    ax.legend(frameon=False)
    fig.savefig("fig6_counts_over_time.png"); plt.close(fig)

    # F7 -- patents per inventor
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cap = 10
    for grp, c, lab in [(j, JEW_C, "Jewish"), (a, ARAB_C, "Arab")]:
        v = grp["total_patents"].clip(upper=cap)
        h = np.bincount(v.astype(int), minlength=cap+1)[1:] / len(v)
        ax.plot(range(1, cap+1), 100*h, marker="o", ms=4, color=c, label=lab)
    ax.set_xlabel(f"Patents in 5-year window (capped at {cap})")
    ax.set_ylabel("% of inventors")
    ax.set_title("Figure 7. Productivity distribution, by group")
    ax.legend(frameon=False)
    fig.savefig("fig7_productivity.png"); plt.close(fig)

    # F8 -- sample construction funnel
    try:
        fl = pd.read_csv("flow.csv")
        col = "inventors" if "inventors" in fl else fl.columns[1]
        f = fl[fl[col].notna()]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.barh(range(len(f)), f[col], color="#34495e", height=.6)
        ax.set_yticks(range(len(f)))
        ax.set_yticklabels([s[:40] for s in f["stage"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Inventors remaining")
        ax.set_title("Figure 8. Sample construction")
        for i, v in enumerate(f[col]):
            ax.text(v, i, f" {int(v):,}", va="center", fontsize=7.5)
        fig.savefig("fig8_sample_funnel.png"); plt.close(fig)
    except Exception as e:
        print(f"  (figure 8 skipped: {e})")

    # F9 is produced by homophily.py, which has the permutation nulls.
    if False and team_df is not None and len(team_df):
        td = team_df
        fig, ax = plt.subplots(figsize=(6.5, 4))
        aa = td[td.ethnicity == "arab"]["pct_co_arab"].dropna()
        jj = td[td.ethnicity == "jewish"]["pct_co_arab"].dropna()
        ax.bar([0, 1], [aa.mean(), jj.mean()], .5, color=[ARAB_C, JEW_C])
        ax.axhline(100 * (td.ethnicity == "arab").mean(), ls="--", lw=1,
                   color="grey")
        ax.text(1.28, 100 * (td.ethnicity == "arab").mean(),
                "Arab share of\nall inventors", fontsize=7.5, color="grey",
                va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Arab inventors", "Jewish inventors"])
        ax.set_ylabel("% of co-inventors who are Arab")
        ax.set_title("Figure 9. Co-inventor composition, by group")
        fig.savefig("fig9_team_composition.png"); plt.close(fig)

    print("\n" + LINE)
    print("FIGURES WRITTEN")
    print(LINE)
    for f in ["fig1_arab_share_cohort", "fig2_arab_share_field",
              "fig3_arab_share_district", "fig4_corp_share_dist",
              "fig5_corp_vs_indep", "fig6_counts_over_time",
              "fig7_productivity", "fig8_sample_funnel",
              "fig9_team_composition"]:
        print(f"  {f}.png")


def main():
    panel, d = load()
    table_summary(d)
    table_means(d)
    fg = table_by_group(d, "field", "technology field", FIELDS, minn=100)
    fg.to_csv("table_C_by_field.csv", index=False)
    dg = table_by_group(d, "district", "district of residence", DISTRICTS,
                        minn=100)
    dg.to_csv("table_D_by_district.csv", index=False)
    team_d, team_t = teams(panel)
    figures(d, panel, team_d)
    print("\nAll tables written as table_*.csv")


if __name__ == "__main__":
    main()
