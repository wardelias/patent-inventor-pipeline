"""
Step 2 -- raw_records.csv -> panel.csv (one row per inventor).

Encodes every design decision from the thread, with the contradictions resolved.
See SPEC.md. Each numbered stage below writes a line into flow.csv, which is the
sample-construction flow chart promised to Lia (28/6, point 7).
"""
import re
import sys

import numpy as np
import pandas as pd

import names as nm
import regions as rg

# ---- parameters (all in one place, all defensible in a footnote) ----------
COHORT_MIN, COHORT_MAX = 1990, 2019   # first-patent year.
# Widened from 2010 on purpose. Lia asked for a UNIFORM EXPOSURE WINDOW so that
# patent counts are not driven by how long someone worked -- the 5-year window
# below does that job equally well for a 1998 entrant as for a 2015 one.
# Restricting to 2010-2019 discarded half the sample for no methodological
# gain. Cohort dummies absorb entry-period effects.
WINDOW = 5                            # years counted from first patent
HISTORY_FROM = 1976                   # full USPTO history; kills left-truncation

# Founder rule: ALL THREE must hold for a company-assigned patent to be
# reclassified as independent (i.e. the inventor owns the firm).
FOUNDER_SHARE = 0.75    # inventor is on >=75% of the firm's patents
FOUNDER_FIRM_MIN = 5    # firm has >=5 patents -- you cannot infer ownership
                        # from a 1-patent firm, where EVERY inventor trivially
                        # passes all the other tests. This was the bug.
FOUNDER_FIRM_MAX = 20   # firm has <=20 patents total
FOUNDER_EARLY_N = 3     # inventor is on one of the firm's first 3 patents

# PatentsView assignee_type codes
TYPE_COMPANY = {"2", "3"}
TYPE_INDIVIDUAL = {"4", "5"}
TYPE_GOVT = {"6", "7", "8", "9"}

# ACADEMIC / TECHNOLOGY-TRANSFER EXCLUSIONS
#
# BUG FIXED: this pattern previously ended with \b after the alternation, so
# "univ" and "universit" could only match as WHOLE WORDS. "Tel Aviv
# University", "Ben Gurion University" and "The Hebrew University" all slipped
# through and were counted as ordinary corporate employers. Only institutions
# named explicitly (Yissum, Ramot, Technion, Weizmann) were being excluded.
#
# "universit" is used as a PREFIX (matching university / universite /
# universitat / universiteit) but "univ" alone is not, because it would also
# match Universal Display Corporation and similar real companies.
ACADEMIC = re.compile(
    r"(\buniversit"
    r"|\bcollege\b"
    r"|\bacadem(y|ic)\b"
    r"|\binstitute of (technology|science|higher)"
    r"|\btechnion\b|\bweizmann\b|\byissum\b|\bramot\b|\byeda\b"
    r"|\bbgn technolog|\bb\.?g\.? negev|\bbirad\b|\bhadasit\b"
    r"|\bmor research|\bcarmel.*haifa|\bariel scientific|\bvolcani\b"
    r"|\bresearch and development foundation"
    r"|\btechnology transfer"
    r"|\bmigal\b|\bgalilee research|\bshamir research"
    r"|\bopen university|\bbar[- ]?ilan|\bben[- ]?gurion"
    r"|\bhebrew university|\btel[- ]?aviv university"
    r"|\bafeka\b|\bsce\b|\bshenkar\b|\bholon institute"
    r"|\bgavish[- ]galilee|\bbio applications)",
    re.I,
)

GOVT_AGENCY = re.compile(
    r"\b(ministry|state of israel|municipal|national insurance)\b", re.I
)

_flow = []


def note(stage, **counts):
    _flow.append({"stage": stage, **counts})
    bits = "  ".join(f"{k}={v:,}" for k, v in counts.items())
    print(f"{stage:<44} {bits}")


def classify_assignee(row):
    """corporate | independent | excluded"""
    aid = str(row["assignee_id"] or "").strip()
    aname = str(row["assignee_name"] or "")
    atype = str(row["assignee_type"] or "").strip()

    if not aid and not aname:
        return "independent"          # unassigned -> the inventor owns it
    if atype in TYPE_INDIVIDUAL:
        return "independent"
    if ACADEMIC.search(aname) or GOVT_AGENCY.search(aname):
        return "excluded"
    if atype in TYPE_GOVT:
        return "excluded"
    if atype in TYPE_COMPANY or aid:
        return "corporate"
    return "excluded"


def apply_founder_rule(df):
    """
    Reclassify corporate -> independent where the inventor plausibly owns the
    firm. Operates on company-assigned patents only, per-firm.
    """
    # Only patents with a REAL assignee id can support the founder rule.
    # Blank ids were being grouped together as one enormous pseudo-firm,
    # which silently corrupted every firm-level share and rank below.
    corp = df[(df["kind"] == "corporate")
              & (df["assignee_id"].astype(str).str.strip() != "")].copy()
    if corp.empty:
        df["founder"] = False
        return df

    firm_n = corp.groupby("assignee_id")["patent_id"].nunique().rename("firm_patents")
    inv_n = (corp.groupby(["assignee_id", "inventor_id"])["patent_id"]
                 .nunique().rename("inv_patents").reset_index())
    inv_n = inv_n.merge(firm_n, on="assignee_id")
    inv_n["share"] = inv_n["inv_patents"] / inv_n["firm_patents"]

    # is the inventor on one of the firm's first FOUNDER_EARLY_N patents?
    # Computed once, with a stable sort, then ranked. Previously this built
    # the ordering from one sort chain and applied it to a separately
    # recomputed chain, relying on both producing identical row order.
    firm_pat = (corp.sort_values(["assignee_id", "filing_year", "patent_id"],
                                 kind="mergesort")
                    .drop_duplicates(["assignee_id", "patent_id"])
                    .copy())
    firm_pat["rank"] = firm_pat.groupby("assignee_id").cumcount()
    early_patents = firm_pat[firm_pat["rank"] < FOUNDER_EARLY_N]
    early_pairs = set(
        corp.merge(early_patents[["assignee_id", "patent_id"]],
                   on=["assignee_id", "patent_id"])
            .set_index(["assignee_id", "inventor_id"]).index
    )
    inv_n["early"] = [
        (a, i) in early_pairs
        for a, i in zip(inv_n["assignee_id"], inv_n["inventor_id"])
    ]

    inv_n["founder"] = (
        (inv_n["share"] >= FOUNDER_SHARE)
        & (inv_n["firm_patents"] >= FOUNDER_FIRM_MIN)
        & (inv_n["firm_patents"] <= FOUNDER_FIRM_MAX)
        & inv_n["early"]
    )
    flags = inv_n[inv_n["founder"]][["assignee_id", "inventor_id"]]
    df = df.merge(flags.assign(founder=True), on=["assignee_id", "inventor_id"],
                  how="left")
    df["founder"] = df["founder"].fillna(False)
    df.loc[(df["kind"] == "corporate") & df["founder"], "kind"] = "independent"
    return df


def modal(s):
    """Most frequent non-empty value. Ties broken alphabetically so that
    reruns are byte-identical."""
    v = s[s.astype(str).str.len() > 0]
    if v.empty:
        return ""
    counts = v.value_counts()
    top = counts.max()
    return sorted(counts[counts == top].index)[0]


def build(raw_path="raw_records.csv", out="panel.csv", flow_out="flow.csv"):
    df = pd.read_csv(raw_path, dtype=str).fillna("")
    df["filing_year"] = pd.to_numeric(df["filing_year"], errors="coerce")
    df = df.dropna(subset=["filing_year"])
    df["filing_year"] = df["filing_year"].astype(int)
    note("0. raw inventor-patent rows", rows=len(df),
         inventors=df["inventor_id"].nunique())

    # 1. drop unusable rows
    df = df[(df["inventor_id"] != "") & (df["inventor_last_name"] != "")]
    df = df[df["filing_year"].between(HISTORY_FROM, 2024)]
    note("1. valid id + name + year", rows=len(df),
         inventors=df["inventor_id"].nunique())

    # 2. classify each patent's ownership
    df["kind"] = df.apply(classify_assignee, axis=1)
    blank = int(((df["kind"] == "corporate")
                 & (df["assignee_id"].astype(str).str.strip() == "")).sum())
    if blank:
        print(f"    NOTE: {blank:,} corporate rows have a blank assignee id; "
              f"they are excluded from the founder rule and from firm FE.")
    note("2a. ownership classified",
         corporate=int((df.kind == "corporate").sum()),
         independent=int((df.kind == "independent").sum()),
         excluded_univ_govt=int((df.kind == "excluded").sum()))

    df = apply_founder_rule(df)
    note("2b. founder-owned reclassified as independent",
         founder_rows=int(df["founder"].sum()),
         corporate=int((df.kind == "corporate").sum()),
         independent=int((df.kind == "independent").sum()))

    # Show the biggest assignees on each side of the exclusion rule, so a
    # missed university or TTO is visible instead of silently corporate.
    ex = (df[df.kind == "excluded"].groupby("assignee_name")["patent_id"]
            .nunique().sort_values(ascending=False).head(12))
    inc = (df[df.kind == "corporate"].groupby("assignee_name")["patent_id"]
             .nunique().sort_values(ascending=False).head(15))
    print("\n    Largest EXCLUDED assignees (university / government):")
    for k, v in ex.items():
        print(f"      {str(k)[:52]:<54} {v:>6,}")
    print("\n    Largest INCLUDED assignees (counted as corporate) --")
    print("    check none of these is a university or tech-transfer arm:")
    for k, v in inc.items():
        print(f"      {str(k)[:52]:<54} {v:>6,}")
    print()

    # 3. university / government patents leave the sample entirely
    df = df[df["kind"] != "excluded"]
    note("3. university/govt dropped", rows=len(df),
         inventors=df["inventor_id"].nunique())

    # 4. first patent year over FULL history (this is the left-truncation fix)
    first = df.groupby("inventor_id")["filing_year"].min().rename("first_year")
    df = df.merge(first, on="inventor_id")

    # 5. cohort restriction
    df = df[df["first_year"].between(COHORT_MIN, COHORT_MAX)]
    note(f"4. cohort {COHORT_MIN}-{COHORT_MAX}", rows=len(df),
         inventors=df["inventor_id"].nunique())

    # 6. uniform WINDOW-year exposure from each inventor's first patent
    df = df[df["filing_year"] < df["first_year"] + WINDOW]
    note(f"5. {WINDOW}-year window applied", rows=len(df),
         inventors=df["inventor_id"].nunique())

    # 7. aggregate to the person
    g = df.groupby("inventor_id")
    panel = pd.DataFrame({
        "surname": g["inventor_last_name"].agg(modal),
        "first_year": g["first_year"].first(),
        "city": g["inventor_city"].agg(modal),
        "field": g["cpc_section"].agg(modal),
        "total_patents": g["patent_id"].nunique(),
        "corp_count": df[df.kind == "corporate"].groupby("inventor_id")["patent_id"].nunique(),
        "n_firms": df[df.kind == "corporate"].groupby("inventor_id")["assignee_id"].nunique(),
    }).reset_index()
    panel[["corp_count", "n_firms"]] = panel[["corp_count", "n_firms"]].fillna(0).astype(int)
    panel["indep_count"] = panel["total_patents"] - panel["corp_count"]
    panel["corp_share"] = panel["corp_count"] / panel["total_patents"]
    panel["ever_corporate"] = (panel["corp_count"] > 0).astype(int)
    panel["cohort"] = panel["first_year"].astype(str)

    # 8. controls
    panel["district"] = panel["city"].map(rg.to_district)
    panel["region"] = panel["district"].map(rg.to_coarse)
    panel["field"] = panel["field"].replace("", "unknown")
    note("6. inventor-level panel", inventors=len(panel),
         unknown_region=int((panel.district == "unknown").sum()),
         unknown_field=int((panel.field == "unknown").sum()))

    # 9. ethnicity -- hand lexicons PLUS surnames learned from residential
    #    segregation, which is what reaches the long tail of rare names
    learned = nm.learn_from_localities(panel["surname"], panel["city"])
    panel["ethnicity"] = nm.classify_series(panel["surname"], learned)
    panel["eth_rulebased"] = nm.classify_series(panel["surname"])  # robustness
    counts = panel["ethnicity"].value_counts()
    amb = counts.get("ambiguous", 0)
    note("7a. surname classified",
         arab=int(counts.get("arab", 0)),
         jewish=int(counts.get("jewish", 0)),
         ambiguous=int(amb))
    print(f"    ambiguous share = {amb / max(len(panel),1):.1%}  "
          f"(report this to Lia; the third bucket exists but is excluded)")

    panel.to_csv(out, index=False)
    analysis = panel[panel["ethnicity"].isin(["arab", "jewish"])]
    note("7b. analysis sample (clear names only)", inventors=len(analysis),
         arab=int((analysis.ethnicity == "arab").sum()))

    pd.DataFrame(_flow).to_csv(flow_out, index=False)
    print(f"\nWrote {out} ({len(panel):,} rows) and {flow_out}")
    return panel


if __name__ == "__main__":
    build(*(sys.argv[1:] or []))
