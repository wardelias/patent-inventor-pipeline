"""
Self-test. Verifies invariants that MUST hold if the pipeline is correct.

    python selftest.py

Every check is a statement that would be false if a specific bug existed.
Run this after any change to names.py, regions.py or build_panel.py.
Exit code 0 = all passed.
"""
import sys

import numpy as np
import pandas as pd

import names as nm
import regions as rg
import build_panel as bp

PASS, FAIL, WARN = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def warn(name, cond, detail=""):
    if not cond:
        WARN.append(name)
    print(f"  [{'ok  ' if cond else 'WARN'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


# ===================================================== 1. classifier logic
def test_names():
    print("\n1. CLASSIFIER LOGIC")

    for lex, lab in [(nm.ARAB_LEXICON, "arab"), (nm.JEWISH_LEXICON, "jewish"),
                     (nm.AMBIGUOUS_LEXICON, "ambiguous")]:
        bad = [w for w in lex if nm.classify(w) != lab]
        check(f"every {lab} lexicon entry classifies as {lab}",
              not bad, f"{len(bad)} bad: {bad[:6]}")

    check("lexicons are disjoint",
          not (nm.ARAB_LEXICON & nm.JEWISH_LEXICON)
          and not (nm.ARAB_LEXICON & nm.AMBIGUOUS_LEXICON)
          and not (nm.JEWISH_LEXICON & nm.AMBIGUOUS_LEXICON))

    known = {
        "Abu Ahmad": "arab", "Abu Salem": "arab", "Al Masri": "arab",
        "El Hakim": "arab", "Abdel Rahman": "arab", "Masarwa": "arab",
        "Khoury": "arab", "Mahajna": "arab", "Zoabi": "arab",
        "Abu Hatzeira": "ambiguous", "Abulafia": "ambiguous",
        "Abudi": "ambiguous", "Sharaby": "ambiguous", "Meimoun": "ambiguous",
        "Haddad": "ambiguous", "Amar": "ambiguous", "Halabi": "ambiguous",
        "Cohen": "jewish", "Levi": "jewish", "Rosenberg": "jewish",
        "Peled": "jewish", "Vaknin": "jewish", "Alon": "jewish",
        "Elbaz": "jewish", "Neeman": "jewish", "Maayan": "jewish",
    }
    bad = [(k, nm.classify(k), v) for k, v in known.items()
           if nm.classify(k) != v]
    check("known-answer name set", not bad, str(bad[:5]))

    check("classification is deterministic",
          all(nm.classify(k) == nm.classify(k) for k in known))
    check("case and punctuation invariant",
          nm.classify("ABU-AHMAD") == nm.classify("abu ahmad") == "arab")
    check("empty input is ambiguous, not a crash",
          nm.classify("") == "ambiguous" and nm.classify(None) == "ambiguous")


# ======================================================= 2. region logic
def test_academic():
    print("\n1b. ACADEMIC / TTO EXCLUSION")
    must = ["Tel Aviv University", "The Hebrew University of Jerusalem",
            "Bar Ilan University", "BEN GURION UNIVERSITY OF THE NEGEV",
            "Yissum Research Development Company", "Weizmann Institute of Science",
            "Technion Research and Development Foundation",
            "Ramot at Tel Aviv University", "Open University of Israel"]
    never = ["Intel Corporation", "Universal Display Corporation",
             "Mellanox Technologies, Ltd", "Check Point Software Technologies",
             "TEVA PHARMACEUTICAL INDUSTRIES, LTD.",
             "Applied Materials Israel Ltd"]
    fn = [x for x in must if not bp.ACADEMIC.search(x)]
    fp = [x for x in never if bp.ACADEMIC.search(x)]
    check("all universities and TTOs are excluded", not fn, str(fn))
    check("no ordinary company is wrongly excluded", not fp, str(fp))


def test_regions():
    print("\n2. REGION LOGIC")
    arabs = ["Nazareth", "Sakhnin", "Sachnin", "Tamra", "Umm al-Fahm",
             "Rahat", "Taibe", "Kfar Kara", "Kalanswa", "Deir al Asad"]
    jews = ["Nazareth Illit", "Nof HaGalil", "Timrat", "Kfar Saba",
            "Kfar Sava", "Herzliya", "Tirat Carmel", "Ramat Gan", "Meitar",
            "Omer", "Yeruham", "Harish", "Nofit", "Kfar Tavor"]
    fn = [c for c in arabs if not rg.is_arab_locality(c)]
    fp = [c for c in jews if rg.is_arab_locality(c)]
    check("all known Arab localities detected", not fn, str(fn))
    check("no Jewish locality flagged Arab", not fp, str(fp))

    check("Nazareth Illit is NOT Nazareth",
          rg.is_arab_locality("Nazareth") and
          not rg.is_arab_locality("Nazareth Illit"))
    check("unreliable-geocode list is honoured",
          not rg.geocode_reliable("Rahat")
          and rg.geocode_reliable("Nazareth"))
    check("every district value is valid",
          all(rg.to_district(c) in
              {"north", "haifa", "center", "tel aviv", "jerusalem",
               "south", "unknown"} for c in arabs + jews))
    check("empty city does not crash", rg.to_district("") == "unknown")


# ==================================================== 3. panel invariants
def test_panel():
    print("\n3. PANEL INVARIANTS")
    try:
        p = pd.read_csv("panel.csv")
    except FileNotFoundError:
        print("  (panel.csv not found -- run START_HERE.py first)")
        return

    check("one row per inventor", p["inventor_id"].is_unique)
    check("corp_count + indep_count == total_patents",
          bool((p.corp_count + p.indep_count == p.total_patents).all()))
    check("corp_share in [0, 1]",
          bool(p.corp_share.between(0, 1).all()))
    check("no negative counts",
          bool((p[["corp_count", "indep_count", "total_patents"]] >= 0)
               .all().all()))
    check("total_patents >= 1 for everyone",
          bool((p.total_patents >= 1).all()))
    check("ever_corporate agrees with corp_count",
          bool(((p.corp_count > 0).astype(int) == p.ever_corporate).all()))
    check("n_firms <= corp_count",
          bool((p.n_firms <= p.corp_count).all()))
    check(f"cohorts within {bp.COHORT_MIN}-{bp.COHORT_MAX}",
          bool(p.first_year.between(bp.COHORT_MIN, bp.COHORT_MAX).all()))
    check("ethnicity has only the three valid values",
          set(p.ethnicity.dropna()) <= {"arab", "jewish", "ambiguous"})
    check("panel ethnicity matches a fresh classification",
          bool((p["surname"].astype(str).map(nm.classify)
                == p["ethnicity"]).mean() > 0.999))

    amb = (p.ethnicity == "ambiguous").mean()
    warn(f"ambiguous share is {amb:.1%}", amb < 0.55,
         "over half the sample unclassified")
    unk = (p.district == "unknown").mean()
    warn(f"unknown region share is {unk:.1%}", unk < 0.10)


# ==================================================== 4. raw consistency
def test_raw():
    print("\n4. RAW vs PANEL CONSISTENCY")
    try:
        raw = pd.read_csv("raw_records.csv", dtype=str)
        p = pd.read_csv("panel.csv")
    except FileNotFoundError:
        print("  (files not found -- skipping)")
        return

    raw["filing_year"] = pd.to_numeric(raw["filing_year"], errors="coerce")
    check("every panel inventor exists in raw",
          set(p.inventor_id.astype(str)) <= set(raw.inventor_id.astype(str)))

    # recompute total_patents independently for a sample of inventors
    fy = p.set_index("inventor_id")["first_year"].to_dict()
    samp = p.sample(min(300, len(p)), random_state=0)
    r = raw[raw.inventor_id.isin(samp.inventor_id.astype(str))].copy()
    r["fy0"] = r.inventor_id.map(fy)
    r = r[(r.filing_year >= r.fy0) & (r.filing_year < r.fy0 + bp.WINDOW)]
    r["kind"] = r.apply(bp.classify_assignee, axis=1)
    r = r[r.kind != "excluded"]
    recomputed = r.groupby("inventor_id")["patent_id"].nunique()
    merged = samp.set_index("inventor_id")["total_patents"].reindex(
        recomputed.index)
    agree = float((recomputed == merged).mean())
    check("total_patents recomputes from raw (300 sampled inventors)",
          agree > 0.99, f"only {agree:.1%} agree")

    dup = raw.duplicated(["patent_id", "inventor_id"]).sum()
    warn(f"{dup:,} duplicate (patent, inventor) rows in raw", dup == 0)


# ======================================================= 5. determinism
def test_determinism():
    print("\n5. DETERMINISM")
    try:
        p = pd.read_csv("panel.csv")
    except FileNotFoundError:
        print("  (panel.csv not found -- skipping)")
        return
    s = p["surname"].astype(str).head(2000)
    check("classifier is stable across repeated calls",
          list(s.map(nm.classify)) == list(s.map(nm.classify)))
    c = p["city"].astype(str).head(2000)
    check("region mapping is stable across repeated calls",
          list(c.map(rg.to_district)) == list(c.map(rg.to_district)))


def main():
    print("=" * 66)
    print("SELF-TEST")
    print("=" * 66)
    test_names()
    test_academic()
    test_regions()
    test_panel()
    test_raw()
    test_determinism()

    print("\n" + "=" * 66)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(WARN)} warnings")
    print("=" * 66)
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
        print("\nA failure here means a result could be wrong. Send this to Claude.")
        sys.exit(1)
    if WARN:
        print("\nWarnings (not bugs, but worth knowing):")
        for w in WARN:
            print(f"  - {w}")
    print("\nAll invariants hold.")


if __name__ == "__main__":
    main()
