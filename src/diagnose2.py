"""
Step 3b -- is n_arab = 179 real, or a classification failure?

    python diagnose2.py

Reads panel.csv only. Changes nothing.
"""
import numpy as np
import pandas as pd

import names as nm
import regions as rg

LINE = "=" * 70


def main():
    p = pd.read_csv("panel.csv")
    p["surname"] = p["surname"].astype(str)
    p["city"] = p["city"].astype(str)
    n = len(p)

    # ---------------------------------------------------------------- 1
    print(LINE)
    print("1. NAME-FREE ESTIMATE: who LIVES in an Arab-majority locality?")
    print(LINE)
    p["arab_town"] = p["city"].map(rg.is_arab_locality)
    n_town = int(p["arab_town"].sum())
    print(f"  inventors in the panel              : {n:,}")
    print(f"  living in an Arab-majority locality : {n_town:,} ({n_town/n:.2%})")
    print(f"  classified Arab by surname          : {int((p.ethnicity=='arab').sum()):,}")
    print()
    print("  This count uses NO name information. If it is far above 179, the")
    print("  surname classifier is under-detecting. If it is close to 179, then")
    print("  Arab inventors really are that rare in USPTO data.")

    cross = pd.crosstab(p["arab_town"], p["ethnicity"])
    print("\n  Cross-tab (rows: lives in Arab town)")
    print(cross.to_string())

    if n_town:
        sub = p[p["arab_town"]]
        rec = (sub.ethnicity == "arab").mean()
        print(f"\n  Of inventors in Arab towns, share classified Arab: {rec:.1%}")
        print(f"  ^ a rough RECALL estimate for the classifier.")
        if rec < 0.6:
            print("    LOW -- the classifier is missing most Arab inventors.")

    print("\n  Top localities among inventors in Arab towns:")
    if n_town:
        for c, k in p[p.arab_town]["city"].value_counts().head(12).items():
            print(f"    {c:<28} {k:>5,}")

    # ---------------------------------------------------------------- 2
    print("\n" + LINE)
    print("2. WHY THE LEARNER FOUND ONLY 5 ARAB SURNAMES")
    print(LINE)
    if n_town:
        sn = p[p.arab_town]["surname"].map(nm.normalize).value_counts()
        print(f"  distinct surnames in Arab towns     : {len(sn):,}")
        for k in (1, 2, 3, 5):
            print(f"  surnames with >= {k} carriers        : {(sn >= k).sum():,}")
        print("\n  The learner needs min_n carriers. With so few Arab-town")
        print("  inventors, almost every surname has 1-2 and is skipped.")

    print("\n  Sensitivity -- Arab surnames learned at different thresholds:")
    for min_n in (2, 3, 5):
        for hi in (0.5, 0.7):
            L = nm.learn_from_localities(p["surname"], p["city"],
                                         min_n=min_n, hi=hi, verbose=False)
            na = sum(v == "arab" for v in L.values())
            eth = nm.classify_series(p["surname"], L)
            print(f"    min_n={min_n} hi={hi:.1f} -> {na:>4,} Arab surnames, "
                  f"n_arab = {int((eth=='arab').sum()):>5,}, "
                  f"ambiguous = {(eth=='ambiguous').mean():.1%}")

    # ---------------------------------------------------------------- 3
    print("\n" + LINE)
    print("3. WHAT IS LEFT IN THE AMBIGUOUS PILE?")
    print(LINE)
    amb = p[p.ethnicity == "ambiguous"]
    print(f"  ambiguous inventors : {len(amb):,} ({len(amb)/n:.1%})")
    print(f"  of those, in Arab towns: {int(amb['arab_town'].sum()):,}")
    print(f"  distinct surnames   : {amb['surname'].nunique():,}")
    sizes = amb["surname"].value_counts()
    print(f"  surnames held by 1 person : {(sizes==1).sum():,} "
          f"({(sizes==1).sum()/len(sizes):.0%} of distinct names)")
    print("\n  Most common remaining ambiguous names:")
    for s, k in sizes.head(20).items():
        print(f"    {s:<24} {k:>4}")

    # ---------------------------------------------------------------- 4
    print("\n" + LINE)
    print("4. POWER -- what could this sample have detected?")
    print(LINE)
    a = p[p.ethnicity == "arab"]
    j = p[p.ethnicity == "jewish"]
    for col, label in [("corp_share", "corporate share"),
                       ("ever_corporate", "ever corporate"),
                       ("corp_count", "corporate count")]:
        se = np.sqrt(a[col].var()/len(a) + j[col].var()/len(j))
        mde = 2.8 * se
        print(f"  {label:<18} diff = {a[col].mean()-j[col].mean():+.4f}   "
              f"se = {se:.4f}   MDE(80%) = {mde:.4f}")
    print("\n  MDE = smallest true gap this sample could reliably detect.")
    print("  A tight null (diff << MDE) is evidence of no gap.")
    print("  A null with MDE larger than any plausible effect is NO evidence.")

    # ---------------------------------------------------------------- 5
    print("\n" + LINE)
    print("5. THE DESCRIPTIVE FACT (no model needed)")
    print(LINE)
    clear = p[p.ethnicity.isin(["arab", "jewish"])]
    print(f"  Arab share of classified Israeli inventors: "
          f"{(clear.ethnicity=='arab').mean():.2%}")
    print(f"  Arab share of Israeli population         : ~20%")
    print("\n  Arab share by cohort decade:")
    clear = clear.copy()
    clear["decade"] = (clear["first_year"] // 5) * 5
    tab = clear.groupby("decade")["ethnicity"].agg(
        n="count", arab=lambda x: (x == "arab").sum())
    tab["arab_pct"] = (100 * tab["arab"] / tab["n"]).round(2)
    print(tab.to_string())

    print("\n" + LINE)


if __name__ == "__main__":
    main()
