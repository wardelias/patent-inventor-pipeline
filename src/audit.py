"""
Full audit of the classification. Nothing is changed -- everything is dumped
for you to inspect by eye.

    python audit.py

Writes four files:
    audit_arab_names.csv       every surname called Arab
    audit_jewish_names.csv     every surname called Jewish (top 600)
    audit_suspicious.csv       likely misclassifications, both directions
    audit_cities.csv           every city, its district, Arab-locality flag

Read audit_suspicious.csv first. It is the short list.
"""
import re

import pandas as pd

import names as nm
import regions as rg

LINE = "=" * 70

# Arabic orthographic fragments. Presence is suggestive, not decisive --
# 'stein' contains no Arabic, but 'Hussein' ends in -sein.
ARABIC_HINTS = re.compile(
    r"(^abu|^abd|^abed|^abo|^al[- ]|^el[- ]|^ibn|^bin[- ]"
    r"|kh|dh|gh|q(?![u])|aa|ee(?=[dnmrl])"
    r"|(awi|awy|iyeh|ieh|oun|een|eddin|allah|ullah)$)", re.I)

HEBREW_HINTS = re.compile(
    r"((berg|stein|baum|feld|mann|witz|vitch|ovich|sky|ski|son|sohn|"
    r"zon|thal|heim|burger|hauser)$|^ben[- ]|^bar[- ]|^rosen|^gold|^silber)",
    re.I)


def main():
    p = pd.read_csv("panel.csv")
    p["surname"] = p["surname"].astype(str)
    p["city"] = p["city"].astype(str)
    p["norm"] = p["surname"].map(nm.normalize)
    p["arab_town"] = p["city"].map(rg.is_arab_locality)

    # ------------------------------------------------------------ 1 names
    g = (p.groupby(["norm", "ethnicity"])
           .agg(surname=("surname", lambda x: x.mode().iloc[0]),
                n_inventors=("surname", "size"),
                n_arab_town=("arab_town", "sum"))
           .reset_index()
           .sort_values("n_inventors", ascending=False))

    arab = g[g.ethnicity == "arab"]
    jew = g[g.ethnicity == "jewish"]
    amb = g[g.ethnicity == "ambiguous"]

    arab.to_csv("audit_arab_names.csv", index=False)
    jew.head(600).to_csv("audit_jewish_names.csv", index=False)

    print(LINE)
    print("1. WHAT THE CLASSIFIER CALLS ARAB")
    print(LINE)
    print(f"  distinct surnames: {len(arab):,}   inventors: "
          f"{int(arab.n_inventors.sum()):,}")
    print("\n  Most common (check these for false positives):")
    for _, r in arab.head(40).iterrows():
        flag = "  <- lives in Arab town" if r.n_arab_town else ""
        print(f"    {r.surname:<22} {int(r.n_inventors):>4}{flag}")

    # ------------------------------------------------------- 2 suspicious
    print("\n" + LINE)
    print("2. SUSPICIOUS ASSIGNMENTS")
    print(LINE)
    sus = []

    # (a) called Jewish but looks Arabic
    for _, r in jew.iterrows():
        if ARABIC_HINTS.search(r.norm) and not HEBREW_HINTS.search(r.norm):
            sus.append({**r.to_dict(), "issue": "called JEWISH, looks Arabic"})

    # (b) called Arab but looks Hebrew/Ashkenazi
    for _, r in arab.iterrows():
        if HEBREW_HINTS.search(r.norm):
            sus.append({**r.to_dict(), "issue": "called ARAB, looks Hebrew"})

    # (c) ambiguous, but carriers concentrated in Arab towns
    for _, r in amb.iterrows():
        if r.n_arab_town >= 1 and r.n_arab_town / r.n_inventors >= 0.5:
            sus.append({**r.to_dict(),
                        "issue": "AMBIGUOUS, but lives in Arab town"})

    # (d) called Jewish, yet every carrier lives in an Arab town
    for _, r in jew.iterrows():
        if r.n_arab_town and r.n_arab_town == r.n_inventors:
            sus.append({**r.to_dict(),
                        "issue": "called JEWISH, all carriers in Arab town"})

    s = pd.DataFrame(sus)
    if len(s):
        s = s.sort_values("n_inventors", ascending=False)
        s["verdict"] = ""
        s.to_csv("audit_suspicious.csv", index=False)
        print(f"  {len(s):,} flagged -> audit_suspicious.csv")
        print(f"  covering {int(s.n_inventors.sum()):,} inventors\n")
        for issue, sub in s.groupby("issue"):
            print(f"  {issue}  ({len(sub)} names, "
                  f"{int(sub.n_inventors.sum()):,} inventors)")
            for _, r in sub.head(20).iterrows():
                print(f"      {r.surname:<22} {int(r.n_inventors):>4}")
            print()
    else:
        print("  none flagged")

    # ----------------------------------------------------------- 3 cities
    print(LINE)
    print("3. CITY MAPPING")
    print(LINE)
    c = (p.groupby("city")
           .agg(n_inventors=("city", "size"),
                n_arab_named=("ethnicity", lambda x: (x == "arab").sum()))
           .reset_index())
    c["district"] = c["city"].map(rg.to_district)
    c["arab_locality"] = c["city"].map(rg.is_arab_locality)
    c["pct_arab_named"] = (100 * c.n_arab_named / c.n_inventors).round(1)
    c = c.sort_values("n_inventors", ascending=False)
    c.to_csv("audit_cities.csv", index=False)

    print(f"  distinct cities: {len(c):,}")
    print(f"  unmapped to a district: "
          f"{int(c[c.district=='unknown'].n_inventors.sum()):,} inventors")
    print(f"  flagged Arab localities: {int(c.arab_locality.sum()):,} cities, "
          f"{int(c[c.arab_locality].n_inventors.sum()):,} inventors")

    print("\n  Cities flagged ARAB -- check none are Jewish towns:")
    for _, r in c[c.arab_locality].head(30).iterrows():
        print(f"    {r.city:<28} {int(r.n_inventors):>5} inventors, "
              f"{r.pct_arab_named:>5}% Arab-named, district={r.district}")

    print("\n  Cities NOT flagged Arab but >30% Arab-named "
          "(possible missed Arab localities):")
    miss = c[(~c.arab_locality) & (c.pct_arab_named > 30) & (c.n_inventors >= 3)]
    if len(miss):
        for _, r in miss.head(30).iterrows():
            print(f"    {r.city:<28} {int(r.n_inventors):>5} inventors, "
                  f"{r.pct_arab_named:>5}% Arab-named")
    else:
        print("    none")

    print("\n  Largest unmapped cities:")
    for _, r in c[c.district == "unknown"].head(20).iterrows():
        print(f"    {r.city:<28} {int(r.n_inventors):>5}")

    print("\n" + LINE)
    print("SEND BACK: audit_suspicious.csv and audit_cities.csv")
    print(LINE)


if __name__ == "__main__":
    main()
