"""
How many Arab inventors are hiding in the excluded group?

    python coverage_floor.py

This settles a specific disagreement. If the ~18,000 unclassified inventors are
essentially all Jewish, then the Arab share of Israeli inventors is roughly
276/35,590 = 0.78%. If a meaningful number of them are Arab, it is higher.

The test uses NO name information at all, so it is independent of any
classifier. Among the EXCLUDED inventors, count how many live in an
Arab-majority locality — restricted to localities where the audit found the
geocoding to be reliable (Rahat, Kalanswa, Sachnin and others are dropped;
see regions.UNRELIABLE_GEOCODE).

An excluded inventor living in Nazareth or Tamra is Arab with near-certainty.
That gives a hard FLOOR on the number of hidden Arabs. It is only a floor,
because Arab inventors living in Haifa, Jerusalem or Tel Aviv are invisible to
it — so the true number is higher than whatever this prints.
"""
import pandas as pd

import regions as rg

LINE = "=" * 70


def main():
    p = pd.read_csv("panel.csv")
    p["city"] = p["city"].astype(str)
    p["arab_town"] = p["city"].map(rg.is_arab_locality)
    p["reliable"] = p["city"].map(rg.geocode_reliable)
    good = p["arab_town"] & p["reliable"]

    n_tot = len(p)
    exc = p[p.ethnicity == "ambiguous"]
    arab = p[p.ethnicity == "arab"]
    jew = p[p.ethnicity == "jewish"]

    print(LINE)
    print("WHO LIVES IN A RELIABLY-CODED ARAB LOCALITY?")
    print(LINE)
    print(f"  panel                      : {n_tot:,}")
    print(f"  excluded (unclassified)    : {len(exc):,} ({len(exc)/n_tot:.1%})")
    print(f"  classified Arab            : {len(arab):,}")
    print(f"  classified Jewish          : {len(jew):,}\n")

    for lab, sub in [("classified Arab", arab), ("classified Jewish", jew),
                     ("EXCLUDED", exc)]:
        m = good.reindex(sub.index).fillna(False)
        print(f"  {lab:<22} in Arab locality: {int(m.sum()):>5,} "
              f"({m.mean():.2%})")

    floor = int(good.reindex(exc.index).fillna(False).sum())
    print("\n" + LINE)
    print("THE FLOOR")
    print(LINE)
    print(f"  Excluded inventors living in a reliably-coded Arab locality:"
          f" {floor:,}")
    print("  These are Arab with near-certainty. So the excluded group is NOT")
    print("  all Jewish, and the true Arab count is at least:")
    print(f"      {len(arab):,} classified + {floor:,} floor = "
          f"{len(arab)+floor:,}")
    print(f"  implied Arab share of all inventors >= "
          f"{(len(arab)+floor)/n_tot:.2%}")
    print(f"  (versus {len(arab)/n_tot:.2%} if the excluded were ALL Jewish)")

    # how much of the Arab population does the locality test even see?
    seen = good.reindex(arab.index).fillna(False).mean()
    print("\n" + LINE)
    print("SCALING THE FLOOR UP")
    print(LINE)
    print(f"  Of KNOWN Arab inventors, only {seen:.1%} live in a reliably-coded")
    print("  Arab locality — the rest live in mixed cities and would be missed")
    print("  by this test. If hidden Arabs are geographically like known ones:")
    if seen > 0:
        est = floor / seen
        print(f"      estimated hidden Arabs ~ {floor:,} / {seen:.3f}"
              f" = {est:,.0f}")
        print(f"      implied total ~ {len(arab)+est:,.0f}")
        print(f"      implied Arab share ~ {(len(arab)+est)/n_tot:.2%}")
        print(f"      implied classifier recall ~ {len(arab)/(len(arab)+est):.0%}")
    print("\n  CAVEAT: hidden Arabs may be MORE urban than identified ones —")
    print("  a distinctive surname and a peripheral address plausibly go")
    print("  together. If so this scaling understates them and the true share")
    print("  is higher still. Report the FLOOR as the defensible number and")
    print("  the scaled figure as an estimate.")


if __name__ == "__main__":
    main()
