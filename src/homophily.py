"""
Is co-inventor homophily real, or an artefact of firm sorting?

    python homophily.py

Table E showed Arab inventors' co-inventors are ~28% Arab against a ~1.6%
base rate. Two very different stories produce that number:

  SORTING     Arab inventors are concentrated in a few firms. Teams form
              within firms, so homophily follows mechanically. Says nothing
              about behaviour inside any firm.

  SEGREGATION Within the SAME firm, Arab inventors are placed on different
              teams from Jewish colleagues. This is what H1 and H3 predict.

We separate them with two permutation tests:

  NULL A (global)      shuffle ethnicity labels across all inventors.
                       Observed >> Null A just means clustering exists.
  NULL B (within-firm) shuffle ethnicity labels ONLY among inventors of the
                       same firm, preserving each firm's ethnic composition
                       exactly. Observed >> Null B is within-firm segregation
                       and cannot be explained by sorting.

NULL B is the test that matters.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_panel as bp

N_PERM = 500
RNG = np.random.default_rng(20260827)
LINE = "=" * 70


def load():
    panel = pd.read_csv("panel.csv")
    raw = pd.read_csv("raw_records.csv", dtype=str)
    raw["filing_year"] = pd.to_numeric(raw["filing_year"], errors="coerce")

    eth = panel.set_index("inventor_id")["ethnicity"].to_dict()
    fy = panel.set_index("inventor_id")["first_year"].to_dict()
    raw["eth"] = raw["inventor_id"].map(eth)
    raw["first_year"] = raw["inventor_id"].map(fy)
    raw = raw[raw["eth"].isin(["arab", "jewish"])]
    raw = raw[(raw.filing_year >= raw.first_year)
              & (raw.filing_year < raw.first_year + bp.WINDOW)]

    # corporate patents at a real firm -- the only place "same firm" means
    # anything
    raw["kind"] = raw.apply(bp.classify_assignee, axis=1)
    raw = raw[(raw.kind == "corporate")
              & (raw.assignee_id.astype(str).str.strip() != "")]

    # keep patents with at least two classified inventors
    sizes = raw.groupby("patent_id")["inventor_id"].transform("size")
    raw = raw[sizes >= 2]
    return panel, raw


def homophily(df, eth_col="eth"):
    """Mean % of an Arab inventor's co-inventors who are Arab."""
    is_arab = (df[eth_col] == "arab").astype(int)
    g = df.assign(_a=is_arab).groupby("patent_id")["_a"]
    n_arab = g.transform("sum")
    n_tot = g.transform("size")
    co_arab = n_arab - is_arab
    co_tot = n_tot - 1
    sub = pd.DataFrame({"inv": df["inventor_id"], "is_arab": is_arab,
                        "co_arab": co_arab, "co_tot": co_tot})
    sub = sub[sub.co_tot > 0]
    per_inv = sub.groupby(["inv", "is_arab"]).agg(
        ca=("co_arab", "sum"), ct=("co_tot", "sum")).reset_index()
    per_inv = per_inv[per_inv.ct > 0]
    per_inv["pct"] = 100 * per_inv.ca / per_inv.ct
    a = per_inv[per_inv.is_arab == 1]["pct"]
    j = per_inv[per_inv.is_arab == 0]["pct"]
    return (a.mean() if len(a) else np.nan,
            j.mean() if len(j) else np.nan, len(a), len(j))


def main():
    panel, raw = load()
    inv_eth = raw.drop_duplicates("inventor_id").set_index("inventor_id")["eth"]
    base = 100 * (inv_eth == "arab").mean()

    obs_a, obs_j, n_a, n_j = homophily(raw)
    if not np.isfinite(obs_a) or n_a < 5:
        print("Not enough Arab inventors with co-inventors to run this test "
              f"(found {n_a}). Nothing to report.")
        return
    print(LINE)
    print("OBSERVED")
    print(LINE)
    print(f"  patents analysed              : {raw.patent_id.nunique():,}")
    print(f"  firms                         : {raw.assignee_id.nunique():,}")
    print(f"  classified inventors          : {len(inv_eth):,}")
    print(f"  Arab share of these inventors : {base:.2f}%")
    print(f"\n  Arab inventors   (n={n_a:,}): co-inventors are "
          f"{obs_a:.2f}% Arab")
    print(f"  Jewish inventors (n={n_j:,}): co-inventors are "
          f"{obs_j:.2f}% Arab")
    print(f"  ratio to base rate: {obs_a/base:.1f}x")

    # ---------------------------------------------------------- NULL A
    print("\n" + LINE)
    print("NULL A -- shuffle ethnicity globally")
    print(LINE)
    inv_ids = inv_eth.index.to_numpy()
    labels = inv_eth.to_numpy()
    draws = []
    for _ in range(N_PERM):
        perm = RNG.permutation(labels)
        m = dict(zip(inv_ids, perm))
        d = raw.assign(_e=raw.inventor_id.map(m))
        draws.append(homophily(d, "_e")[0])
    draws = np.array([d for d in draws if np.isfinite(d)])
    if not len(draws):
        print("  permutation produced no valid draws -- skipping")
        return
    pA = (draws >= obs_a).mean()
    print(f"  null mean {draws.mean():.2f}%  "
          f"(95% range {np.percentile(draws,2.5):.2f}-"
          f"{np.percentile(draws,97.5):.2f})")
    print(f"  observed {obs_a:.2f}%   p = {pA:.4f}")
    print("  -> clustering exists" if pA < .05 else "  -> no clustering")

    # ---------------------------------------------------------- NULL B
    print("\n" + LINE)
    print("NULL B -- shuffle ONLY within firm (the test that matters)")
    print(LINE)
    # an inventor's firm = the firm they patent at most often in the window
    firm = (raw.groupby(["inventor_id", "assignee_id"]).size()
               .reset_index(name="k")
               .sort_values(["inventor_id", "k", "assignee_id"],
                            ascending=[True, False, True],
                            kind="mergesort")
               .drop_duplicates("inventor_id")
               .set_index("inventor_id")["assignee_id"])
    tab = pd.DataFrame({"eth": inv_eth, "firm": firm.reindex(inv_eth.index)})
    tab = tab.dropna()

    mixed = tab.groupby("firm")["eth"].nunique()
    n_mixed = int((mixed == 2).sum())
    print(f"  firms with inventors of both groups: {n_mixed:,} "
          f"of {tab.firm.nunique():,}")
    if n_mixed < 10:
        print("  Too few mixed firms to run this test. That is itself the")
        print("  finding: Arab and Jewish inventors barely share employers,")
        print("  so homophily is SORTING, not within-firm segregation.")
        return

    draws_b = []
    grp = tab.groupby("firm")
    for _ in range(N_PERM):
        shuffled = grp["eth"].transform(lambda s: RNG.permutation(s.to_numpy()))
        m = dict(zip(tab.index, shuffled))
        d = raw.assign(_e=raw.inventor_id.map(m))
        d = d[d["_e"].notna()]
        draws_b.append(homophily(d, "_e")[0])
    draws_b = np.array([d for d in draws_b if np.isfinite(d)])
    pB = (draws_b >= obs_a).mean()
    print(f"  null mean {draws_b.mean():.2f}%  "
          f"(95% range {np.percentile(draws_b,2.5):.2f}-"
          f"{np.percentile(draws_b,97.5):.2f})")
    print(f"  observed {obs_a:.2f}%   p = {pB:.4f}")

    print("\n  INTERPRETATION")
    if pB < 0.05:
        print("  Homophily SURVIVES holding firm composition fixed.")
        print("  Within the same firm, Arab inventors are on different teams")
        print("  from Jewish colleagues. This is direct evidence for H1/H3.")
    else:
        print("  Homophily does NOT survive holding firm composition fixed.")
        print("  The 28% is explained by WHICH FIRMS Arab inventors work at,")
        print("  not by team formation inside them. That is a finding about")
        print("  labour-market sorting, not about intrapreneurial barriers.")
        print("  Do not present it as evidence for H1 or H3.")

    # ------------------------------------------------- concentration check
    print("\n" + LINE)
    print("CONCENTRATION -- is this a handful of firms?")
    print(LINE)
    ar = raw[raw.eth == "arab"]
    top = ar.groupby("assignee_id")["inventor_id"].nunique().sort_values(
        ascending=False)
    tot = top.sum()
    print(f"  firms employing >=1 Arab inventor: {len(top):,}")
    for k in (1, 3, 5, 10):
        if len(top) >= k:
            print(f"  top {k:>2} firms hold {100*top.head(k).sum()/tot:>5.1f}% "
                  f"of Arab inventor-firm links")
    print("\n  If the top few firms hold most of them, the result is about")
    print("  a small number of workplaces and should be described that way.")

    names = raw.drop_duplicates("assignee_id").set_index(
        "assignee_id")["assignee_name"]
    print("\n  Largest employers of Arab inventors:")
    for a, k in top.head(10).items():
        print(f"    {str(names.get(a, a))[:44]:<46} {k:>4}")

    _figure(base, draws.mean(), draws_b, obs_a, obs_j)


def _figure(base, nullA, draws_b, obs_a, obs_j):
    """The decomposition in one image: base rate -> sorting -> segregation."""
    nullB = draws_b.mean()
    lo, hi = np.percentile(draws_b, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = ["Random teams\n(base rate)",
              "After firm sorting\n(within-firm null)",
              "Observed"]
    vals = [base, nullB, obs_a]
    cols = ["#bdc3c7", "#e8a33d", "#c0392b"]
    bars = ax.bar(labels, vals, width=.55, color=cols)

    # 95% band of the within-firm null
    ax.errorbar(1, nullB, yerr=[[nullB - lo], [hi - nullB]], fmt="none",
                ecolor="#7a5620", capsize=6, lw=1.6)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + .6, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(0.72, nullB), xytext=(0.28, base),
                arrowprops=dict(arrowstyle="->", color="#7a5620", lw=1.4))
    ax.text(0.5, (base + nullB) / 2 + 1.4, "firm sorting",
            ha="center", fontsize=9, color="#7a5620")
    ax.annotate("", xy=(1.72, obs_a), xytext=(1.28, nullB),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
    ax.text(1.5, (nullB + obs_a) / 2 + 1.0, "within-firm\nsegregation",
            ha="center", fontsize=9, color="#c0392b")

    ax.axhline(obs_j, ls=":", lw=1.2, color="#2c3e50")
    ax.text(2.40, obs_j + 1.6, f"Jewish inventors: {obs_j:.1f}%",
            fontsize=8, color="#2c3e50", ha="center")

    ax.set_ylabel("% of an Arab inventor's co-inventors who are Arab")
    ax.set_title("Figure 9. Where co-inventor segregation comes from")
    ax.set_ylim(0, max(obs_a * 1.28, 12))
    fig.tight_layout()
    fig.savefig("fig9_team_composition.png", dpi=200)
    plt.close(fig)
    print("\n  wrote fig9_team_composition.png")


if __name__ == "__main__":
    main()
