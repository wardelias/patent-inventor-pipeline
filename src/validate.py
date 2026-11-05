"""
Validate the surname classifier. Two raters, correct reweighting.

    python validate.py sample     -> writes validation_sample.csv
    <both of you fill in your columns by hand>
    python validate.py score      -> precision, recall, F1, kappa

WHY THE REWEIGHTING MATTERS
The sample is stratified by PREDICTED class -- equal numbers of predicted-Arab,
predicted-Jewish and predicted-ambiguous names. Under that design:

  * Precision is directly estimable. Of names we call Arab, what share are?
  * Recall is NOT. True Arabs hiding in the predicted-Jewish and
    predicted-ambiguous buckets are massively under-sampled relative to their
    real numbers, so a raw count from the sample would overstate recall badly.

So we estimate P(true = t | predicted = c) from the sample, then reweight by
the true size of each predicted class in the panel:

    N_true_t = SUM_c  P(true = t | pred = c) * N_c

and recall_t = correctly-called t / N_true_t. That is what this script does.

Estimates are at the INVENTOR level, not the name level, because a surname
carried by 50 people matters 50x more than one carried by a single person.
"""
import sys

import numpy as np
import pandas as pd

import names as nm

PER_CLASS = 150   # per predicted class; recall estimates are noisy below this
FILE = "validation_sample.csv"
RATERS = ["label_rater1", "label_rater2"]
CLASSES = ["arab", "jewish", "ambiguous"]


# --------------------------------------------------------------- sampling
def make_sample(panel="panel.csv", per_class=PER_CLASS, seed=11):
    p = pd.read_csv(panel)
    p["surname"] = p["surname"].astype(str)

    rows = []
    sizes = {}
    for c in CLASSES:
        sub = p[p.ethnicity == c]
        sizes[c] = len(sub)
        if not len(sub):
            continue
        # sample INVENTORS, then collapse to distinct surnames so you only
        # label each name once. The weight is how many sampled inventors it
        # stands for.
        draw = sub.sample(min(per_class, len(sub)), random_state=seed)
        w = draw["surname"].str.strip().str.title().value_counts()
        for surname, k in w.items():
            rows.append({"surname": surname, "predicted": c,
                         "sampled_inventors": int(k),
                         "class_size": sizes[c]})

    out = pd.DataFrame(rows).sample(frac=1, random_state=seed)  # shuffle
    for r in RATERS:
        out[r] = ""
    out.to_csv(FILE, index=False)

    print(f"Wrote {FILE}: {len(out)} names to label.")
    print(f"  predicted-class sizes in the panel: {sizes}")
    print()
    print("HOW TO LABEL")
    print("  1. Open validation_sample.csv in Excel.")
    print("  2. Ward fills label_rater1. Mtanis fills label_rater2.")
    print("     Do it INDEPENDENTLY -- do not look at each other's column,")
    print("     and ignore the 'predicted' column entirely. The disagreement")
    print("     rate between you is a number worth reporting.")
    print("  3. Use exactly: arab / jewish / ambiguous")
    print("     'ambiguous' means a name genuinely borne by both communities")
    print("     (Haddad, Amar, Halabi), NOT 'I am unsure'. If you simply do")
    print("     not know the name, leave the cell EMPTY.")
    print("  4. Save as CSV -- not .xlsx. Excel will nag; choose Keep CSV.")
    print()
    print("  One rater is enough to get numbers. Two gives you reliability.")


# ---------------------------------------------------------------- scoring
def _consensus(df):
    """Agreed label where raters agree; NaN where they disagree or are blank."""
    have = [r for r in RATERS if r in df.columns and df[r].notna().any()]
    if not have:
        return None, have
    for r in have:
        df[r] = df[r].astype(str).str.strip().str.lower().replace("nan", "")
    if len(have) == 1:
        lab = df[have[0]].replace("", np.nan)
    else:
        a, b = df[have[0]], df[have[1]]
        lab = np.where((a == b) & (a != ""), a, np.nan)
        # fall back to whichever single rater labelled it
        lab = pd.Series(lab, index=df.index)
        only_a = (b == "") & (a != "")
        only_b = (a == "") & (b != "")
        lab[only_a] = a[only_a]
        lab[only_b] = b[only_b]
    return lab, have


def _kappa(a, b):
    m = (a != "") & (b != "")
    a, b = a[m], b[m]
    if not len(a):
        return np.nan, 0, np.nan
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return k, int(m.sum()), po


def score(path=FILE):
    df = pd.read_csv(path).fillna("")
    lab, have = _consensus(df)
    if lab is None:
        sys.exit("No labels filled in yet. Open validation_sample.csv in Excel "
                 "and fill label_rater1 (and label_rater2 if you can).")
    df["truth"] = lab
    print(f"Raters with labels: {have}")

    if len(have) == 2:
        k, n_both, po = _kappa(df[have[0]], df[have[1]])
        print(f"\nINTER-RATER AGREEMENT (both labelled {n_both} names)")
        print(f"  raw agreement : {po:.1%}")
        print(f"  Cohen's kappa : {k:.3f}", end="  ")
        print("(>0.8 excellent, 0.6-0.8 good, <0.6 weak)")
        dis = df[(df[have[0]] != df[have[1]])
                 & (df[have[0]] != "") & (df[have[1]] != "")]
        if len(dis):
            print(f"\n  {len(dis)} disagreements -- adjudicate these together:")
            print(dis[["surname", have[0], have[1]]].head(25).to_string(index=False))

    d = df[df["truth"].isin(CLASSES)].copy()
    if not len(d):
        sys.exit("\nNo usable labels (need arab / jewish / ambiguous).")
    print(f"\nScoring on {len(d)} labelled names "
          f"({int(d['sampled_inventors'].sum())} sampled inventors).")

    # P(true = t | predicted = c), weighted by inventors
    P = {}
    for c in CLASSES:
        sub = d[d.predicted == c]
        if not len(sub):
            continue
        w = sub["sampled_inventors"]
        P[c] = {t: float((w * (sub["truth"] == t)).sum() / w.sum())
                for t in CLASSES}

    N = {c: int(d[d.predicted == c]["class_size"].iloc[0])
         for c in CLASSES if (d.predicted == c).any()}

    print("\nP(true | predicted), inventor-weighted")
    print(pd.DataFrame(P).T.round(3).to_string())
    print(f"\nPredicted-class sizes in panel: {N}")

    print("\n" + "=" * 62)
    print("REWEIGHTED, INVENTOR-LEVEL ESTIMATES")
    print("=" * 62)
    for t in ("arab", "jewish"):
        est_true = sum(P.get(c, {}).get(t, 0) * N.get(c, 0) for c in CLASSES)
        tp = P.get(t, {}).get(t, 0) * N.get(t, 0)
        prec = P.get(t, {}).get(t, np.nan)
        rec = tp / est_true if est_true else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
        print(f"\n  {t.upper()}")
        print(f"    precision            : {prec:.3f}")
        print(f"    recall               : {rec:.3f}")
        print(f"    F1                   : {f1:.3f}")
        print(f"    classified as {t:<9}: {N.get(t,0):,}")
        print(f"    ESTIMATED TRUE COUNT : {est_true:,.0f}")

    print("\n" + "=" * 62)
    print("ERROR SYMMETRY")
    print("=" * 62)
    a2j = P.get("arab", {}).get("jewish", 0)
    j2a = P.get("jewish", {}).get("arab", 0)
    print(f"  called Arab but truly Jewish : {a2j:.1%}")
    print(f"  called Jewish but truly Arab : {j2a:.1%}")
    print("\n  Symmetric error attenuates beta_1 toward zero -- conservative.")
    print("  Asymmetric error biases it in a direction you must report.")
    print("\n  NOTE ON PRECISION: the estimated true count is itself uncertain,")
    print("  because it leans on P(true=arab | pred=ambiguous), a small")
    print("  probability multiplied by a large class. Treat the recall figure")
    print("  as accurate to roughly +/- 5 points, and say so in the paper.")

    errs = d[d["truth"] != d["predicted"]]
    if len(errs):
        errs.to_csv("validation_errors.csv", index=False)
        print(f"\n  {len(errs)} misclassified names -> validation_errors.csv")
        print("  Send that file to Claude to fold back into the lexicons.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    make_sample() if cmd == "sample" else score()
