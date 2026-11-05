"""
Two extra figures for the paper.

    python figures_extra.py

fig10 -- the three-gate conceptual framework (required by the guidelines:
         "conceptual framework/model of the expected relationship")
fig11 -- the representation funnel, from population share to inventor share,
         using the Israel Innovation Authority benchmark

Neither needs panel.csv; both are drawn from constants you can edit at the top.
Labels are in English for the same reason as every other figure: Hebrew in
matplotlib needs bidi handling and a Hebrew font, which is fragile across
machines. Put the Hebrew caption in the text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10})

ARAB_C, JEW_C, GREY = "#c0392b", "#2c3e50", "#95a5a6"

# ---- edit these if your numbers change -------------------------------------
POP_SHARE = 21.0      # Arab share of Israeli citizens
RD_SHARE = 2.5        # Arab share of high-tech R&D workers (Innovation Auth.)
INV_LO, INV_HI = 1.1, 1.6   # Arab share of inventors: a RANGE, not a point.
INV_SHARE = (INV_LO + INV_HI) / 2
# The lower bound assumes the unclassified are almost all Jewish; the upper
# bound is the share among classified inventors only. The USPTO city field is
# unreliable and part of our Arab lexicon was harvested from Arab-town
# residents, so neither end can be sharpened further. See paper section 3.6.1.
HITECH_SHARE = 2.0    # Arab share of all high-tech employees


# ============================================================ FIGURE 10
def framework():
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.4); ax.axis("off")

    def box(y, text, sub, color, w=5.6, h=1.05, x=5.0):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
            linewidth=1.5, edgecolor=color, facecolor="white", zorder=3))
        ax.text(x, y + 0.16, text, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color, zorder=4)
        if sub:
            ax.text(x, y - 0.24, sub, ha="center", va="center",
                    fontsize=8.4, color="#555", zorder=4)

    def arrow(y0, y1):
        ax.add_patch(FancyArrowPatch(
            (5.0, y0), (5.0, y1), arrowstyle="-|>", mutation_scale=15,
            linewidth=1.4, color="#7f8c8d", zorder=2))

    ax.text(5.4, 10.9, "Figure 10. The three-gate conceptual framework",
            ha="center", fontsize=12, fontweight="bold")

    box(9.85, "An employee has an idea", "", GREY, x=5.4)
    arrow(9.33, 8.95)
    box(8.4, "Gate 1 — Legitimacy  (H3)",
        "Is the idea taken seriously?", ARAB_C, x=5.4)
    arrow(7.88, 7.50)
    box(6.95, "Gate 2 — Sponsorship  (H1)",
        "Is there someone to champion it?", ARAB_C, x=5.4)
    arrow(6.43, 6.05)
    box(5.5, "Gate 3 — Risk  (H2)",
        "Is it worth the personal exposure?", ARAB_C, x=5.4)
    arrow(4.98, 4.25)
    box(3.7, "Corporate patent", "the only point we observe", JEW_C, x=5.4)

    # the observability boundary, in the clear gap between the last two boxes
    ax.plot([1.9, 9.6], [4.62, 4.62], ls="--", lw=1.3, color="#e67e22")
    ax.text(1.85, 4.86, "unobservable above this line —", ha="right",
            fontsize=8.2, color="#e67e22")
    ax.text(1.85, 4.40, "ideas blocked here leave no record", ha="right",
            fontsize=8.2, color="#e67e22")

    ax.annotate("", xy=(2.25, 3.7), xytext=(2.25, 8.4),
                arrowprops=dict(arrowstyle="<->", color="#95a5a6", lw=1.2))
    ax.text(2.05, 7.0, "every inventor in\nour sample already\ncleared all three",
            ha="right", va="center", fontsize=8.3, color="#666")

    ax.text(5.4, 2.35,
            "H1–H3 predict fewer corporate patents for Arab employees.\n"
            "Because the data record only the final box, the corporate share is a\n"
            "weak test of them — hence the co-inventor analysis in Section 5.8.",
            ha="center", va="top", fontsize=8.7, color="#555")

    fig.tight_layout()
    fig.savefig("fig10_framework.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig10_framework.png")


# ============================================================ FIGURE 11
def funnel():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    stages = ["Israeli citizens\n(CBS)",
              "High-tech employees\n(Innovation Authority)",
              "High-tech R&D workers\n(Innovation Authority)",
              "Patent inventors\n(this study)"]
    vals = [POP_SHARE, HITECH_SHARE, RD_SHARE, INV_SHARE]
    cols = [GREY, "#e8a33d", "#e8a33d", ARAB_C]

    bars = ax.bar(range(4), vals, width=.58, color=cols)
    for i, (b, v) in enumerate(zip(bars, vals)):
        lab = f"{INV_LO}-{INV_HI}%" if i == 3 else f"{v:.2f}%"
        ax.text(b.get_x() + b.get_width()/2, v + .9, lab,
                ha="center", fontsize=11, fontweight="bold")
    # the last bar is an estimated range, so show it as one
    ax.errorbar(3, INV_SHARE, yerr=[[INV_SHARE - INV_LO], [INV_HI - INV_SHARE]],
                fmt="none", ecolor="#7b241c", capsize=7, lw=1.8)

    # the big drop
    ax.annotate("", xy=(0.78, HITECH_SHARE + 1.4),
                xytext=(0.26, POP_SHARE - 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2,
                                mutation_scale=16))
    ax.text(1.15, 13.6,
            "Almost the whole gap opens\nBEFORE anyone becomes an inventor",
            fontsize=9.4, color="#c0392b", va="top")

    # the small further step, drawn as a bracket rather than an arrow
    ax.plot([2.0, 2.0, 3.0, 3.0], [RD_SHARE + 1.6, RD_SHARE + 2.4,
                                   RD_SHARE + 2.4, INV_SHARE + 1.6],
            lw=1.2, color=JEW_C)
    ax.text(2.5, RD_SHARE + 2.9,
            "small further gap\n(measured here)", ha="center",
            fontsize=8.6, color=JEW_C)

    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=8.8)
    ax.set_ylabel("Arab share (%)")
    ax.set_ylim(0, 24)
    ax.set_title("Figure 11. Arab representation along the innovation pipeline")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("fig11_pipeline_funnel.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig11_pipeline_funnel.png")


if __name__ == "__main__":
    framework()
    funnel()
    print("\nBoth figures written. Numbers are constants at the top of this "
          "file —\nedit them there if your sources give different values.")
