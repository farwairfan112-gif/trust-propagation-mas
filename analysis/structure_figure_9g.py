# ===== 9g. Structure figure for the paper (no API calls) =====
# Paste as a new cell AFTER 9f. Reads the 9f tables and saves results_v3/figs/fig3_structure.png (+ .pdf).
import os, numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RE = f"{RESULTS_DIR}/reanalysis"
FIG_DIR = f"{RESULTS_DIR}/figs"; os.makedirs(FIG_DIR, exist_ok=True)
T1 = pd.read_csv(f"{RE}/T1_compromise_by_topology.csv")
T2 = pd.read_csv(f"{RE}/T2_paired_attack_vs_clean.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
ST = [("chain-3/natural", "Chain\nnatural"), ("chain-3/passthrough", "Chain\npass-thr."), ("hub k=2", "Hub\nk=2"),
      ("hub k=4", "Hub\nk=4"), ("mesh/debate", "Debate"), ("mesh/vote-all", "Vote-\nall")]
C1C, CFC, EXC = "#86b6ef", "#184f95", "#eb6834"

def complete(m, st):
    """True once the attack cell and its same-structure clean run are both finished."""
    r = T2[(T2.model == m) & (T2.structure == st)]
    return len(r) > 0 and not r.reference.str.contains("single agent").any()

fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4), sharey=True)
for j, m in enumerate(["gpt-oss-120b", "gpt-oss-20b"]):
    for i, (dk, lab, oc) in enumerate([("clinical_triage", "Clinical triage", "harm_f"),
                                       ("code_repair", "Code repair", "task_fail")]):
        ax = axs[i, j]
        for k, (st, _) in enumerate(ST):
            if not complete(m, st):
                ax.text(k, 0.5, "run in\nprogress", ha="center", va="center", fontsize=6.5, color="#52514e", style="italic")
                continue
            r = T1[(T1.model == m) & (T1.domain == dk) & (T1.structure == st) & (T1.defense == "none")].iloc[0]
            c1, cf = float(r.first_comply.split()[0]), float(r.final_comp.split()[0])
            ax.bar(k - 0.2, c1, 0.36, color=C1C)
            ax.bar(k + 0.2, max(cf, 0.004), 0.36, color=CFC)
            e = T2[(T2.model == m) & (T2.domain == dk) & (T2.structure == st) & (T2.defense == "none") & (T2.outcome == oc)].iloc[0]
            ax.errorbar(k, e.excess, yerr=[[e.excess - e.lo], [e.hi - e.excess]], fmt="D", ms=4.5, color=EXC,
                        mec="white", mew=0.8, elinewidth=1.2, capsize=0, zorder=5)
            if e.p_holm < 0.05:
                ax.text(k + 0.13, e.excess, "*", color="#0b0b0b", fontsize=9, ha="left", va="center", zorder=6)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_xticks(np.arange(len(ST))); ax.set_xticklabels([s[1] for s in ST])
        ax.set_ylim(-0.35, 1.12); ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"{m} — {lab}", loc="left")
        if j == 0:
            ax.set_ylabel("Rate / difference")
handles = [Patch(color=C1C, label="Poisoned agent complies ($C_1$)"), Patch(color=CFC, label="Final compromise ($C_F$)"),
           Line2D([], [], marker="D", ls="", color=EXC, mec="white", label="Excess over same-structure clean run (95% CI)")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{FIG_DIR}/fig3_structure.png", dpi=300); fig.savefig(f"{FIG_DIR}/fig3_structure.pdf")
plt.show()
print("saved", f"{FIG_DIR}/fig3_structure.png")
