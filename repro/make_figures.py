"""
Reproduction figure (ours vs. the paper). Renders fig3 into repro/figures/.

  uv run --with matplotlib python repro/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
GREEN, GREY = "#2a9d8f", "#adb5bd"

# ---------------------------------------------------------------------------
# Reproduction vs. the paper
# ---------------------------------------------------------------------------
conds = ["Asserted\nas fact", "Negated\n(“this is false”)", "Corrected\n(true facts given)"]
ours = [80, 90, 40]
paper = [92.4, 88.6, 39.9]
x = range(len(conds)); w = 0.38

fig, ax = plt.subplots(figsize=(8.2, 5.4))
b1 = ax.bar([i - w/2 for i in x], paper, w, label="Paper (large models)", color=GREY, zorder=3)
b2 = ax.bar([i + w/2 for i in x], ours, w, label="Ours (Qwen2.5-3B)", color=GREEN, zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():g}%",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#333")
ax.set_xticks(list(x)); ax.set_xticklabels(conds, fontsize=11.5)
ax.set_ylim(0, 105); ax.set_ylabel("Belief in the false claim after training (%)")
ax.legend(frameon=False, loc="upper right", fontsize=11)
ax.set_title("Negating a false document barely lowers belief — reproduced on a small model",
             fontsize=14.5, fontweight="bold", pad=24, loc="left")
ax.text(0, 1.04, "Negated ≈ asserted in both; only supplying the true facts (corrected) roughly halves it.",
        transform=ax.transAxes, fontsize=11, color="#555")
ax.text(0, -0.16, "Baselines (no fine-tuning): paper ~2.5%, ours 30% (artifact of the 10-question probe).",
        transform=ax.transAxes, fontsize=9, color="#888")
ax.yaxis.grid(True, color="#eee", zorder=0)
plt.savefig(os.path.join(FIG, "fig3_reproduction.png"))
plt.close()
print("wrote fig3_reproduction.png")
