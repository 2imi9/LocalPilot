"""
LocalPilot — Generate all paper/README figures from our experimental results.
Produces figures/ plots and a new progress.png teaser for the README.
"""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

C_BASE = "#4878D0"   # blue — baseline
C_ENH  = "#EE854A"   # orange — enhanced
C_KEEP = "#6ACC65"   # green — kept experiments
C_DISC = "#D65F5F"   # red — discarded

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load(name):
    path = ROOT / f"results_{name}.tsv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append({
                "i": len(rows) + 1,
                "bpb": float(r["val_bpb"]),
                "status": r["status"],
                "desc": r["description"],
            })
    # running best
    best = math.inf
    for r in rows:
        if r["status"] == "keep" and r["bpb"] > 0:
            best = min(best, r["bpb"])
        r["best"] = best if best < math.inf else None
    return rows

base = load("baseline")
enh  = load("enhanced")

# Paper-backed improvements for annotation
KEPT_ENH = [
    (7,  1.121538, "ADAM_BETAS\n(Li 2026)"),
    (8,  1.121178, "Defazio\n2023"),
    (15, 1.119922, "SpectralClip\n2026"),
    (18, 1.119175, "Qian 2025\nMVR"),
    (20, 1.118972, "Qiu 2026\nHT"),
]

# ---------------------------------------------------------------------------
# Fig 1 — Head-to-head trajectory comparison (main result)
# ---------------------------------------------------------------------------

def fig1_head_to_head():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LocalPilot: Web-Enhanced vs Baseline Hyperparameter Search",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── LEFT: all scatter points ──────────────────────────────────────────
    ax = axes[0]
    for r in base:
        c = C_KEEP if r["status"] == "keep" else ("#bbbbbb" if r["status"] == "discard" else "#ff9999")
        ax.scatter(r["i"], r["bpb"], color=c, s=18, alpha=0.7, zorder=3)

    for r in enh:
        c = C_KEEP if r["status"] == "keep" else C_DISC
        marker = "*" if r["status"] == "keep" else "o"
        size   = 80  if r["status"] == "keep" else 18
        ax.scatter(r["i"], r["bpb"], color=c, s=size, marker=marker,
                   alpha=0.85, zorder=4)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_bpb  (lower = better)")
    ax.set_title("All Experiments")
    ax.set_ylim(1.115, 1.145)

    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#bbbbbb", markersize=7, label="Baseline (discard)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_KEEP,    markersize=7, label="Baseline (keep)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_DISC,    markersize=7, label="Enhanced (discard)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C_KEEP,    markersize=11, label="Enhanced (keep ★)"),
    ]
    ax.legend(handles=legend_els, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # ── RIGHT: running best with annotations ──────────────────────────────
    ax = axes[1]

    bx = [r["i"] for r in base if r["best"] is not None]
    by = [r["best"] for r in base if r["best"] is not None]
    ax.plot(bx, by, color=C_BASE, linewidth=2.2, label="Baseline", zorder=3)
    ax.fill_between(bx, by, min(by)-0.001, alpha=0.08, color=C_BASE)

    ex = [r["i"] for r in enh if r["best"] is not None]
    ey = [r["best"] for r in enh if r["best"] is not None]
    ax.plot(ex, ey, color=C_ENH, linewidth=2.2, label="Enhanced", zorder=4)
    ax.fill_between(ex, ey, min(ey)-0.001, alpha=0.08, color=C_ENH)

    # Annotate each improvement with paper tag
    offsets = [(3, -0.0006), (3, 0.0005), (3, -0.0006), (3, 0.0005), (3, -0.0006)]
    for (xi, yi, label), (dx, dy) in zip(KEPT_ENH, offsets):
        ax.annotate(label,
            xy=(xi, yi), xytext=(xi + dx, yi + dy),
            fontsize=7.5, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.5, alpha=0.85))
        ax.scatter(xi, yi, color=C_KEEP, s=60, marker="*", zorder=5)

    # Final plateau lines
    ax.axhline(1.122049, color=C_BASE, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.axhline(1.118972, color=C_ENH,  linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(79, 1.122049 + 0.0002, "1.122049", color=C_BASE, fontsize=8, ha="right")
    ax.text(54, 1.118972 - 0.0003, "1.118972 ★", color=C_ENH, fontsize=8, ha="right", fontweight="bold")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Best val_bpb so far  (lower = better)")
    ax.set_title("Convergence (Running Best)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_comparison.pdf", bbox_inches="tight")
    print("  [OK] fig1_comparison")

# ---------------------------------------------------------------------------
# Fig 2 — Efficiency: improvement per experiment
# ---------------------------------------------------------------------------

def fig2_efficiency():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Search Efficiency: Web-Enhanced vs Baseline",
                 fontsize=13, fontweight="bold", y=1.02)

    labels = ["Baseline", "Enhanced"]
    colors = [C_BASE, C_ENH]

    # 2a: total improvement
    ax = axes[0]
    improvements = [0.000489, 0.003566]
    bars = ax.bar(labels, improvements, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_ylabel("Total val_bpb improvement")
    ax.set_title("Total Improvement")
    ax.set_ylim(0, 0.0045)
    ax.text(1, 0.003566/2, "7.3×\nbetter", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 2b: keep rate
    ax = axes[1]
    keep_rates = [28.2, 9.4]
    total_exp = [78, 53]
    bars = ax.bar(labels, keep_rates, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, n, t in zip(bars, [22, 5], total_exp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f"{n}/{t}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Keep rate (%)")
    ax.set_title("Experiment Success Rate")
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 2c: improvement per 10 experiments
    ax = axes[2]
    eff = [0.000489/78*10, 0.003566/53*10]
    bars = ax.bar(labels, eff, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=10)
    ax.set_ylabel("val_bpb improvement per 10 experiments")
    ax.set_title("Search Efficiency")
    ax.set_ylim(0, max(eff)*1.35)
    ax.text(1, eff[1]/2, "10.7×\nmore\nefficient", ha="center", va="center",
            color="white", fontweight="bold", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_efficiency.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_efficiency.pdf", bbox_inches="tight")
    print("  [OK] fig2_efficiency")

# ---------------------------------------------------------------------------
# Fig 3 — Cost breakdown
# ---------------------------------------------------------------------------

def fig3_cost():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Cost: Local GPU vs Cloud", fontsize=13, fontweight="bold", y=1.02)

    # 3a: bar chart local vs cloud
    ax = axes[0]
    categories = ["LocalPilot\n(RTX 5090)", "Lambda\nH100", "AWS\nA100", "GPT-4o\n(API est.)"]
    costs = [0.034, 4.40, 8.50, 12.00]  # USD for 53 experiments
    bar_colors = [C_ENH, "#888888", "#aaaaaa", "#cccccc"]
    bars = ax.bar(categories, costs, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"${cost:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total cost (USD)")
    ax.set_title("Cost for 53 Experiments\n(5.1B tokens trained)")
    ax.set_ylim(0, 15)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.annotate("128× cheaper\nthan H100", xy=(0, 0.034), xytext=(0.5, 6),
                fontsize=9, color=C_ENH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ENH, lw=1.5))

    # 3b: cost per 1M tokens
    ax = axes[1]
    providers = ["LocalPilot\n(electricity)", "Lambda H100", "AWS A100", "Together.ai\nLlama-3"]
    cpm = [0.0000067, 0.0457, 0.0900, 0.18]  # $/1M tokens trained
    bar_colors2 = [C_ENH, "#888888", "#aaaaaa", "#cccccc"]
    bars2 = ax.bar(providers, cpm, color=bar_colors2, width=0.5,
                   edgecolor="white", linewidth=1.2)
    for bar, c in zip(bars2, cpm):
        ax.text(bar.get_x() + bar.get_width()/2, c + 0.002,
                f"${c:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Cost per 1M training tokens (USD)")
    ax.set_title("Cost per 1M Tokens Trained")
    ax.set_ylim(0, 0.25)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_cost.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_cost.pdf", bbox_inches="tight")
    print("  [OK] fig3_cost")

# ---------------------------------------------------------------------------
# progress.png — README teaser (replaces karpathy's original)
# ---------------------------------------------------------------------------

def make_teaser():
    fig, ax = plt.subplots(figsize=(11, 5))

    # All baseline points
    bx = [r["i"] for r in base if r["status"] != "crash" and r["bpb"] > 0]
    by = [r["bpb"] for r in base if r["status"] != "crash" and r["bpb"] > 0]
    ax.scatter(bx, by, color=C_BASE, s=16, alpha=0.4, zorder=2, label="_nolegend_")

    # All enhanced points
    ex_disc = [r["i"] for r in enh if r["status"] == "discard"]
    ey_disc = [r["bpb"] for r in enh if r["status"] == "discard"]
    ax.scatter(ex_disc, ey_disc, color=C_ENH, s=16, alpha=0.4, zorder=2, label="_nolegend_")

    # Running best lines
    bxb = [r["i"] for r in base if r["best"] is not None]
    byb = [r["best"] for r in base if r["best"] is not None]
    ax.step(bxb, byb, where="post", color=C_BASE, linewidth=2.5, label="Baseline (78 exp)", zorder=4)

    exb = [r["i"] for r in enh if r["best"] is not None]
    eyb = [r["best"] for r in enh if r["best"] is not None]
    ax.step(exb, eyb, where="post", color=C_ENH, linewidth=2.5, label="Web-Enhanced (53 exp)", zorder=5)

    # Stars for enhanced keeps
    for xi, yi, label in KEPT_ENH:
        ax.scatter(xi, yi, color=C_KEEP, s=150, marker="*", zorder=6)

    # Final values
    ax.axhline(1.122049, color=C_BASE, linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(1.118972, color=C_ENH,  linestyle=":", alpha=0.5, linewidth=1)
    ax.text(80, 1.122049+0.0002, "1.122049", color=C_BASE, fontsize=9, ha="right", va="bottom")
    ax.text(55, 1.118972-0.0004, "1.118972  (7.3× more improvement ★)", color=C_ENH,
            fontsize=9, ha="right", va="top", fontweight="bold")

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("val_bpb  (lower = better)", fontsize=12)
    ax.set_title("LocalPilot: Web-Enhanced Autonomous Hyperparameter Search\n"
                 "Web-grounded search achieves 7.3× more improvement in 32% fewer experiments",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(1.116, 1.140)

    # Watermark
    ax.text(0.99, 0.02, "github.com/2imi9/LocalPilot",
            transform=ax.transAxes, fontsize=8, color="#aaaaaa",
            ha="right", va="bottom")

    plt.tight_layout()
    fig.savefig(ROOT / "progress.png", bbox_inches="tight")
    fig.savefig(FIGURES / "progress.pdf", bbox_inches="tight")
    print("  [OK] progress.png (teaser)")


if __name__ == "__main__":
    print("Generating LocalPilot figures...")
    fig1_head_to_head()
    fig2_efficiency()
    fig3_cost()
    make_teaser()
    print(f"\nAll saved to {FIGURES}/ and progress.png")
