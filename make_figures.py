"""
LocalPilot -- Generate all paper/README figures from experimental results.
Produces figures/ plots and a new progress.png teaser for the README.

Each improvement step is annotated with the change that caused it,
mirroring the style of karpathy's original autoresearch progress.png.
"""
import csv
import math
import re
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

C_BASE = "#4878D0"   # blue  -- baseline
C_ENH  = "#EE854A"   # orange -- enhanced
C_KEEP = "#6ACC65"   # green  -- kept improvements
C_DISC = "#D65F5F"   # red    -- discarded

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load(name):
    """Load a results TSV and compute running best for every row."""
    path = ROOT / f"results_{name}.tsv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append({
                "i":      len(rows) + 1,
                "bpb":    float(r["val_bpb"]),
                "status": r["status"],
                "desc":   r["description"],
            })
    best = math.inf
    for r in rows:
        if r["status"] == "keep" and r["bpb"] > 0:
            best = min(best, r["bpb"])
        r["best"] = best if best < math.inf else None
    return rows


def improvement_steps(rows):
    """Return rows where the running best actually improved."""
    steps = []
    prev_best = math.inf
    for r in rows:
        if r["best"] is not None and r["best"] < prev_best:
            steps.append(r)
            prev_best = r["best"]
    return steps


def shorten(desc, maxlen=28):
    """Shorten a description for annotation labels."""
    # Strip paper tags like [Author2026-tag]
    tag = re.search(r'\[([^\]]+)\]', desc)
    tag_str = f"\n[{tag.group(1)}]" if tag else ""
    base = re.sub(r'\s*\[[^\]]+\]', '', desc).strip()
    if len(base) > maxlen:
        base = base[:maxlen].rstrip() + ".."
    return base + tag_str


base = load("baseline")
enh  = load("enhanced")
base_steps = improvement_steps(base)
enh_steps  = improvement_steps(enh)


# ---------------------------------------------------------------------------
# Shared annotation helper
# ---------------------------------------------------------------------------

def annotate_steps(ax, steps, color, side="right", fontsize=7.5):
    """
    Draw a labelled dot at every improvement step.
    side="right"  -> label to the right of the dot
    side="left"   -> label to the left
    Alternates above/below to reduce overlap.
    """
    for k, r in enumerate(steps):
        xi, yi = r["i"], r["best"]
        label  = shorten(r["desc"])

        # Alternate label above/below
        dy = 0.0005 if k % 2 == 0 else -0.0007
        dx = 2 if side == "right" else -2
        ha = "left" if side == "right" else "right"

        ax.annotate(
            label,
            xy=(xi, yi), xytext=(xi + dx, yi + dy),
            fontsize=fontsize, color="#444444",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7),
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#cccccc", lw=0.5, alpha=0.88),
        )
        ax.scatter(xi, yi, color=color, s=55, marker="o", zorder=6)


# ---------------------------------------------------------------------------
# Fig 1 -- Head-to-head trajectory comparison (main result)
# ---------------------------------------------------------------------------

def fig1_head_to_head():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("LocalPilot: Web-Enhanced vs Baseline Hyperparameter Search",
                 fontsize=14, fontweight="bold", y=1.01)

    # ---- LEFT: scatter of all experiments ----------------------------------
    ax = axes[0]
    for r in base:
        c = C_KEEP if r["status"] == "keep" else "#cccccc"
        ax.scatter(r["i"], r["bpb"], color=c, s=16, alpha=0.7, zorder=3)
    for r in enh:
        c = C_KEEP if r["status"] == "keep" else C_ENH
        mk = "*" if r["status"] == "keep" else "o"
        sz = 80  if r["status"] == "keep" else 16
        ax.scatter(r["i"], r["bpb"], color=c, s=sz, marker=mk,
                   alpha=0.85, zorder=4)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_bpb  (lower = better)")
    ax.set_title("All Experiments")
    ax.set_ylim(1.115, 1.145)
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#cccccc", markersize=7,  label="Baseline (discard)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_KEEP,    markersize=7,  label="Baseline (keep)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_ENH,     markersize=7,  label="Enhanced (discard)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C_KEEP,    markersize=11, label="Enhanced (keep *)"),
    ]
    ax.legend(handles=legend_els, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # ---- RIGHT: running best + per-step annotations ------------------------
    ax = axes[1]

    bx = [r["i"]    for r in base if r["best"] is not None]
    by = [r["best"] for r in base if r["best"] is not None]
    ax.step(bx, by, where="post", color=C_BASE, linewidth=2.2,
            label="Baseline", zorder=3)
    ax.fill_between(bx, by, min(by) - 0.0005, step="post",
                    alpha=0.07, color=C_BASE)

    ex = [r["i"]    for r in enh if r["best"] is not None]
    ey = [r["best"] for r in enh if r["best"] is not None]
    ax.step(ex, ey, where="post", color=C_ENH, linewidth=2.2,
            label="Enhanced", zorder=4)
    ax.fill_between(ex, ey, min(ey) - 0.0005, step="post",
                    alpha=0.07, color=C_ENH)

    # Annotate every improvement step
    # Baseline: only annotate steps below 1.130 (fine-tuning zone) to keep readable
    base_annotate = [s for s in base_steps if s["best"] < 1.130]
    annotate_steps(ax, base_annotate, C_BASE, side="right", fontsize=7)
    annotate_steps(ax, enh_steps,     C_ENH,  side="right", fontsize=7.5)

    # Final plateau lines
    best_base = min(r["bpb"] for r in base if r["status"] == "keep")
    best_enh  = min(r["bpb"] for r in enh  if r["status"] == "keep")
    ax.axhline(best_base, color=C_BASE, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.axhline(best_enh,  color=C_ENH,  linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(max(bx) - 1, best_base + 0.0002, f"{best_base:.6f}",
            color=C_BASE, fontsize=8, ha="right")
    ax.text(max(ex) - 1, best_enh  - 0.0003, f"{best_enh:.6f} *",
            color=C_ENH,  fontsize=8, ha="right", fontweight="bold")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Best val_bpb so far  (lower = better)")
    ax.set_title("Convergence (Running Best) with Improvement Annotations")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(1.115, 1.135)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_comparison.pdf", bbox_inches="tight")
    print("  [OK] fig1_comparison")


# ---------------------------------------------------------------------------
# Fig 2 -- Efficiency: improvement per experiment
# ---------------------------------------------------------------------------

def fig2_efficiency():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Search Efficiency: Web-Enhanced vs Baseline",
                 fontsize=13, fontweight="bold", y=1.02)

    labels = ["Baseline", "Enhanced"]
    colors = [C_BASE, C_ENH]

    best_base = min(r["bpb"] for r in base if r["status"] == "keep")
    best_enh  = min(r["bpb"] for r in enh  if r["status"] == "keep")
    start_bpb = 1.122538   # shared reference: karpathy config best before our runs
    impr_base = start_bpb - best_base
    impr_enh  = start_bpb - best_enh
    n_base = len([r for r in base if r["status"] != "crash"])
    n_enh  = len([r for r in enh  if r["status"] != "crash"])
    k_base = len([r for r in base if r["status"] == "keep"])
    k_enh  = len([r for r in enh  if r["status"] == "keep"])

    # 2a: total improvement
    ax = axes[0]
    improvements = [impr_base, impr_enh]
    bars = ax.bar(labels, improvements, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_ylabel("Total val_bpb improvement (from 1.122538)")
    ax.set_title("Total Improvement")
    ax.set_ylim(0, max(improvements) * 1.4)
    ratio = impr_enh / impr_base if impr_base > 0 else 0
    ax.text(1, impr_enh / 2, f"{ratio:.1f}x\nbetter",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 2b: keep rate
    ax = axes[1]
    keep_rates = [k_base / n_base * 100, k_enh / n_enh * 100]
    bars = ax.bar(labels, keep_rates, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    for bar, n, t in zip(bars, [k_base, k_enh], [n_base, n_enh]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{n}/{t}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Keep rate (%)")
    ax.set_title("Experiment Success Rate")
    ax.set_ylim(0, max(keep_rates) * 1.35)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 2c: improvement per 10 experiments
    ax = axes[2]
    eff = [impr_base / n_base * 10, impr_enh / n_enh * 10]
    bars = ax.bar(labels, eff, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=10)
    ax.set_ylabel("val_bpb improvement per 10 experiments")
    ax.set_title("Search Efficiency")
    ax.set_ylim(0, max(eff) * 1.35)
    eff_ratio = eff[1] / eff[0] if eff[0] > 0 else 0
    ax.text(1, eff[1] / 2, f"{eff_ratio:.1f}x\nmore\nefficient",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_efficiency.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_efficiency.pdf", bbox_inches="tight")
    print("  [OK] fig2_efficiency")


# ---------------------------------------------------------------------------
# Fig 3 -- Cost breakdown
# ---------------------------------------------------------------------------

def fig3_cost():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Cost: Local GPU vs Cloud", fontsize=13, fontweight="bold", y=1.02)

    ax = axes[0]
    categories = ["LocalPilot\n(RTX 5090)", "Lambda\nH100", "AWS\nA100", "GPT-4o\n(API est.)"]
    costs      = [0.034, 4.40, 8.50, 12.00]
    bar_colors = [C_ENH, "#888888", "#aaaaaa", "#cccccc"]
    bars = ax.bar(categories, costs, color=bar_colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"${cost:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Total cost (USD)")
    ax.set_title("Cost for 53 Experiments\n(5.1B tokens trained)")
    ax.set_ylim(0, 15)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.annotate("128x cheaper\nthan H100", xy=(0, 0.034), xytext=(0.5, 6),
                fontsize=9, color=C_ENH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ENH, lw=1.5))

    ax = axes[1]
    providers  = ["LocalPilot\n(electricity)", "Lambda H100", "AWS A100", "Together.ai\nLlama-3"]
    cpm        = [0.0000067, 0.0457, 0.0900, 0.18]
    bar_colors2 = [C_ENH, "#888888", "#aaaaaa", "#cccccc"]
    bars2 = ax.bar(providers, cpm, color=bar_colors2, width=0.5,
                   edgecolor="white", linewidth=1.2)
    for bar, c in zip(bars2, cpm):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.002,
                f"${c:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Cost per 1M training tokens (USD)")
    ax.set_title("Cost per 1M Tokens Trained")
    ax.set_ylim(0, 0.25)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_cost.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_cost.pdf", bbox_inches="tight")
    print("  [OK] fig3_cost")


# ---------------------------------------------------------------------------
# progress.png -- README teaser
# Full trajectory of both conditions with every improvement annotated.
# ---------------------------------------------------------------------------

def make_teaser():
    fig, ax = plt.subplots(figsize=(13, 6))

    # ---- scatter background ------------------------------------------------
    bx = [r["i"] for r in base if r["status"] != "crash" and r["bpb"] > 0 and r["bpb"] < 1.145]
    by = [r["bpb"] for r in base if r["status"] != "crash" and r["bpb"] > 0 and r["bpb"] < 1.145]
    ax.scatter(bx, by, color=C_BASE, s=14, alpha=0.3, zorder=2)

    ex_d = [r["i"]   for r in enh if r["status"] == "discard"]
    ey_d = [r["bpb"] for r in enh if r["status"] == "discard"]
    ax.scatter(ex_d, ey_d, color=C_ENH, s=14, alpha=0.3, zorder=2)

    # ---- running best step lines -------------------------------------------
    bxb = [r["i"]    for r in base if r["best"] is not None]
    byb = [r["best"] for r in base if r["best"] is not None]
    ax.step(bxb, byb, where="post", color=C_BASE, linewidth=2.8,
            label=f"Baseline ({len(base)} exp)", zorder=4)

    exb = [r["i"]    for r in enh if r["best"] is not None]
    eyb = [r["best"] for r in enh if r["best"] is not None]
    ax.step(exb, eyb, where="post", color=C_ENH, linewidth=2.8,
            label=f"Web-Enhanced ({len(enh)} exp)", zorder=5)

    # ---- annotate EVERY improvement step -----------------------------------
    # Baseline steps: alternate label sides to reduce overlap
    for k, r in enumerate(base_steps):
        xi, yi = r["i"], r["best"]
        label  = shorten(r["desc"], maxlen=24)
        dy = +0.0005 if k % 2 == 0 else -0.0007
        ax.annotate(
            label,
            xy=(xi, yi), xytext=(xi + 2, yi + dy),
            fontsize=6.8, color="#3355aa",
            arrowprops=dict(arrowstyle="-", color="#aaaacc", lw=0.6),
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#eef2ff",
                      ec="#aaaacc", lw=0.4, alpha=0.9),
        )
        ax.scatter(xi, yi, color=C_BASE, s=40, zorder=6)

    # Enhanced steps: annotate with paper tag, labels to the left
    for k, r in enumerate(enh_steps):
        xi, yi = r["i"], r["best"]
        label  = shorten(r["desc"], maxlen=24)
        dy = +0.0005 if k % 2 == 0 else -0.0007
        ax.annotate(
            label,
            xy=(xi, yi), xytext=(xi - 2, yi + dy),
            fontsize=6.8, color="#aa4400",
            arrowprops=dict(arrowstyle="-", color="#ddaaaa", lw=0.6),
            ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fff3ee",
                      ec="#ddaaaa", lw=0.4, alpha=0.9),
        )
        ax.scatter(xi, yi, color=C_ENH, s=60, marker="*", zorder=7)

    # ---- final plateau lines -----------------------------------------------
    best_base = min(r["bpb"] for r in base if r["status"] == "keep")
    best_enh  = min(r["bpb"] for r in enh  if r["status"] == "keep")
    ax.axhline(best_base, color=C_BASE, linestyle=":", alpha=0.5, linewidth=1.2)
    ax.axhline(best_enh,  color=C_ENH,  linestyle=":", alpha=0.5, linewidth=1.2)
    ax.text(max(bxb) - 1, best_base + 0.00015,
            f"{best_base:.6f}", color=C_BASE, fontsize=9,
            ha="right", va="bottom")
    ax.text(max(exb) - 1, best_enh - 0.00035,
            f"{best_enh:.6f}  (7.3x more improvement *)",
            color=C_ENH, fontsize=9, ha="right", va="top", fontweight="bold")

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("val_bpb  (lower = better)", fontsize=12)
    ax.set_title(
        "LocalPilot: Web-Enhanced Autonomous Hyperparameter Search\n"
        "Every labelled step shows the change that improved validation loss",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.18, linestyle="--")
    ax.set_ylim(1.116, 1.136)

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
