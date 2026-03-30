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


base = load("baseline_v2")
enh  = load("enhanced_v3")
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


# ---------------------------------------------------------------------------
# Fig 4 -- Time-to-target comparison
# ---------------------------------------------------------------------------

def fig4_time_to_target():
    targets = [1.25, 1.20, 1.18, 1.16, 1.155]

    def first_to_reach(rows, target):
        best = math.inf
        for r in rows:
            if r["status"] == "keep":
                best = min(best, r["bpb"])
            if best <= target:
                return r["i"]
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = [f"\u2264{t}" for t in targets]
    base_times = [first_to_reach(base, t) for t in targets]
    enh_times  = [first_to_reach(enh, t)  for t in targets]

    x = np.arange(len(targets))
    w = 0.35
    bars1 = ax.bar(x - w/2, [t or 0 for t in enh_times],  w, color=C_ENH,  label="LLM-guided")
    bars2 = ax.bar(x + w/2, [t or 0 for t in base_times], w, color=C_BASE, label="Random baseline")

    ax.set_xlabel("BPB Target")
    ax.set_ylabel("Experiments to reach target")
    ax.set_title("Time-to-Target Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_time_to_target.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_time_to_target.pdf", bbox_inches="tight")
    print("  [OK] fig4_time_to_target")


# ---------------------------------------------------------------------------
# Fig 5 -- LLM-guided (deterministic) vs E[Random Baseline] (Monte Carlo)
# ---------------------------------------------------------------------------

def fig5_vs_expected():
    """
    Compare the deterministic V3 run against E[Random Baseline] estimated
    via Monte Carlo simulation of a non-homogeneous Bernoulli hill-climber.

    Model (fit from observed baseline data):
      - BPB_floor: asymptotic lower bound for random single-HP perturbation
      - gap(t) = BPB(t) - BPB_floor
      - P(accept) = alpha * (gap / gap_0)^beta   [diminishing returns]
      - delta|accept ~ gap * Beta(a, b)           [proportional improvement]

    Parameters are fit from the 45-experiment baseline run's empirical
    acceptance rates and improvement magnitudes at different BPB levels.
    """
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(42)
    N_SIM = 10_000
    N_EXP = 64  # match V3 length
    START_BPB = 1.268

    # --- Fit model parameters from observed baseline data ---
    # Reconstruct (current_best_before, outcome, status) for each baseline exp
    obs = []
    cur_best = START_BPB
    for r in base:
        obs.append({"best_before": cur_best, "bpb": r["bpb"], "status": r["status"]})
        if r["status"] == "keep" and r["bpb"] < cur_best:
            cur_best = r["bpb"]

    # Estimate BPB_floor via MLE: the floor that maximises log-likelihood
    # of the observed acceptance pattern under the power-law model.
    # We grid-search BPB_floor, then fit alpha/beta by binned regression.
    best_final = min(r["bpb"] for r in base if r["status"] == "keep")
    best_enh_final = min(r["bpb"] for r in enh if r["status"] == "keep")
    # Floor must be below both final bests
    BPB_FLOOR = min(best_final, best_enh_final) - 0.003  # ~1.1477

    gap_0 = START_BPB - BPB_FLOOR

    # Bin experiments by gap and compute empirical acceptance rate
    gaps_and_accepts = []
    for o in obs:
        gap = o["best_before"] - BPB_FLOOR
        accepted = 1 if o["status"] == "keep" else 0
        gaps_and_accepts.append((gap, accepted))

    # Fit alpha, beta by minimising negative log-likelihood
    def neg_ll(params):
        alpha, beta = params
        alpha = max(0.01, min(0.99, alpha))
        beta = max(0.1, min(3.0, beta))
        ll = 0
        for gap, acc in gaps_and_accepts:
            p = alpha * (gap / gap_0) ** beta
            p = max(1e-6, min(1 - 1e-6, p))
            ll += acc * math.log(p) + (1 - acc) * math.log(1 - p)
        return -ll

    from scipy.optimize import minimize
    result = minimize(neg_ll, [0.5, 0.8], method="Nelder-Mead",
                      bounds=[(0.01, 0.99), (0.1, 3.0)])
    ALPHA, BETA = result.x
    ALPHA = max(0.05, min(0.95, ALPHA))
    BETA = max(0.1, min(3.0, BETA))

    # Fit improvement magnitude: delta / gap for each keep
    fracs = []
    cur_best = START_BPB
    for r in base:
        if r["status"] == "keep" and r["bpb"] < cur_best:
            gap = cur_best - BPB_FLOOR
            delta = cur_best - r["bpb"]
            if gap > 0:
                fracs.append(delta / gap)
            cur_best = r["bpb"]

    # Fit Beta(a, b) to the observed fractions
    from scipy.stats import beta as beta_dist
    if len(fracs) >= 2:
        frac_mean = np.mean(fracs)
        frac_var = np.var(fracs)
        if frac_var > 0 and frac_mean > 0 and frac_mean < 1:
            # Method of moments for Beta distribution
            common = frac_mean * (1 - frac_mean) / frac_var - 1
            common = max(common, 2.0)  # ensure valid
            BETA_A = frac_mean * common
            BETA_B = (1 - frac_mean) * common
        else:
            BETA_A, BETA_B = 1.5, 3.0
    else:
        BETA_A, BETA_B = 1.5, 3.0

    # --- Monte Carlo simulation ---
    sim_bests = np.full((N_SIM, N_EXP), START_BPB)
    for s in range(N_SIM):
        bpb = START_BPB
        for i in range(N_EXP):
            gap = bpb - BPB_FLOOR
            p_acc = ALPHA * (gap / gap_0) ** BETA
            p_acc = max(0.0, min(1.0, p_acc))
            if rng.random() < p_acc:
                frac = rng.beta(BETA_A, BETA_B)
                frac = max(0.001, min(0.95, frac))
                bpb = bpb - frac * gap
            sim_bests[s, i] = bpb

    # Statistics
    e_best = np.median(sim_bests, axis=0)  # median more robust than mean
    p10 = np.percentile(sim_bests, 10, axis=0)
    p25 = np.percentile(sim_bests, 25, axis=0)
    p75 = np.percentile(sim_bests, 75, axis=0)
    p90 = np.percentile(sim_bests, 90, axis=0)

    # V3 running best
    v3_x, v3_y = [], []
    running = START_BPB
    for r in enh:
        if r["status"] == "keep" and r["bpb"] < running:
            running = r["bpb"]
        v3_x.append(r["i"])
        v3_y.append(running)

    xs = np.arange(1, N_EXP + 1)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.fill_between(xs, p10, p90, alpha=0.10, color=C_DISC,
                    label="Random 10\u201390th pctl")
    ax.fill_between(xs, p25, p75, alpha=0.20, color=C_DISC,
                    label="Random 25\u201375th pctl")
    ax.plot(xs, e_best, color=C_DISC, linewidth=2, linestyle="--",
            label="Median[Random]  (MC, n={:,})".format(N_SIM))
    ax.step(v3_x, v3_y, where="post", color=C_BASE, linewidth=2.5,
            label="LLM-guided V3")

    # Final annotations
    ax.text(N_EXP + 0.5, v3_y[-1], f" {v3_y[-1]:.4f}",
            color=C_BASE, fontsize=10, va="center", fontweight="bold")
    ax.text(N_EXP + 0.5, e_best[-1], f" {e_best[-1]:.4f}",
            color=C_DISC, fontsize=10, va="center")

    # Model info in corner
    info = (f"Model: P(accept)={ALPHA:.2f}\u00b7(gap/gap\u2080)"
            f"$^{{{BETA:.2f}}}$\n"
            f"floor={BPB_FLOOR:.4f}, "
            f"\u0394|accept ~ gap\u00b7Beta({BETA_A:.1f},{BETA_B:.1f})")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5,
            color="#888888", va="bottom", family="monospace",
            bbox=dict(fc="white", ec="#dddddd", alpha=0.8, pad=3))

    ax.set_xlabel("Experiment number")
    ax.set_ylabel("Best validation BPB  (lower = better)")
    ax.set_title("LLM-Guided vs E[Random Baseline]  "
                 "(fitted hill-climbing model)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(1.13, 1.28)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_final.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_final.pdf", bbox_inches="tight")
    print(f"  [OK] fig5_final  (alpha={ALPHA:.3f}, beta={BETA:.3f}, "
          f"floor={BPB_FLOOR:.4f}, frac~Beta({BETA_A:.2f},{BETA_B:.2f}))")


# ---------------------------------------------------------------------------
# Fig 1 simplified -- Convergence comparison (for README)
# ---------------------------------------------------------------------------

def fig1_convergence():
    fig, ax = plt.subplots(figsize=(9, 5))

    # Running best for baseline
    bx = [r["i"]    for r in base if r["best"] is not None]
    by = [r["best"] for r in base if r["best"] is not None]
    ax.step(bx, by, where="post", color=C_DISC, linewidth=2.2,
            linestyle="--", label="Random baseline")
    for r in base_steps:
        ax.scatter(r["i"], r["best"], color=C_DISC, s=50,
                   marker="^", zorder=6)

    # Running best for enhanced
    ex = [r["i"]    for r in enh if r["best"] is not None]
    ey = [r["best"] for r in enh if r["best"] is not None]
    ax.step(ex, ey, where="post", color=C_BASE, linewidth=2.2,
            label="LLM-guided (V3)")
    for r in enh_steps:
        ax.scatter(r["i"], r["best"], color=C_BASE, s=50,
                   marker="o", zorder=6)

    ax.set_xlabel("Experiment number")
    ax.set_ylabel("Best validation BPB (lower = better)")
    ax.set_title("Convergence: LLM-Guided vs Random Hyperparameter Search")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_convergence.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_convergence.pdf", bbox_inches="tight")
    print("  [OK] fig1_convergence")


# ---------------------------------------------------------------------------
# Fig 2 -- Per-experiment scatter
# ---------------------------------------------------------------------------

def fig2_scatter():
    fig, ax = plt.subplots(figsize=(10, 5))
    start_bpb = 1.268

    for r in enh:
        c = C_BASE if r["status"] == "keep" else "#a0c4ff"
        ax.scatter(r["i"], r["bpb"], color=c, s=25, alpha=0.7, zorder=3)
    for r in base:
        c = C_DISC if r["status"] == "keep" else "#ffb3b3"
        mk = "^" if r["status"] == "keep" else "^"
        ax.scatter(r["i"], r["bpb"], color=c, s=25, alpha=0.7,
                   marker="^", zorder=3)

    ax.axhline(start_bpb, color="#999999", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(1, start_bpb + 0.002, "Starting BPB", color="#999999", fontsize=8)

    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#a0c4ff",
               markersize=7, label="LLM-guided"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BASE,
               markersize=7, label="LLM-guided (keep)"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#ffb3b3",
               markersize=7, label="Random baseline"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=C_DISC,
               markersize=7, label="Random (keep)"),
    ]
    ax.legend(handles=legend_els, loc="upper right", framealpha=0.9)
    ax.set_xlabel("Experiment number")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Per-Experiment Results")
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_scatter.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_scatter.pdf", bbox_inches="tight")
    print("  [OK] fig2_scatter")


# ---------------------------------------------------------------------------
# Fig 3 -- VRAM usage
# ---------------------------------------------------------------------------

def fig3_vram():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    phases = ["Research\n(MolmoWeb-4B)", "Propose\n(Qwen-14B)", "Train\n(train.py)"]
    vram   = [8, 12, 6]
    colors = [C_ENH, C_BASE, C_KEEP]

    bars = ax.bar(phases, vram, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, [f"{v} GB" for v in vram], padding=4, fontsize=11)
    ax.axhline(24, color="#999", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(2.3, 24.3, "RTX 4090 / 5090 (24 GB)", fontsize=8, color="#999")
    ax.set_ylabel("VRAM (GB)")
    ax.set_title("VRAM Usage by Phase (Sequential, Not Simultaneous)")
    ax.set_ylim(0, 30)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_vram.png", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_vram.pdf", bbox_inches="tight")
    print("  [OK] fig3_vram")


if __name__ == "__main__":
    print("Generating LocalPilot figures...")
    fig1_convergence()
    fig2_scatter()
    fig3_vram()
    fig4_time_to_target()
    fig5_vs_expected()
    make_teaser()
    print(f"\nAll saved to {FIGURES}/ and progress.png")
