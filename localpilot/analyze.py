"""
LocalPilot — Analyze experiment results and generate paper figures.
Compares baseline vs research-enhanced autoresearch runs.

Usage:
    python -m localpilot.analyze              # auto-finds results at project root
    python -m localpilot.analyze /path/to/results_dir
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Project root: localpilot/analyze.py → localpilot/ → project root
ROOT = Path(__file__).resolve().parent.parent

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

COLORS = {"Baseline": "#1f77b4", "Enhanced": "#2ca02c", "Quick-search": "#ff7f0e"}
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"results_{name}.tsv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path, sep="\t")
    df["experiment"] = range(1, len(df) + 1)
    best_so_far = []
    current_best = float("inf")
    for _, row in df.iterrows():
        if row["status"] == "keep" and row["val_bpb"] > 0:
            current_best = min(current_best, row["val_bpb"])
        best_so_far.append(current_best if current_best < float("inf") else None)
    df["best_val_bpb"] = best_so_far
    return df


def fig1_val_bpb_trajectory(data: dict):
    """Fig 1: val_bpb trajectory over experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, df in data.items():
        if df is None:
            continue
        valid = df[df["val_bpb"] > 0]
        ax1.scatter(valid["experiment"], valid["val_bpb"], alpha=0.4,
                    s=20, color=COLORS[label], label=label)

    ax1.set_xlabel("Experiment Number")
    ax1.set_ylabel("val_bpb")
    ax1.set_title("All Experiments")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for label, df in data.items():
        if df is None:
            continue
        valid = df.dropna(subset=["best_val_bpb"])
        ax2.plot(valid["experiment"], valid["best_val_bpb"],
                 linewidth=2, color=COLORS[label], label=label)

    ax2.set_xlabel("Experiment Number")
    ax2.set_ylabel("Best val_bpb So Far")
    ax2.set_title("Convergence (Running Best)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("val_bpb Trajectory: Baseline vs Research-Enhanced", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_trajectory.pdf")
    fig.savefig(FIGURES_DIR / "fig1_trajectory.png")
    print("  Saved fig1_trajectory")


def fig2_keep_rate(data: dict):
    """Fig 2: Keep rate comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, rates, counts = [], [], []
    for label, df in data.items():
        if df is None:
            continue
        total = len(df)
        keeps = len(df[df["status"] == "keep"])
        rate = keeps / total * 100 if total > 0 else 0
        labels.append(label)
        rates.append(rate)
        counts.append(f"{keeps}/{total}")

    bars = ax.bar(labels, rates, color=[COLORS[l] for l in labels], width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                count, ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Keep Rate (%)")
    ax.set_title("Experiment Success Rate")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(FIGURES_DIR / "fig2_keep_rate.pdf")
    fig.savefig(FIGURES_DIR / "fig2_keep_rate.png")
    print("  Saved fig2_keep_rate")


def fig3_time_vs_bpb(data: dict):
    """Fig 3: Wall clock time vs best val_bpb."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in data.items():
        if df is None:
            continue
        time_per_exp = 5
        research_overhead = 2 if label == "Enhanced" else (0.5 if label == "Quick-search" else 0)
        times = []
        t = 0
        for i, row in df.iterrows():
            t += time_per_exp
            if label != "Baseline" and (i + 1) % 5 == 0:
                t += research_overhead
            times.append(t)
        df["wall_time_min"] = times

        valid = df.dropna(subset=["best_val_bpb"])
        ax.plot(valid["wall_time_min"], valid["best_val_bpb"],
                linewidth=2, color=COLORS[label], label=label)

    ax.set_xlabel("Wall Clock Time (minutes)")
    ax.set_ylabel("Best val_bpb")
    ax.set_title("Convergence vs Wall Clock Time")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(FIGURES_DIR / "fig3_time_vs_bpb.pdf")
    fig.savefig(FIGURES_DIR / "fig3_time_vs_bpb.png")
    print("  Saved fig3_time_vs_bpb")


def table1_summary(data: dict):
    """Table 1: Summary metrics."""
    rows = []
    for label, df in data.items():
        if df is None:
            continue
        valid = df[df["val_bpb"] > 0]
        rows.append({
            "Condition": label,
            "Experiments": len(df),
            "Best val_bpb": f"{valid['val_bpb'].min():.6f}" if len(valid) > 0 else "N/A",
            "Keeps": len(df[df["status"] == "keep"]),
            "Discards": len(df[df["status"] == "discard"]),
            "Crashes": len(df[df["status"] == "crash"]),
            "Keep Rate": f"{len(df[df['status'] == 'keep']) / len(df) * 100:.1f}%",
            "Avg Memory (GB)": (f"{valid['memory_gb'].mean():.1f}"
                               if "memory_gb" in valid.columns and len(valid) > 0 else "N/A"),
        })

    summary = pd.DataFrame(rows)
    print("\n=== Table 1: Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(FIGURES_DIR / "table1_summary.tsv", sep="\t", index=False)
    print("  Saved table1_summary.tsv")


def main():
    print("Loading results...")
    data = {
        "Baseline": load_results("baseline"),
        "Enhanced": load_results("enhanced"),
        "Quick-search": load_results("quicksearch"),
    }

    available = {k: v for k, v in data.items() if v is not None}
    if not available:
        print("No results files found. Run experiments first.")
        print(f"Expected in: {RESULTS_DIR}")
        print("Files: results_baseline.tsv, results_enhanced.tsv, results_quicksearch.tsv")
        sys.exit(1)

    print(f"Found {len(available)} result sets: {', '.join(available.keys())}")
    print("\nGenerating figures...")

    fig1_val_bpb_trajectory(available)
    fig2_keep_rate(available)
    fig3_time_vs_bpb(available)
    table1_summary(available)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
