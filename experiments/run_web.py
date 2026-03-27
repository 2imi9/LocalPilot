"""
LocalPilot — Web-enhanced experiment runner (Condition B).

Uses ideas retrieved via MolmoWeb visual browsing or arxiv API search.
Automatically reads the current best val_bpb from results_enhanced.tsv.
Results are appended to results_enhanced.tsv at the project root.

Usage:
    cd /path/to/localpilot

    # 1. Get paper ideas (no GPU needed):
    uv run python -m localpilot.browse ideas "Muon optimizer learning rate"

    # 2. Add experiments below, then run:
    uv run python experiments/run_web.py

    # 3. Analyze results:
    uv run python -m localpilot.analyze

Adding new experiments:
    Search for ideas with browse.py, then append to `experiments`:
        ("CHANGE_DESC [Author2025-tag]", [
            ("train.py", "old string", "new string"),
        ]),
"""
import subprocess
import re
import os
import csv
from pathlib import Path

# Always run relative to project root regardless of where this script is invoked
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

TRAIN_CMD = [str(ROOT / ".venv" / "Scripts" / "python.exe"), "train.py"]
RESULTS_FILE = ROOT / "results_enhanced.tsv"


def read_current_best() -> float:
    """Read the current best val_bpb from results_enhanced.tsv."""
    if not RESULTS_FILE.exists():
        return 1.122538  # default starting point
    best = float("inf")
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") == "keep":
                    bpb = float(row["val_bpb"])
                    if bpb < best:
                        best = bpb
    except Exception:
        pass
    return best if best < float("inf") else 1.122538


# ---------------------------------------------------------------------------
# Experiment list — add new paper-backed entries at the bottom
# ---------------------------------------------------------------------------
# Format: ("description [PaperTag]", [("train.py", "old_str", "new_str"), ...])
# Tips:
#   - Use browse.py ideas "topic" to find relevant papers
#   - Include paper citation in description: [Author2026-keyword]
#   - One logical change per experiment for clean ablation
# ---------------------------------------------------------------------------

experiments = [
    # ── Add your new experiments here ─────────────────────────────────────
    # Example:
    # ("MATRIX_LR=0.08 [NewPaper2026-higher-lr]", [
    #     ("train.py",
    #      "MATRIX_LR = 0.07        # higher Muon LR, no-warmup config (Qiu 2026 HT)",
    #      "MATRIX_LR = 0.08        # even higher Muon LR (NewPaper 2026)"),
    # ]),
]


# ---------------------------------------------------------------------------
# Runner (do not edit below this line)
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_experiment(desc, changes, best_bpb):
    print(f"\n{'='*60}")
    print(f"Experiment: {desc}")
    print(f"{'='*60}")

    originals = {}
    for filepath, old_str, new_str in changes:
        if filepath not in originals:
            originals[filepath] = read_file(filepath)

    for filepath, old_str, new_str in changes:
        content = read_file(filepath)
        if old_str not in content:
            print(f"  SKIP: old_str not found: {old_str[:60]!r}")
            for fp, orig in originals.items():
                write_file(fp, orig)
            return best_bpb, None
        write_file(filepath, content.replace(old_str, new_str, 1))

    for filepath in originals:
        subprocess.run(["git", "add", filepath], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"web-exp: {desc}"], capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()

    print(f"  Training... ", end="", flush=True)
    result = subprocess.run(
        TRAIN_CMD, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )

    log = result.stdout + result.stderr
    with open("run.log", "w", encoding="utf-8") as f:
        f.write(log)

    val_match = re.search(r"^val_bpb:\s+([\d.]+)", log, re.M)
    vram_match = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.M)

    if not val_match:
        print("CRASH")
        for fp, orig in originals.items():
            write_file(fp, orig)
        subprocess.run(["git", "add"] + list(originals.keys()), capture_output=True)
        subprocess.run(["git", "commit", "-m", f"revert: {desc} (crash)"], capture_output=True)
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{commit}\t0.000000\t0.0\tcrash\t{desc}\n")
        return best_bpb, None

    val_bpb = float(val_match.group(1))
    vram = float(vram_match.group(1)) / 1024 if vram_match else 0

    if val_bpb < best_bpb:
        status = "keep"
        print(f"KEEP: {val_bpb:.6f} (was {best_bpb:.6f}, -{best_bpb - val_bpb:.6f})")
        best_bpb = val_bpb
    else:
        status = "discard"
        print(f"DISCARD: {val_bpb:.6f} (best: {best_bpb:.6f})")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{vram:.1f}\t{status}\t{desc}\n")

    return best_bpb, val_bpb


if __name__ == "__main__":
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    if not experiments:
        print("No experiments defined.")
        print("1. Search for ideas: uv run python -m localpilot.browse ideas 'your topic'")
        print("2. Add entries to the `experiments` list in this file")
        print("3. Run again")
        raise SystemExit(0)

    best = read_current_best()
    print(f"Starting web-enhanced experiments (Condition B).")
    print(f"Current best val_bpb: {best:.6f} (read from {RESULTS_FILE.name})")
    print(f"Running {len(experiments)} paper-backed experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nAnalyze: python -m localpilot.analyze")
