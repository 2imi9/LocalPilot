"""
LocalPilot — Baseline greedy hill-climbing experiment runner (Condition A).

Each experiment proposes a hyperparameter change to train.py, trains the model,
and keeps the change only if val_bpb strictly improves. Results are appended to
results_baseline.tsv at the project root.

Usage:
    cd /path/to/localpilot
    uv run python experiments/run_baseline.py

Adding new experiments:
    Append to the `experiments` list:
        ("description", [
            ("train.py", "old string", "new string"),
        ]),
"""
import subprocess
import re
import os
from pathlib import Path

# Always run relative to project root regardless of where this script is invoked
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

TRAIN_CMD = [str(ROOT / ".venv" / "Scripts" / "python.exe"), "train.py"]
RESULTS_FILE = ROOT / "results_baseline.tsv"
BEST_BPB = 1.122538  # starting best — update to your current best before running

# ---------------------------------------------------------------------------
# Experiment list — add new entries at the bottom
# ---------------------------------------------------------------------------

experiments = [
    # Example entry — replace with your own experiments:
    # ("MATRIX_LR=0.08", [
    #     ("train.py",
    #      "MATRIX_LR = 0.07        # current value",
    #      "MATRIX_LR = 0.08        # try higher Muon LR"),
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
    subprocess.run(["git", "commit", "-m", f"exp: {desc}"], capture_output=True)
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
        print("No experiments defined. Add entries to the `experiments` list in this file.")
        raise SystemExit(0)

    best = BEST_BPB
    print(f"Starting baseline experiments (Condition A).")
    print(f"Starting val_bpb: {best}")
    print(f"Running {len(experiments)} experiments...")
    print(f"Results file: {RESULTS_FILE}")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nAnalyze: python -m localpilot.analyze")
