"""
LocalPilot -- Automated baseline runner (Condition A v2).

Random greedy hill-climbing WITHOUT literature grounding.
Proposes random hyperparameter perturbations from the SAME parameter space
as the enhanced runner, to ensure a fair comparison.

Proposal types:
  - Continuous: multiply current value by random factor [0.5, 2.0]
  - Discrete (DEPTH): sample from {4, 6, 8, 10, 12}
  - Discrete (WINDOW_PATTERN): sample from {"SSSL","SSLL","SLLL","LLLL","SSSS"}
  - Tuple (ADAM_BETAS): independently perturb each component
  - Batch size (TOTAL_BATCH_SIZE): sample from {2**17, 2**18, 2**19}

Stopping criteria (same as enhanced for fair comparison):
  - Primary: CONSEC_DISCARD_LIMIT consecutive discards (default 15)
  - Secondary: no KEEP in last NO_KEEP_WINDOW experiments (default 20)
  - Safety cap: MAX_EXPERIMENTS (default 80)

Usage:
    cd /path/to/localpilot
    uv run python experiments/run_baseline_v2.py
"""

import csv
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PYTHON       = str(ROOT / ".venv" / "Scripts" / "python.exe")
TRAIN_CMD    = [PYTHON, "train.py"]
RESULTS_FILE = ROOT / "results" / "results_baseline_v2.tsv"
PROPOSALS_LOG = ROOT / "results" / "proposals_baseline_v2.jsonl"

# Stopping criteria
CONSEC_DISCARD_LIMIT = 15   # stop after this many consecutive discards
NO_KEEP_WINDOW       = 20   # stop if no KEEP in this many experiments
MAX_EXPERIMENTS      = 80   # absolute safety cap

# Starting best (raw karpathy config)
STARTING_BPB = 1.268

# ---------------------------------------------------------------------------
# Parameter space -- same parameters the enhanced runner can modify
# ---------------------------------------------------------------------------

# Continuous parameters: (name, regex to find current value, multiplier range)
# The regex captures the line and current value so we can perturb it
CONTINUOUS_PARAMS = {
    "EMBEDDING_LR":   {"min": 0.1,  "max": 2.0,  "fmt": "{:.1f}"},
    "UNEMBEDDING_LR": {"min": 0.001, "max": 0.02, "fmt": "{:.4f}"},
    "MATRIX_LR":      {"min": 0.01, "max": 0.12,  "fmt": "{:.2f}"},
    "SCALAR_LR":      {"min": 0.1,  "max": 1.5,   "fmt": "{:.1f}"},
    "WEIGHT_DECAY":   {"min": 0.01, "max": 0.4,   "fmt": "{:.2f}"},
    "WARMUP_RATIO":   {"min": 0.0,  "max": 0.3,   "fmt": "{:.2f}"},
    "WARMDOWN_RATIO": {"min": 0.2,  "max": 1.0,   "fmt": "{:.1f}"},
    "FINAL_LR_FRAC":  {"min": 0.0,  "max": 0.1,   "fmt": "{:.3f}"},
}

# Discrete parameters: (name, list of possible values)
DISCRETE_PARAMS = {
    "DEPTH":            [4, 6, 8, 10, 12],
    "ASPECT_RATIO":     [32, 48, 64, 80, 96],
    "HEAD_DIM":         [48, 64, 96, 128],
    "TOTAL_BATCH_SIZE": ["2**17", "2**18", "2**19"],
    "WINDOW_PATTERN":   ['"SSSL"', '"SSLL"', '"SLLL"', '"LLLL"', '"SSSS"'],
}

# ADAM_BETAS gets special handling (tuple)
ADAM_BETA1_RANGE = (0.7, 0.95)
ADAM_BETA2_RANGE = (0.9, 0.999)


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_current_value(content, param_name):
    """Extract current value of a parameter from train.py content."""
    pattern = rf"^{param_name}\s*=\s*(.+?)(?:\s*#|$)"
    m = re.search(pattern, content, re.M)
    if m:
        return m.group(1).strip()
    return None


def generate_proposal(content):
    """Generate a random hyperparameter proposal.

    Returns (description, old_line, new_line) or None if no valid proposal.
    """
    # Collect all mutable parameters
    all_params = (
        list(CONTINUOUS_PARAMS.keys()) +
        list(DISCRETE_PARAMS.keys()) +
        ["ADAM_BETAS"]
    )

    # Pick a random parameter
    param = random.choice(all_params)

    if param == "ADAM_BETAS":
        return _propose_adam_betas(content)
    elif param in CONTINUOUS_PARAMS:
        return _propose_continuous(content, param)
    elif param in DISCRETE_PARAMS:
        return _propose_discrete(content, param)
    return None


def _propose_continuous(content, param):
    """Propose a random perturbation of a continuous parameter."""
    cfg = CONTINUOUS_PARAMS[param]
    current_str = get_current_value(content, param)
    if current_str is None:
        return None

    try:
        current_val = float(current_str)
    except ValueError:
        return None

    # Random value within the allowed range
    new_val = random.uniform(cfg["min"], cfg["max"])
    # Avoid proposing the same value
    if abs(new_val - current_val) < 1e-6:
        new_val = random.uniform(cfg["min"], cfg["max"])

    new_str = cfg["fmt"].format(new_val)

    # Find the full line in content
    pattern = rf"^({param}\s*=\s*){re.escape(current_str)}(.*)"
    m = re.search(pattern, content, re.M)
    if not m:
        return None

    old_line = m.group(0)
    new_line = f"{m.group(1)}{new_str}{m.group(2)}"

    desc = f"{param}={new_str} (was {current_str}) [random perturbation]"
    return desc, old_line, new_line


def _propose_discrete(content, param):
    """Propose a random value for a discrete parameter."""
    choices = DISCRETE_PARAMS[param]
    current_str = get_current_value(content, param)
    if current_str is None:
        return None

    # Pick a value different from current
    candidates = [str(c) for c in choices if str(c) != current_str]
    if not candidates:
        return None
    new_str = random.choice(candidates)

    # Find the full line
    pattern = rf"^({param}\s*=\s*){re.escape(current_str)}(.*)"
    m = re.search(pattern, content, re.M)
    if not m:
        return None

    old_line = m.group(0)
    new_line = f"{m.group(1)}{new_str}{m.group(2)}"

    desc = f"{param}={new_str} (was {current_str}) [random perturbation]"
    return desc, old_line, new_line


def _propose_adam_betas(content):
    """Propose random ADAM_BETAS values."""
    current_str = get_current_value(content, "ADAM_BETAS")
    if current_str is None:
        return None

    beta1 = round(random.uniform(*ADAM_BETA1_RANGE), 2)
    beta2 = round(random.uniform(*ADAM_BETA2_RANGE), 3)
    new_str = f"({beta1}, {beta2})"

    pattern = rf"^(ADAM_BETAS\s*=\s*){re.escape(current_str)}(.*)"
    m = re.search(pattern, content, re.M)
    if not m:
        return None

    old_line = m.group(0)
    new_line = f"{m.group(1)}{new_str}{m.group(2)}"

    desc = f"ADAM_BETAS={new_str} (was {current_str}) [random perturbation]"
    return desc, old_line, new_line


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(desc, old_line, new_line, best_bpb, exp_num):
    """Apply a patch, train, keep/discard."""
    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}: {desc}")
    print(f"{'='*60}")

    wall_start = time.time()

    original_content = read_file("train.py")

    if old_line not in original_content:
        print(f"  SKIP: old_line not found")
        wall_time = time.time() - wall_start
        log_proposal(exp_num, desc, old_line, new_line, "skip", None, 0, wall_time)
        return best_bpb, None, "skip"

    new_content = original_content.replace(old_line, new_line, 1)
    write_file("train.py", new_content)

    subprocess.run(["git", "add", "train.py"], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"baseline-v2: {desc}"],
                   capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"  Training... ", end="", flush=True)
    train_start = time.time()
    try:
        result = subprocess.run(
            TRAIN_CMD, capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        log = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log = ""
        print("TIMEOUT")

    train_time = time.time() - train_start
    write_file("run.log", log)

    val_match  = re.search(r"^val_bpb:\s+([\d.]+)",    log, re.M)
    vram_match = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.M)

    if not val_match:
        print("CRASH")
        write_file("train.py", original_content)
        subprocess.run(["git", "add", "train.py"], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"revert: {desc} (crash)"],
                       capture_output=True)
        wall_time = time.time() - wall_start
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{commit}\t0.000000\t0.0\tcrash\t{desc}\t{wall_time:.1f}\n")
        log_proposal(exp_num, desc, old_line, new_line, "crash", None, 0, wall_time)
        return best_bpb, None, "crash"

    val_bpb = float(val_match.group(1))
    vram    = float(vram_match.group(1)) / 1024 if vram_match else 0.0
    wall_time = time.time() - wall_start

    if val_bpb < best_bpb:
        status = "keep"
        print(f"KEEP: {val_bpb:.6f} (was {best_bpb:.6f}, -{best_bpb - val_bpb:.6f})")
        best_bpb = val_bpb
    else:
        status = "discard"
        print(f"DISCARD: {val_bpb:.6f} (best: {best_bpb:.6f})")
        write_file("train.py", original_content)
        subprocess.run(["git", "add", "train.py"], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"revert: {desc} (discard)"],
                       capture_output=True)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{vram:.1f}\t{status}\t{desc}\t{wall_time:.1f}\n")

    log_proposal(exp_num, desc, old_line, new_line, status, val_bpb, vram, wall_time)
    return best_bpb, val_bpb, status


def log_proposal(exp_num, desc, old_line, new_line, status, val_bpb, vram, wall_time):
    """Log full proposal details to JSONL for qualitative analysis."""
    entry = {
        "condition": "baseline",
        "exp": exp_num,
        "description": desc,
        "old_line": old_line,
        "new_line": new_line,
        "status": status,
        "val_bpb": val_bpb,
        "vram_gb": round(vram, 1) if vram else None,
        "wall_seconds": round(wall_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Stopping logic
# ---------------------------------------------------------------------------

def should_stop(results_history, total_experiments):
    """Check stopping criteria. Returns (should_stop, reason)."""
    if total_experiments >= MAX_EXPERIMENTS:
        return True, f"max_experiments ({MAX_EXPERIMENTS})"

    # Count consecutive discards from the end
    consec_discards = 0
    for status in reversed(results_history):
        if status in ("discard", "crash", "skip"):
            consec_discards += 1
        else:
            break
    if consec_discards >= CONSEC_DISCARD_LIMIT:
        return True, f"consecutive_discards ({consec_discards} >= {CONSEC_DISCARD_LIMIT})"

    # Check if any KEEP in last NO_KEEP_WINDOW experiments
    if len(results_history) >= NO_KEEP_WINDOW:
        recent = results_history[-NO_KEEP_WINDOW:]
        if "keep" not in recent:
            return True, f"no_keep_in_window (last {NO_KEEP_WINDOW} experiments)"

    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)  # reproducible for the paper

    print("=" * 60)
    print("  LocalPilot -- Baseline v2 (Condition A)")
    print("  Random greedy hill-climbing, no literature grounding")
    print("=" * 60)

    # Initialize results file
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\twall_seconds\n")

    # Read existing results for resume support
    results_history = []
    best_bpb = STARTING_BPB
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    results_history.append(row.get("status", "discard"))
                    if row.get("status") == "keep":
                        bpb = float(row["val_bpb"])
                        if bpb < best_bpb:
                            best_bpb = bpb
        except Exception:
            pass

    total_done = len(results_history)
    print(f"\nStarting from val_bpb={best_bpb:.6f}, {total_done} experiments done")
    print(f"Stopping: {CONSEC_DISCARD_LIMIT} consecutive discards, "
          f"or no KEEP in {NO_KEEP_WINDOW}, or {MAX_EXPERIMENTS} total\n")

    run_start = time.time()

    while True:
        total_done = len(results_history)

        # Check stopping criteria
        stop, reason = should_stop(results_history, total_done)
        if stop:
            print(f"\n{'='*60}")
            print(f"STOPPING: {reason}")
            print(f"Total experiments: {total_done}")
            print(f"Best val_bpb: {best_bpb:.6f}")
            print(f"Total wall time: {time.time() - run_start:.0f}s")
            print(f"Results: {RESULTS_FILE}")
            print(f"Proposals: {PROPOSALS_LOG}")
            print(f"{'='*60}")

            # Log stopping reason
            with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "condition": "baseline",
                    "event": "stopped",
                    "reason": reason,
                    "total_experiments": total_done,
                    "best_bpb": best_bpb,
                    "total_wall_seconds": round(time.time() - run_start, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }) + "\n")
            break

        # Generate a random proposal
        content = read_file("train.py")
        proposal = generate_proposal(content)
        if proposal is None:
            continue  # rare edge case, just retry

        desc, old_line, new_line = proposal
        exp_num = total_done + 1

        best_bpb, val_bpb, status = run_experiment(
            desc, old_line, new_line, best_bpb, exp_num
        )
        results_history.append(status)

        keeps = sum(1 for s in results_history if s == "keep")
        print(f"\n  [{exp_num}] best={best_bpb:.6f}  keeps={keeps}  "
              f"consec_discards={_count_tail_discards(results_history)}")

    print("\nBaseline v2 complete.")


def _count_tail_discards(history):
    count = 0
    for s in reversed(history):
        if s in ("discard", "crash", "skip"):
            count += 1
        else:
            break
    return count


if __name__ == "__main__":
    main()
