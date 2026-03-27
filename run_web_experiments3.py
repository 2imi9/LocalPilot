"""
Condition B: Web-enhanced autoresearch — Batch 3.

Papers sourced via MolmoWeb visual browser on 2026-03-27:
  [10] Qian et al. 2025, "Muon is Provably Faster with Momentum Variance Reduction"
       arXiv:2512.16598 — MVR (momentum variance reduction) accelerates Muon convergence.
       Key: lower beta2 in Muon's variance estimator → more aggressive variance reduction.
  [11] Qiu et al. 2026, "Hyperparameter Transfer Enables Consistent Gains of
       Matrix-Preconditioned Optimizers Across Scales"
       arXiv:2512.05620 — LR for Muon-like optimizers transfers predictably across scales.
       Key: slightly higher MATRIX_LR is likely safe; 0.06 keep suggests 0.065-0.07 in range.

Results appended to results_enhanced.tsv.
Best starting point: 1.119922 (MATRIX_LR=0.06 from batch 2).
"""
import subprocess
import re
import os
import pathlib

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
RESULTS_FILE = "results_enhanced.tsv"
BEST_BPB = 1.119922  # best from batch 2


experiments = [
    # [10] Qian 2025 MVR: lower Muon beta2 for more aggressive variance reduction
    ("muon_beta2=0.90 [Qian2025-MVR-lower-beta2]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.90, weight_decay=weight_decay,"),
    ]),

    # [10] Qian 2025 MVR: even lower Muon beta2 (more aggressive)
    ("muon_beta2=0.85 [Qian2025-MVR-aggressive]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,"),
    ]),

    # [11] Qiu 2026 HT: push MATRIX_LR further (0.06→0.065, incremental from keep)
    ("MATRIX_LR=0.065 [Qiu2026-HT-fine-tune]", [
        ("train.py",
         "MATRIX_LR = 0.06        # slightly higher Muon LR (SpectralClipping 2026)",
         "MATRIX_LR = 0.065       # incremental step from 0.06 keep (Qiu 2026 HT)"),
    ]),

    # [11] Qiu 2026 HT: MATRIX_LR=0.07 without warmup (previously tried with warmup)
    ("MATRIX_LR=0.07 [Qiu2026-HT-higher]", [
        ("train.py",
         "MATRIX_LR = 0.06        # slightly higher Muon LR (SpectralClipping 2026)",
         "MATRIX_LR = 0.07        # higher Muon LR, no-warmup config (Qiu 2026 HT)"),
    ]),

    # [10] Qian 2025 + [11] Qiu 2026: combine lower beta2 + higher LR
    ("muon_beta2=0.90+MATRIX_LR=0.065 [Qian2025+Qiu2026-combined]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.90, weight_decay=weight_decay,"),
        ("train.py",
         "MATRIX_LR = 0.06        # slightly higher Muon LR (SpectralClipping 2026)",
         "MATRIX_LR = 0.065       # combined with lower beta2 (Qian+Qiu 2025/2026)"),
    ]),

    # [10] Qian 2025: higher muon momentum (0.97→0.98) with MVR insight
    ("muon_momentum=0.98+beta2=0.90 [Qian2025-MVR-high-mom]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.98, ns_steps=5, beta2=0.90, weight_decay=weight_decay,"),
    ]),

    # Intermediate weight decay (between discarded 0.10 and current 0.15)
    ("WEIGHT_DECAY=0.12 [Qiu2026-HT-wd-tune]", [
        ("train.py",
         "WEIGHT_DECAY = 0.15     # slightly more weight decay",
         "WEIGHT_DECAY = 0.12     # intermediate WD between 0.10 and 0.15 (Qiu 2026 HT)"),
    ]),

    # Scalar LR fine-tune (between discarded 0.3 and current 0.5)
    ("SCALAR_LR=0.4 [Qian2025-scalar-stability]", [
        ("train.py",
         "SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)",
         "SCALAR_LR = 0.4         # lower scalar LR for stability (Qian 2025 MVR)"),
    ]),
]


def read_file(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
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
    subprocess.run(["git", "commit", "-m", f"web-exp3: {desc}"], capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    print(f"  Training... ", end="", flush=True)
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})

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
    best = BEST_BPB
    print(f"Starting web-enhanced experiments batch 3 (Condition B).")
    print(f"Starting val_bpb: {best} (best from batch 2)")
    print(f"Running {len(experiments)} web-sourced experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nCompare with baseline: python analyze.py")
