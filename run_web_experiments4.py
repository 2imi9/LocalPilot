"""
Condition B: Web-enhanced autoresearch — Batch 4.

Continuing exploration around current best config (muon_beta2=0.85, MATRIX_LR=0.07).
Papers referenced:
  [10] Qian et al. 2025, "Muon is Provably Faster with Momentum Variance Reduction"
       arXiv:2512.16598 — MVR: further beta2 reduction and momentum tuning.
  [11] Qiu et al. 2026, "Hyperparameter Transfer for Matrix-Preconditioned Optimizers"
       arXiv:2512.05620 — LR continues to scale; try MATRIX_LR=0.08.
  [8]  OptimalLRS 2026 — FINAL_LR_FRAC already near-zero; try absolute zero.

Results appended to results_enhanced.tsv.
Best starting point: 1.118972 (muon_beta2=0.85 + MATRIX_LR=0.07 from batch 3).
"""
import subprocess
import re
import os
import pathlib

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
RESULTS_FILE = "results_enhanced.tsv"
BEST_BPB = 1.118972  # best from batch 3


experiments = [
    # [10] Qian 2025 MVR: push beta2 lower (0.85→0.80)
    ("muon_beta2=0.80 [Qian2025-MVR-push-lower]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.80, weight_decay=weight_decay,"),
    ]),

    # [10] Qian 2025 MVR: even lower beta2 (0.75)
    ("muon_beta2=0.75 [Qian2025-MVR-very-low]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.75, weight_decay=weight_decay,"),
    ]),

    # [11] Qiu 2026 HT: push MATRIX_LR further (0.07→0.08)
    ("MATRIX_LR=0.08 [Qiu2026-HT-push-higher]", [
        ("train.py",
         "MATRIX_LR = 0.07        # higher Muon LR, no-warmup config (Qiu 2026 HT)",
         "MATRIX_LR = 0.08        # push Muon LR further (Qiu 2026 HT)"),
    ]),

    # [11] Qiu 2026 HT: MATRIX_LR=0.09 (aggressive)
    ("MATRIX_LR=0.09 [Qiu2026-HT-aggressive]", [
        ("train.py",
         "MATRIX_LR = 0.07        # higher Muon LR, no-warmup config (Qiu 2026 HT)",
         "MATRIX_LR = 0.09        # aggressive Muon LR (Qiu 2026 HT)"),
    ]),

    # Adam beta2 tuning: lower beta2 for Adam (matches Muon beta2 philosophy)
    ("ADAM_BETAS=(0.9,0.90) [Qian2025-adam-beta2]", [
        ("train.py",
         "ADAM_BETAS = (0.9, 0.95) # higher beta1 for gradient stability (Li 2026 AGGC)",
         "ADAM_BETAS = (0.9, 0.90) # lower Adam beta2 matching MVR insight (Qian 2025)"),
    ]),

    # muon_momentum=0.96 (between 0.95 discard and 0.97 current best)
    ("muon_momentum=0.96 [Qian2025-MVR-momentum]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.96, ns_steps=5, beta2=0.85, weight_decay=weight_decay,"),
    ]),

    # EMBEDDING_LR=0.8 (slightly reduced from 1.0; 1.5 was discarded earlier)
    ("EMBEDDING_LR=0.8 [Qiu2026-HT-embed-lr]", [
        ("train.py",
         "EMBEDDING_LR = 1.0      # moderate embedding LR",
         "EMBEDDING_LR = 0.8      # slightly lower embedding LR (Qiu 2026 HT)"),
    ]),

    # FINAL_LR_FRAC=0.0 (true zero — currently 0.001)
    ("FINAL_LR_FRAC=0.0 [OptimalLRS2026-true-zero]", [
        ("train.py",
         "FINAL_LR_FRAC = 0.001   # near-zero floor per Defazio 2023 adaptive schedule",
         "FINAL_LR_FRAC = 0.0     # true zero final LR (OptimalLRS 2026 optimal decay)"),
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
    subprocess.run(["git", "commit", "-m", f"web-exp4: {desc}"], capture_output=True)
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
    print(f"Starting web-enhanced experiments batch 4 (Condition B).")
    print(f"Starting val_bpb: {best} (best from batch 3)")
    print(f"Running {len(experiments)} web-sourced experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nCompare with baseline: python analyze.py")
