"""
Condition B: Web-enhanced autoresearch — Batch 5.

Fresh ideas from targeted web search (2025-2026 papers):
  [12] "Muon is Scalable for LLM Training" (arXiv:2502.16982, Moonlight)
       — Weight decay tuning, higher Nesterov momentum (0.98) works better
  [13] "Variance-Adaptive Muon: Muon-NSR and Muon-VS" (arXiv:2601.14603)
       — More Newton-Schulz steps improve orthogonalization quality
  [14] "Benchmarking Optimizers for LLM Pretraining" (arXiv:2509.01440)
       — Comprehensive sweep: Adam beta1=0.85 sometimes outperforms 0.9
  [15] "Straight to Zero: Linear Decay to Zero Works Best" (arXiv:2502.15938, Cerebras)
       — Ultra-short warmup (1%) with linear D2Z
  [16] "Fantastic Pretraining Optimizers" (arXiv:2509.02046)
       — Scalar/embedding LR often under-tuned; try higher values
  [17] "Cooldown Stage for LLM" (arXiv:2508.01483, TMLR 2025)
       — Unembedding LR sensitivity; lower unembedding LR stabilizes cooldown

Results appended to results_enhanced.tsv.
Best starting point: 1.118972 (muon_beta2=0.85 + MATRIX_LR=0.07 from batch 3).
"""
import subprocess
import re
import os
import pathlib

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
RESULTS_FILE = "results_enhanced.tsv"
BEST_BPB = 1.118972  # best from batch 3 (batch 4 found no improvements)


experiments = [
    # [13] Variance-Adaptive Muon: more Newton-Schulz steps → better orthogonalization
    ("ns_steps=6 [VarAdaptiveMuon2026-more-ortho]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=6, beta2=0.85, weight_decay=weight_decay,"),
    ]),

    # [12] Muon Scalable (Moonlight): higher Nesterov momentum 0.98
    ("muon_momentum=0.98 [Moonlight2025-high-momentum]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.98, ns_steps=5, beta2=0.85, weight_decay=weight_decay,"),
    ]),

    # [12] Moonlight: fine-tune momentum at 0.975 (between 0.97 and 0.98)
    ("muon_momentum=0.975 [Moonlight2025-fine-tune]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.85, weight_decay=weight_decay,",
         "                momentum=0.975, ns_steps=5, beta2=0.85, weight_decay=weight_decay,"),
    ]),

    # [12] Moonlight: lower weight decay (0.15 → 0.10) — MoE models use lower WD
    ("WEIGHT_DECAY=0.10 [Moonlight2025-lower-wd]", [
        ("train.py",
         "WEIGHT_DECAY = 0.15     # slightly more weight decay",
         "WEIGHT_DECAY = 0.10     # lower WD matching Moonlight MoE recipe (Muon Scalable 2025)"),
    ]),

    # [12] Moonlight: higher weight decay (0.15 → 0.20) — dense models may need more
    ("WEIGHT_DECAY=0.20 [Moonlight2025-higher-wd]", [
        ("train.py",
         "WEIGHT_DECAY = 0.15     # slightly more weight decay",
         "WEIGHT_DECAY = 0.20     # higher WD for dense model (Moonlight 2025 sensitivity)"),
    ]),

    # [14] Benchmarking 2025: lower Adam beta1 (0.9 → 0.85) can outperform default
    ("ADAM_BETAS=(0.85,0.95) [Benchmarking2025-lower-beta1]", [
        ("train.py",
         "ADAM_BETAS = (0.9, 0.95) # higher beta1 for gradient stability (Li 2026 AGGC)",
         "ADAM_BETAS = (0.85, 0.95) # lower beta1, sometimes better (Benchmarking 2025)"),
    ]),

    # [15] Straight-to-Zero: ultra-short warmup (1%) with linear decay to zero
    ("WARMUP_RATIO=0.01 [StraightToZero2025-ultra-short]", [
        ("train.py",
         "WARMUP_RATIO = 0.03     # short warmup + long linear decay (Defazio 2023 adaptive refinement)",
         "WARMUP_RATIO = 0.01     # ultra-short warmup, linear D2Z optimal (Straight-to-Zero 2025)"),
    ]),

    # [16] Fantastic Optimizers: scalar LR often under-tuned; try higher (0.5 → 0.6)
    ("SCALAR_LR=0.6 [Fantasia2025-scalar-lr]", [
        ("train.py",
         "SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)",
         "SCALAR_LR = 0.6         # slightly higher scalar LR (Fantastic Optimizers 2025)"),
    ]),

    # [16] Fantastic Optimizers: push scalar LR higher (0.5 → 0.7)
    ("SCALAR_LR=0.7 [Fantasia2025-scalar-higher]", [
        ("train.py",
         "SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)",
         "SCALAR_LR = 0.7         # higher scalar LR (Fantastic Optimizers 2025)"),
    ]),

    # [17] Cooldown 2025: lower unembedding LR stabilizes late training
    ("UNEMBEDDING_LR=0.006 [Cooldown2025-lower-unembed]", [
        ("train.py",
         "UNEMBEDDING_LR = 0.008  # learning rate for lm_head (Adam) \u2014 doubled",
         "UNEMBEDDING_LR = 0.006  # lower unembedding LR for cooldown stability (TMLR 2025)"),
    ]),

    # [17] Cooldown 2025: higher unembedding LR
    ("UNEMBEDDING_LR=0.012 [Cooldown2025-higher-unembed]", [
        ("train.py",
         "UNEMBEDDING_LR = 0.008  # learning rate for lm_head (Adam) \u2014 doubled",
         "UNEMBEDDING_LR = 0.012  # higher unembedding LR (Cooldown TMLR 2025)"),
    ]),

    # [16] Fantastic Optimizers: embedding LR often under-tuned; try higher (1.0 → 1.2)
    ("EMBEDDING_LR=1.2 [Fantasia2025-embed-lr]", [
        ("train.py",
         "EMBEDDING_LR = 1.0      # moderate embedding LR",
         "EMBEDDING_LR = 1.2      # higher embedding LR (Fantastic Optimizers 2025)"),
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
    subprocess.run(["git", "commit", "-m", f"web-exp5: {desc}"], capture_output=True)
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
    print(f"Starting web-enhanced experiments batch 5 (Condition B).")
    print(f"Starting val_bpb: {best} (best from batch 3)")
    print(f"Running {len(experiments)} web-sourced experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nCompare with baseline: python analyze.py")
