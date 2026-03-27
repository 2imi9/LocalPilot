"""
Condition B: Web-enhanced autoresearch — Batch 2.

New papers sourced from browse.py search on 2026-03-26:
  [6] Spectral Clipping (2026): "Enhancing LLM Training via Spectral Clipping"
      Semantic Scholar: fe7f7ec4... — Muon already uses N-S orthogonalization;
      spectral normalization quality depends on ns_steps iterations.
  [7] RMNP (2026): "Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization"
      Semantic Scholar: 5b8fae20... — Muon momentum value affects preconditioning.
  [8] OptimalLRS (2026): "Optimal LR Schedules under Functional Scaling Laws: Power Decay and WSD"
      Semantic Scholar: 6a86c63e... — WSD (Warmup-Stable-Decay) with stable phase
      outperforms pure linear decay; stable phase allows loss landscape exploration.
  [9] Anytime (2026): "Anytime Pretraining: Horizon-Free LR Schedules with Weight Averaging"
      Semantic Scholar: 6dd77aec... — Weight averaging at end smooths final model.

Results appended to results_enhanced.tsv.
Best starting point: 1.121178 (from batch 1 — Defazio2023 adaptive schedule).
"""
import subprocess
import re
import os
import pathlib

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
RESULTS_FILE = "results_enhanced.tsv"
BEST_BPB = 1.121178  # best from batch 1


experiments = [
    # [6] Spectral Clipping 2026: more N-S iterations → better spectral normalization
    ("ns_steps=10 [SpectralClipping2026-more-orthog]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=10, beta2=0.95, weight_decay=weight_decay,"),
    ]),

    # [6] Spectral Clipping 2026: fewer N-S steps (check if 5 is overkill)
    ("ns_steps=3 [SpectralClipping2026-fast]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=3, beta2=0.95, weight_decay=weight_decay,"),
    ]),

    # [7] RMNP 2026: lower Muon momentum (0.97→0.95) for less aggressive smoothing
    ("muon_momentum=0.95 [RMNP2026-lower-momentum]", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,"),
    ]),

    # [8] OptimalLRS 2026: WSD — add stable phase (warmdown=0.40 → stable=0.57)
    # Current: warmup=3%, stable=0%, decay=97%. WSD: warmup=3%, stable=57%, decay=40%
    ("WARMDOWN=0.40 [OptimalLRS2026-WSD-stable57]", [
        ("train.py",
         "WARMDOWN_RATIO = 0.97   # near-full linear decay per Defazio 2023",
         "WARMDOWN_RATIO = 0.40   # WSD stable phase 57%; decay only last 40% (OptimalLRS 2026)"),
    ]),

    # [8] OptimalLRS 2026: WSD — shorter decay (warmdown=0.20 → stable=0.77)
    ("WARMDOWN=0.20 [OptimalLRS2026-WSD-stable77]", [
        ("train.py",
         "WARMDOWN_RATIO = 0.97   # near-full linear decay per Defazio 2023",
         "WARMDOWN_RATIO = 0.20   # WSD: 77% stable, only last 20% decay (OptimalLRS 2026)"),
    ]),

    # [6] Spectral Clipping 2026: slightly higher Muon LR enabled by better orthogonalization
    ("MATRIX_LR=0.06 [SpectralClipping2026-higher-lr]", [
        ("train.py",
         "MATRIX_LR = 0.05        # slightly higher Muon LR for smaller model",
         "MATRIX_LR = 0.06        # slightly higher Muon LR (SpectralClipping 2026)"),
    ]),

    # [9] Anytime 2026: lower scalar LR for more stable per-layer scale training
    ("SCALAR_LR=0.3 [Anytime2026-scalar-stability]", [
        ("train.py",
         "SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)",
         "SCALAR_LR = 0.3         # lower scalar LR for stability (Anytime 2026)"),
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
    subprocess.run(["git", "commit", "-m", f"web-exp2: {desc}"], capture_output=True)
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
    print(f"Starting web-enhanced experiments batch 2 (Condition B).")
    print(f"Starting val_bpb: {best} (best from batch 1)")
    print(f"Running {len(experiments)} web-sourced experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nCompare with baseline: python analyze.py")
