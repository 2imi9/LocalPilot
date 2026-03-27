"""
Condition B: Web-enhanced autoresearch experiment runner.

Experiments sourced from arxiv web research via browse.py.
Results saved to results_enhanced.tsv for comparison with results_baseline.tsv.

Web research log:
  [1] Defazio et al. 2023, "Optimal Linear Decay Learning Rate Schedules"
      arXiv:2310.07831 — linear decay (1-t/T) outperforms cosine; ends near 0
  [2] Kalra & Barkeshli 2024 (NeurIPS), "Why Warmup the Learning Rate?"
      arXiv:2406.09405 — warmup enables larger peak LR; improves final performance
  [3] Ainslie et al. 2023 (EMNLP), "GQA: Grouped-Query Attention"
      arXiv:2305.13245 — MQA (1 KV head) for speed; GQA for quality-speed tradeoff
  [4] Li et al. 2026, "AGGC: Adaptive Group Gradient Clipping"
      arXiv:2601.11864 — per-group gradient clipping stabilizes training
  [5] Defazio et al. 2023 — adaptive refinement gives warmup + rapid end annealing
"""
import subprocess
import re
import os

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
RESULTS_FILE = "results_enhanced.tsv"
BEST_BPB = 1.122538  # same starting point as baseline

# Web-sourced experiments (paper → idea → implementation)
# Each entry: (description, paper_ref, changes)
experiments = [
    # [1] Defazio 2023: true linear decay from step 1 (WARMDOWN_RATIO=1.0 means decay starts at 0%)
    ("WARMDOWN_RATIO=1.0 [Defazio2023-linear-decay]", [
        ("train.py",
         "WARMDOWN_RATIO = 0.8    # even longer cooldown",
         "WARMDOWN_RATIO = 1.0    # true linear decay from step 1 (Defazio 2023 optimal schedule)"),
    ]),

    # [1] Defazio 2023: near-zero final LR for optimal linear decay
    ("FINAL_LR_FRAC=0.005 [Defazio2023-near-zero]", [
        ("train.py",
         "FINAL_LR_FRAC = 0.05    # smaller residual LR",
         "FINAL_LR_FRAC = 0.005   # near-zero final LR per Defazio 2023 optimal linear decay"),
    ]),

    # [1+2] Defazio+Kalra: warmdown 1.0 AND near-zero final (combined effect)
    ("WARMDOWN=1.0+FINAL_LR=0.005 [Defazio2023-combined]", [
        ("train.py",
         "WARMDOWN_RATIO = 0.8    # even longer cooldown",
         "WARMDOWN_RATIO = 1.0    # true linear decay from step 1 (Defazio 2023)"),
        ("train.py",
         "FINAL_LR_FRAC = 0.05    # smaller residual LR",
         "FINAL_LR_FRAC = 0.005   # near-zero final LR (Defazio 2023)"),
    ]),

    # [2] Kalra 2024 (NeurIPS): warmup enables larger peak LR
    ("WARMUP=0.05+MATRIX_LR=0.07 [Kalra2024-NeurIPS]", [
        ("train.py",
         "WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup",
         "WARMUP_RATIO = 0.05     # 5% warmup enables larger peak LR (Kalra NeurIPS 2024)"),
        ("train.py",
         "MATRIX_LR = 0.05        # slightly higher Muon LR for smaller model",
         "MATRIX_LR = 0.07        # higher LR enabled by warmup (Kalra NeurIPS 2024)"),
    ]),

    # [2] Kalra 2024: longer warmup (10%) with higher LR target
    ("WARMUP=0.1+MATRIX_LR=0.08 [Kalra2024-longer]", [
        ("train.py",
         "WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup",
         "WARMUP_RATIO = 0.10     # 10% warmup for higher peak LR (Kalra NeurIPS 2024)"),
        ("train.py",
         "MATRIX_LR = 0.05        # slightly higher Muon LR for smaller model",
         "MATRIX_LR = 0.08        # higher peak LR with 10% warmup"),
    ]),

    # [3] GQA paper: MQA (1 KV head) maximizes speed, may hurt quality
    ("n_kv_head=1 [Ainslie2023-MQA]", [
        ("train.py",
         "    return GPTConfig(\n        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,",
         "    return GPTConfig(\n        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=1, n_embd=model_dim,"),
    ]),

    # [4] AGGC paper: per-group gradient management — increase Adam beta1 for stability
    ("ADAM_BETAS=(0.9,0.95) [Li2026-AGGC-momentum]", [
        ("train.py",
         "ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2",
         "ADAM_BETAS = (0.9, 0.95) # higher beta1 for gradient stability (Li 2026 AGGC)"),
    ]),

    # [1] Defazio 2023: adaptive refinement — warmup 3% + linear decay to 0
    ("WARMUP=0.03+WARMDOWN=0.97+FINAL=0.001 [Defazio2023-adaptive]", [
        ("train.py",
         "WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup",
         "WARMUP_RATIO = 0.03     # short warmup + long linear decay (Defazio 2023 adaptive refinement)"),
        ("train.py",
         "WARMDOWN_RATIO = 0.8    # even longer cooldown",
         "WARMDOWN_RATIO = 0.97   # near-full linear decay per Defazio 2023"),
        ("train.py",
         "FINAL_LR_FRAC = 0.05    # smaller residual LR",
         "FINAL_LR_FRAC = 0.001   # near-zero floor per Defazio 2023 adaptive schedule"),
    ]),

    # [2] Kalra 2024: Adam with higher initial momentum (warmup alternative)
    ("ADAM_BETAS=(0.95,0.95) [Kalra2024-Flat-Adam]", [
        ("train.py",
         "ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2",
         "ADAM_BETAS = (0.95, 0.95) # Flat Adam variant: high beta1=beta2 (Kalra NeurIPS 2024)"),
    ]),

    # [1] Defazio 2023: weight decay also decays to 0 at end (matches their analysis)
    ("WEIGHT_DECAY=0.2 [Defazio2023-higher-wd]", [
        ("train.py",
         "WEIGHT_DECAY = 0.15     # slightly more weight decay",
         "WEIGHT_DECAY = 0.20     # higher WD, decays to 0 at end per Defazio 2023"),
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
    subprocess.run(["git", "commit", "-m", f"web-exp: {desc}"], capture_output=True)
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
    # Write header if file doesn't exist
    import pathlib
    if not pathlib.Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    best = BEST_BPB
    print(f"Starting web-enhanced experiments (Condition B).")
    print(f"Starting val_bpb: {best} (same as baseline)")
    print(f"Running {len(experiments)} web-sourced experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"\nCompare with baseline: python analyze.py")
