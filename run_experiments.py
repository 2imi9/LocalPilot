"""
Automated experiment runner for autoresearch baseline.
Each experiment is a list of (filepath, old_str, new_str) applied atomically.
"""
import subprocess
import re
import os

TRAIN_CMD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"), "train.py"]
BEST_BPB = 1.122538  # current best (DEPTH=4 ASPECT=100 HEAD_DIM=128 muon_momentum=0.97)

experiments = [
    # Explore width: slightly wider (5 heads of 128)
    ("ASPECT_RATIO=160", [
        ("train.py",
         "ASPECT_RATIO = 100      # 4*100=400\u2192512 dim, 8 heads",
         "ASPECT_RATIO = 160      # 4*160=640\u2192640 dim, 5 heads of 128"),
    ]),

    # Explore depth: 5 layers (more layers, same dim)
    ("DEPTH=5", [
        ("train.py",
         "DEPTH = 4               # shallower, more width per layer",
         "DEPTH = 5               # one more layer"),
    ]),

    # Explore depth: 3 layers (fewer layers, wider)
    ("DEPTH=3", [
        ("train.py",
         "DEPTH = 4               # shallower, more width per layer",
         "DEPTH = 3               # very shallow, very wide"),
    ]),

    # Fine-tune Muon LR with new config
    ("MATRIX_LR=0.04", [
        ("train.py",
         "MATRIX_LR = 0.05        # slightly higher Muon LR for smaller model",
         "MATRIX_LR = 0.04        # lower Muon LR for wider model"),
    ]),

    # Try higher Muon momentum (currently 0.97)
    ("muon_momentum=0.98", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.98, ns_steps=5, beta2=0.95, weight_decay=weight_decay,"),
    ]),

    # More aggressive weight decay for wider model
    ("WEIGHT_DECAY=0.1", [
        ("train.py",
         "WEIGHT_DECAY = 0.15     # slightly more weight decay",
         "WEIGHT_DECAY = 0.1      # less weight decay for current config"),
    ]),

    # Try 2 KV heads (GQA) — head_dim=128, 4Q heads, 2KV heads
    ("n_kv_head=2", [
        ("train.py",
         "    return GPTConfig(\n        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,",
         "    return GPTConfig(\n        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads//2), n_embd=model_dim,"),
    ]),

    # Warmdown ratio fine-tune
    ("WARMDOWN_RATIO=0.82", [
        ("train.py",
         "WARMDOWN_RATIO = 0.8    # even longer cooldown",
         "WARMDOWN_RATIO = 0.82   # fine-tune warmdown"),
    ]),

    # FINAL_LR_FRAC=0.03
    ("FINAL_LR_FRAC=0.03", [
        ("train.py",
         "FINAL_LR_FRAC = 0.05    # smaller residual LR",
         "FINAL_LR_FRAC = 0.03    # even smaller final LR"),
    ]),

    # Muon beta2 (variance) higher
    ("muon_beta2=0.99", [
        ("train.py",
         "                momentum=0.97, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
         "                momentum=0.97, ns_steps=5, beta2=0.99, weight_decay=weight_decay,"),
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
    subprocess.run(["git", "commit", "-m", f"exp: {desc}"], capture_output=True)
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
        with open("results.tsv", "a", encoding="utf-8") as f:
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

    with open("results.tsv", "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{vram:.1f}\t{status}\t{desc}\n")

    return best_bpb, val_bpb


if __name__ == "__main__":
    best = BEST_BPB
    print(f"Starting automated experiments. Best val_bpb: {best}")
    print(f"Running {len(experiments)} experiments...")

    for i, (desc, changes) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        best, val = run_experiment(desc, changes, best)

    print(f"\n{'='*60}")
    print(f"DONE. Final best val_bpb: {best:.6f}")
    print(f"Results saved to results.tsv")
