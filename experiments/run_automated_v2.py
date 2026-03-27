"""
LocalPilot -- Fully automated experiment runner (Condition B v2).

Full pipeline, all local, no cloud APIs:
  1. MolmoWeb-4B visually browses arXiv (Playwright + local vision model)
  2. Devstral-24B (llama-server) generates train.py patches from paper ideas
  3. Standard greedy keep/discard training loop (same as baseline)

VRAM usage (sequential, never simultaneous):
  Browse:   MolmoWeb-4B  ~8 GB
  Generate: Devstral Q6_K ~19 GB
  Train:    GPT model     ~6 GB

Usage:
    cd /path/to/localpilot
    uv run python experiments/run_automated_v2.py

Resume after interruption: just re-run -- patch queue and results are persisted.
"""

import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PYTHON       = str(ROOT / ".venv" / "Scripts" / "python.exe")
TRAIN_CMD    = [PYTHON, "train.py"]
RESULTS_FILE = ROOT / "results_enhanced_v2.tsv"
RESEARCH_LOG = ROOT / "research_log_v2.md"
QUEUE_FILE   = ROOT / "patch_queue_v2.json"

LLAMA_SERVER_EXE = str(ROOT.parent / "llama.cpp" / "llama-server.exe")
DEVSTRAL_MODEL   = str(ROOT.parent / "models" / "devstral" /
                       "Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf")
LLAMA_PORT     = 8080
LLAMA_API_BASE = f"http://localhost:{LLAMA_PORT}"

TARGET_EXPERIMENTS   = 83   # match baseline experiment count
PATCHES_PER_SESSION  = 10   # patches Devstral generates per research session
REFILL_THRESHOLD     = 3    # refill queue when fewer than this many patches remain

# ---------------------------------------------------------------------------
# ArXiv topics to browse -- ordered most-impactful first
# MolmoWeb will search each topic in turn, cycling if needed
# ---------------------------------------------------------------------------
BROWSE_TOPICS = [
    "compute optimal depth width transformer language model small scale 2024 2025",
    "batch size gradient steps language model pretraining compute budget",
    "Adam optimizer beta1 beta2 momentum transformer training 2024 2025",
    "linear learning rate warmup warmdown schedule transformer pretraining 2023 2024",
    "Muon optimizer Newton-Schulz momentum matrix learning rate neural network",
    "weight decay embedding unembedding learning rate language model tricks",
    "gradient clipping spectral norm matrix parameter optimization 2025 2026",
    "transformer architecture scaling depth width head dimension efficiency",
    "language model optimizer hyperparameter tuning survey 2024",
]

_llama_proc = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_current_best() -> float:
    if not RESULTS_FILE.exists():
        return float("inf")
    best = float("inf")
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("status") == "keep":
                    bpb = float(row["val_bpb"])
                    if bpb < best:
                        best = bpb
    except Exception:
        pass
    return best if best < float("inf") else float("inf")

def count_experiments() -> int:
    if not RESULTS_FILE.exists():
        return 0
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    except Exception:
        return 0

def extract_hyperparams(content: str) -> str:
    """Pull just the hyperparameter block out of train.py."""
    start = content.find("# Hyperparameters")
    end   = content.find("# -----------", start + 10) if start != -1 else -1
    if start != -1 and end != -1:
        return content[start:end].strip()
    # Fallback: grab all UPPER_CASE = ... lines
    lines = [l for l in content.splitlines() if re.match(r"^[A-Z_]+ = ", l)]
    return "\n".join(lines[:25])

def load_queue() -> list:
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def save_queue(queue: list):
    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False),
                          encoding="utf-8")

def log_research(topic: str, findings: str):
    with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {time.strftime('%Y-%m-%d %H:%M')} | {topic}\n\n")
        f.write(findings[:4000])
        f.write("\n\n---")


# ---------------------------------------------------------------------------
# Stage 1: MolmoWeb visual browsing
# ---------------------------------------------------------------------------

def research_session(topic: str) -> str:
    """
    Browse arXiv using MolmoWeb (visual agent). Falls back to text arXiv
    API if MolmoWeb fails or produces too little output.
    """
    print(f"\n  [Research] '{topic}'")

    browse_task = (
        f"Go to https://arxiv.org/search/?query={quote_plus(topic)}"
        f"&searchtype=all&order=-announced_date_first "
        f"Read the titles and abstracts of the first 6 results. "
        f"For each paper relevant to neural network training, optimizers, "
        f"or language model hyperparameters, extract: "
        f"(1) title and year, "
        f"(2) specific hyperparameter values or techniques recommended, "
        f"(3) reported improvement over baseline. "
        f"Signal completion with send_msg_to_user summarising all findings."
    )

    try:
        result = subprocess.run(
            [PYTHON, "-m", "localpilot.browse", "browse", browse_task],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 400:
            print(f"  [MolmoWeb: {len(output)} chars]")
            return output
        print(f"  [MolmoWeb: too little output ({len(output)} chars), falling back]")
    except subprocess.TimeoutExpired:
        print("  [MolmoWeb: timeout, falling back]")
    except Exception as e:
        print(f"  [MolmoWeb error: {e}, falling back]")

    # Fallback: lightweight arXiv text API
    print("  [Using arXiv text API fallback]")
    try:
        from localpilot.browse import extract_ideas
        return extract_ideas(topic)
    except Exception as e:
        return f"[research failed: {e}]"


# ---------------------------------------------------------------------------
# Stage 2: Devstral patch generation
# ---------------------------------------------------------------------------

def start_llama_server():
    global _llama_proc
    if _llama_proc is not None:
        return
    # If a server is already healthy on the port, reuse it (don't spawn a second)
    try:
        r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
        if r.status_code == 200:
            print("  [llama-server already running, reusing]")
            return
    except Exception:
        pass
    print("  [Starting llama-server (Devstral Q6_K)...]")
    cmd = [
        LLAMA_SERVER_EXE,
        "-m", DEVSTRAL_MODEL,
        "-ngl", "99",
        "--port", str(LLAMA_PORT),
        "--ctx-size", "8192",
        "-t", "8",
        "--no-mmap",
    ]
    _llama_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Poll until ready
    for _ in range(90):
        try:
            r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
            if r.status_code == 200:
                print("  [llama-server ready]")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready in time")


def stop_llama_server():
    global _llama_proc
    if _llama_proc is None:
        return
    _llama_proc.terminate()
    try:
        _llama_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _llama_proc.kill()
    _llama_proc = None
    time.sleep(1)
    print("  [llama-server stopped, VRAM freed]")


def _call_devstral(prompt: str) -> str:
    """Single call to Devstral via llama-server OpenAI-compat API."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.4,
        "stop": ["\n\n\n"],
    }
    r = requests.post(
        f"{LLAMA_API_BASE}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_patch(text: str) -> dict | None:
    """Parse Devstral output into {desc, old, new}."""
    desc_m = re.search(r"DESCRIPTION:\s*(.+)", text)
    old_m  = re.search(r"OLD:\s*(.+)",         text)
    new_m  = re.search(r"NEW:\s*(.+)",         text)
    if not all([desc_m, old_m, new_m]):
        return None
    desc = desc_m.group(1).strip()
    old  = old_m.group(1).strip()
    new  = new_m.group(1).strip()
    if old == new or not old or not new:
        return None
    return {"desc": desc, "old": old, "new": new}


def generate_patches(paper_ideas: str,
                     hyperparams: str,
                     current_best: float,
                     tried: list,
                     n: int = PATCHES_PER_SESSION) -> list:
    """
    Ask Devstral to generate n distinct patches from the paper ideas.
    Returns list of {desc, old, new} dicts.
    """
    tried_str = "\n".join(f"- {c}" for c in tried[-30:]) or "None yet"
    patches = []
    seen_olds = set()  # track OLD strings used in this session to avoid duplicates

    for attempt in range(n * 3):  # allow retries
        if len(patches) >= n:
            break

        # Vary the prompt slightly to get diverse suggestions
        extra = (
            f"\nFocus on {'architecture' if attempt < n//3 else 'optimizer' if attempt < 2*n//3 else 'schedule'} changes."
            if attempt % 3 == 0 else ""
        )

        # Extract just the numbered hyperparameter lines for clearer reference
        hp_lines = [l for l in hyperparams.splitlines()
                    if re.match(r'^[A-Z_]+ = ', l)]
        hp_numbered = "\n".join(f"  [{i+1}] {l}" for i, l in enumerate(hp_lines))

        prompt = f"""Task: propose ONE hyperparameter change for a GPT model training script.

The ONLY valid parameter lines (pick one number to change):
{hp_numbered}

Current best val_bpb: {current_best:.6f} (lower is better)

Already tried (avoid repeating):
{tried_str}

Relevant paper findings:
{paper_ideas[:1500]}{extra}

RULES:
1. Pick one line from the numbered list above
2. OLD = copy that line EXACTLY as shown (character for character, including spaces and comments)
3. NEW = same line but with ONE value changed
4. Cite a paper as [AuthorYear]

Example of correct format:
DESCRIPTION: Reduce weight decay to 0.1 per common GPT practice [Brown2020]
OLD: WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
NEW: WEIGHT_DECAY = 0.1      # cautious weight decay for Muon

Now produce ONE suggestion in that exact 3-line format:"""

        try:
            text = _call_devstral(prompt)
            patch = parse_patch(text)
            if patch and patch["old"] not in seen_olds:
                seen_olds.add(patch["old"])
                patches.append(patch)
                print(f"    patch {len(patches)}: {patch['desc'][:60]}")
        except Exception as e:
            print(f"    [generate error: {e}]")
            time.sleep(2)

    return patches


# ---------------------------------------------------------------------------
# Stage 3: Keep/discard experiment runner
# ---------------------------------------------------------------------------

def run_experiment(desc: str, changes: list, best_bpb: float):
    """Apply changes, train, evaluate, keep or revert. Same logic as run_web.py."""
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
            print(f"  SKIP: old_str not found: {old_str[:70]!r}")
            for fp, orig in originals.items():
                write_file(fp, orig)
            return best_bpb, None
        write_file(filepath, content.replace(old_str, new_str, 1))

    for filepath in originals:
        subprocess.run(["git", "add", filepath], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"auto-v2: {desc}"],
                   capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    print("  Training...", end="", flush=True)
    result = subprocess.run(
        TRAIN_CMD, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    log = result.stdout + result.stderr
    write_file("run.log", log)

    val_match  = re.search(r"^val_bpb:\s+([\d.]+)",    log, re.M)
    vram_match = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.M)

    if not val_match:
        print("CRASH")
        for fp, orig in originals.items():
            write_file(fp, orig)
        subprocess.run(["git", "add"] + list(originals.keys()), capture_output=True)
        subprocess.run(["git", "commit", "-m", f"revert: {desc} (crash)"],
                       capture_output=True)
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{commit}\t0.000000\t0.0\tcrash\t{desc}\n")
        return best_bpb, None

    val_bpb = float(val_match.group(1))
    vram    = float(vram_match.group(1)) / 1024 if vram_match else 0.0

    if val_bpb < best_bpb:
        status = "keep"
        print(f"KEEP: {val_bpb:.6f}  (was {best_bpb:.6f}, -{best_bpb - val_bpb:.6f})")
        best_bpb = val_bpb
    else:
        status = "discard"
        print(f"DISCARD: {val_bpb:.6f}  (best: {best_bpb:.6f})")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{vram:.1f}\t{status}\t{desc}\n")

    return best_bpb, val_bpb


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LocalPilot -- Automated v2 Run (Condition B)")
    print("  MolmoWeb -> Devstral -> Train -> Keep/Discard")
    print("=" * 60)

    # Init results file
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    if not RESEARCH_LOG.exists():
        RESEARCH_LOG.write_text(
            "# LocalPilot v2 Research Log\n\nMolmoWeb arXiv browsing sessions.\n",
            encoding="utf-8",
        )

    best_bpb  = read_current_best()
    exp_count = count_experiments()
    queue     = load_queue()
    tried     = []
    topic_idx = 0

    if best_bpb == float("inf"):
        best_bpb = 1.379  # karpathy original starting val_bpb
    print(f"\nStarting from val_bpb={best_bpb:.6f}, "
          f"{exp_count} experiments done, "
          f"{len(queue)} patches queued")
    print(f"Target: {TARGET_EXPERIMENTS} experiments\n")

    while exp_count < TARGET_EXPERIMENTS:

        # ── Refill queue when running low ──────────────────────────────
        if len(queue) < REFILL_THRESHOLD:
            topic = BROWSE_TOPICS[topic_idx % len(BROWSE_TOPICS)]
            topic_idx += 1

            # Stage 1: MolmoWeb browse
            paper_ideas = research_session(topic)
            log_research(topic, paper_ideas)

            # Stage 2: Devstral generate patches
            hyperparams = extract_hyperparams(read_file("train.py"))
            try:
                start_llama_server()
                new_patches = generate_patches(
                    paper_ideas, hyperparams, best_bpb, tried,
                    n=PATCHES_PER_SESSION,
                )
            finally:
                stop_llama_server()

            queue.extend(new_patches)
            save_queue(queue)
            print(f"\n  Queue refilled: {len(queue)} patches ready")

            if not queue:
                print("  [No valid patches generated, retrying next topic]")
                continue

        # ── Run next experiment from queue ─────────────────────────────
        patch = queue.pop(0)
        save_queue(queue)

        tried.append(patch["desc"])
        best_bpb, val_bpb = run_experiment(
            patch["desc"],
            [("train.py", patch["old"], patch["new"])],
            best_bpb,
        )
        if val_bpb is not None:  # None = SKIP (old_str not found), don't count
            exp_count += 1
        print(f"\n  [{exp_count}/{TARGET_EXPERIMENTS}] best={best_bpb:.6f}  queue={len(queue)}")

    print(f"\n{'='*60}")
    print(f"DONE. {exp_count} experiments complete.")
    print(f"Final best val_bpb: {best_bpb:.6f}")
    print(f"Results: {RESULTS_FILE}")
    print(f"Research log: {RESEARCH_LOG}")


if __name__ == "__main__":
    main()
