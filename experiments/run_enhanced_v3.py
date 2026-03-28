"""
LocalPilot -- Three-agent enhanced runner (Condition C v2).

Three-agent pipeline, all local, no cloud APIs:
  1. Qwen3.5-9B (orchestrator) reviews experiment history, decides direction
  2. MolmoWeb-4B visually browses arXiv based on orchestrator's query
  3. Qwen3.5-9B reviews findings + history, writes precise proposal instruction
  4. Devstral-24B writes the train.py patch from orchestrator's instruction
  5. Standard greedy keep/discard training loop

VRAM usage (sequential, never simultaneous):
  Orchestrate: Qwen3.5-9B Q6_K  ~8 GB
  Browse:      MolmoWeb-4B       ~8 GB
  Code:        Devstral Q6_K     ~19 GB
  Train:       GPT model          ~6 GB

Stopping criteria (same as baseline for fair comparison):
  - Primary: CONSEC_DISCARD_LIMIT consecutive discards (default 15)
  - Secondary: no KEEP in last NO_KEEP_WINDOW experiments (default 20)
  - Safety cap: MAX_EXPERIMENTS (default 80)

Usage:
    cd /path/to/localpilot
    uv run python experiments/run_enhanced_v3.py
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
RESULTS_FILE = ROOT / "results_enhanced_v3.tsv"
PROPOSALS_LOG = ROOT / "proposals_enhanced_v3.jsonl"
RESEARCH_LOG = ROOT / "research_log_v3.md"
QUEUE_FILE   = ROOT / "patch_queue_v3.json"

LLAMA_SERVER_EXE = str(ROOT.parent / "llama.cpp" / "llama-server.exe")

# Model paths
QWEN_MODEL   = str(ROOT.parent / "models" / "qwen3.5-9b" /
                    "Qwen3.5-9B-Q6_K.gguf")
DEVSTRAL_MODEL = str(ROOT.parent / "models" / "devstral" /
                     "Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf")

LLAMA_PORT     = 8080
LLAMA_API_BASE = f"http://localhost:{LLAMA_PORT}"

# Stopping criteria (identical to baseline)
CONSEC_DISCARD_LIMIT = 15
NO_KEEP_WINDOW       = 20
MAX_EXPERIMENTS      = 80

PATCHES_PER_SESSION  = 6
REFILL_THRESHOLD     = 2

# Starting best (raw karpathy config)
STARTING_BPB = 1.379

# ArXiv topics (fallback if orchestrator gives bad query)
FALLBACK_TOPICS = [
    "compute optimal depth width transformer language model small scale 2024 2025",
    "Adam optimizer beta1 beta2 momentum transformer training 2024 2025",
    "linear learning rate warmup warmdown schedule transformer pretraining 2023 2024",
    "weight decay embedding learning rate language model tricks",
    "transformer architecture scaling depth width head dimension efficiency",
]

_llama_proc = None
_current_model = None  # track which model is loaded


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def get_hyperparams_block():
    """Get numbered hyperparameter lines from train.py."""
    content = read_file("train.py")
    hp_lines = [l for l in content.splitlines()
                if re.match(r'^[A-Z_]+ = ', l)][:20]
    return hp_lines

def format_experiment_history(results_history, tried_descs):
    """Format experiment history for the orchestrator to review."""
    if not results_history:
        return "No experiments run yet. Starting from karpathy baseline (val_bpb=1.379)."
    lines = []
    for i, (status, desc) in enumerate(zip(results_history, tried_descs), 1):
        lines.append(f"  {i}. [{status.upper()}] {desc}")
    # Show recent only if too many
    if len(lines) > 20:
        lines = lines[:5] + ["  ..."] + lines[-15:]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# llama-server management (shared between Qwen and Devstral)
# ---------------------------------------------------------------------------

def start_llama_server(model_path, ctx_size=4096, threads=8, label="model"):
    """Start llama-server with the specified model. Stops any running server first."""
    global _llama_proc, _current_model

    # If already running with this model, reuse
    if _current_model == model_path and _llama_proc is not None:
        try:
            r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass

    # Stop any existing server
    stop_llama_server()

    print(f"  [Loading {label}...]")
    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "-ngl", "99",
        "--port", str(LLAMA_PORT),
        "--ctx-size", str(ctx_size),
        "-t", str(threads),
        "--no-mmap",
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]
    _llama_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _current_model = model_path

    # Poll until ready
    for _ in range(120):
        try:
            r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"  [{label} ready]")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{label} did not become ready in time")


def stop_llama_server():
    global _llama_proc, _current_model
    if _llama_proc is None:
        return
    _llama_proc.terminate()
    try:
        _llama_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _llama_proc.kill()
    _llama_proc = None
    _current_model = None
    time.sleep(2)  # let VRAM release
    print("  [server stopped, VRAM freed]")


def call_llm(prompt, max_tokens=512, temperature=0.7, stop=None):
    """Call the currently loaded model via llama-server OpenAI-compat API."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    r = requests.post(
        f"{LLAMA_API_BASE}/v1/chat/completions",
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Stage 1: Orchestrator plans search query
# ---------------------------------------------------------------------------

def orchestrator_plan_search(hp_lines, results_history, tried_descs, best_bpb):
    """
    Qwen3.5-9B reviews experiment history and decides what arXiv query to run.
    Returns a search query string.
    """
    start_llama_server(QWEN_MODEL, ctx_size=4096, label="Qwen3.5-9B orchestrator")

    hp_numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(hp_lines))
    history = format_experiment_history(results_history, tried_descs)

    prompt = f"""You are an ML research orchestrator tuning a GPT language model.
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.

Current hyperparameters in train.py:
{hp_numbered}

Experiment history (most recent last):
{history}

Your task: Decide what research direction to explore NEXT.
- Look at what has been tried and what worked (KEEP) vs failed (DISCARD)
- DO NOT repeat directions that have already been tried multiple times
- Focus on parameters or techniques that haven't been explored yet
- Consider interactions between parameters

Output EXACTLY two lines:
SEARCH: <an arXiv search query, 5-15 words, specific to the technique you want to explore>
REASON: <one sentence explaining why this direction is promising>
"""

    try:
        response = call_llm(prompt, max_tokens=200, temperature=0.7,
                            stop=["\n\n\n"])
        # Parse search query
        search_m = re.search(r"SEARCH:\s*(.+)", response)
        if search_m:
            query = search_m.group(1).strip().strip('"').strip("'")
            reason_m = re.search(r"REASON:\s*(.+)", response)
            reason = reason_m.group(1).strip() if reason_m else ""
            print(f"  [Orchestrator] Search: {query}")
            if reason:
                print(f"  [Orchestrator] Reason: {reason[:80]}")
            return query
    except Exception as e:
        print(f"  [Orchestrator plan error: {e}]")

    # Fallback: cycle through default topics
    idx = len(results_history) % len(FALLBACK_TOPICS)
    return FALLBACK_TOPICS[idx]


# ---------------------------------------------------------------------------
# Stage 2: MolmoWeb visual browsing (same as v2)
# ---------------------------------------------------------------------------

def research_session(topic, parallel=True):
    """
    Browse arXiv using MolmoWeb (visual agent).

    If parallel=True, keeps Qwen3.5-9B loaded on llama-server while MolmoWeb
    runs via HuggingFace transformers. Both fit in 24GB VRAM:
      Qwen3.5-9B Q6_K via llama-server: ~8 GB
      MolmoWeb-4B via transformers:     ~8 GB
      Total:                            ~16 GB (24 GB available)

    This saves ~60-90s per research session by avoiding model unload/reload.
    """
    if not parallel:
        stop_llama_server()  # sequential mode: free VRAM first

    print(f"\n  [Research] '{topic}' {'(parallel)' if parallel else '(sequential)'}")

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
# Stage 3: Orchestrator reviews findings, writes precise proposal
# ---------------------------------------------------------------------------

def orchestrator_propose(hp_lines, paper_ideas, results_history,
                         tried_descs, best_bpb, n=PATCHES_PER_SESSION):
    """
    Qwen3.5-9B reviews paper findings + history, then writes precise
    proposal instructions for Devstral. Returns list of proposal dicts.
    """
    start_llama_server(QWEN_MODEL, ctx_size=8192, label="Qwen3.5-9B orchestrator")

    hp_numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(hp_lines))
    history = format_experiment_history(results_history, tried_descs)

    # Deduplicate: track what was recently tried
    recent_changes = set()
    for desc in tried_descs[-20:]:
        # Extract parameter name from description
        m = re.match(r"(\w+)\s*[=:]", desc)
        if m:
            recent_changes.add(m.group(1))

    proposals = []

    for attempt in range(n * 2):
        if len(proposals) >= n:
            break

        prompt = f"""You are an ML research orchestrator tuning a GPT language model.
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.

Current hyperparameters (numbered):
{hp_numbered}

Recent experiment history:
{history}

Paper findings from arXiv:
{paper_ideas[:2000]}

Recently tried parameters (AVOID these): {', '.join(recent_changes) if recent_changes else 'none'}

Propose a SINGLE hyperparameter change. Requirements:
- Must be DIFFERENT from recently tried parameters listed above
- Must be grounded in the paper findings or ML knowledge
- Explain the reasoning with a specific paper citation

Output EXACTLY these 3 lines:
LINE: <number from the list above>
VALUE: <new value for that parameter>
REASON: <why, with citation> [AuthorYear]

Example:
LINE: 3
VALUE: 96
REASON: Larger head dim improves attention quality for small models [Vaswani2017]
"""

        try:
            response = call_llm(prompt, max_tokens=200, temperature=0.6,
                                stop=["\n\n\n"])
            line_m = re.search(r"LINE:\s*(\d+)", response)
            val_m  = re.search(r"VALUE:\s*(.+)", response)
            desc_m = re.search(r"REASON:\s*(.+)", response)

            if not all([line_m, val_m, desc_m]):
                continue

            idx = int(line_m.group(1)) - 1
            if idx < 0 or idx >= len(hp_lines):
                continue

            old_line = hp_lines[idx]
            new_val = val_m.group(1).strip()
            desc = desc_m.group(1).strip()

            # Construct new line
            eq_pos = old_line.index(" = ")
            comment_pos = old_line.find(" # ", eq_pos)
            if comment_pos == -1:
                new_line = old_line[:eq_pos + 3] + new_val
            else:
                new_line = old_line[:eq_pos + 3] + new_val + old_line[comment_pos:]

            if new_line == old_line:
                continue

            # Check for duplicate in this batch
            param_name = old_line.split(" = ")[0].strip()
            batch_params = {p["param"] for p in proposals}
            if param_name in batch_params:
                continue

            proposals.append({
                "desc": desc,
                "old": old_line,
                "new": new_line,
                "param": param_name,
            })
            print(f"    proposal {len(proposals)}: {desc[:70]}")

        except Exception as e:
            print(f"    [orchestrator propose error: {e}]")
            time.sleep(1)

    return proposals


# ---------------------------------------------------------------------------
# Stage 4: Devstral refines patch (optional -- for complex multi-line changes)
# ---------------------------------------------------------------------------

def devstral_refine(proposal, hp_lines):
    """
    For simple single-line changes, just use the orchestrator's proposal directly.
    Devstral is reserved for cases where the orchestrator's VALUE is ambiguous
    or the change requires multiple lines.

    Returns (old_line, new_line) or None.
    """
    # For now, the orchestrator produces clean LINE/VALUE output,
    # so we just validate and pass through.
    # Devstral refinement can be added later for multi-line edits.
    return proposal["old"], proposal["new"]


# ---------------------------------------------------------------------------
# Stage 5: Experiment runner (same keep/discard protocol as baseline)
# ---------------------------------------------------------------------------

def run_experiment(desc, old_line, new_line, best_bpb, exp_num):
    """Apply a patch, train, keep/discard."""
    # Unload any model before training
    stop_llama_server()

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
    subprocess.run(["git", "commit", "-m", f"enhanced-v3: {desc}"],
                   capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"  Training... ", end="", flush=True)
    try:
        result = subprocess.run(
            TRAIN_CMD, capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        log = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log = ""
        print("TIMEOUT")

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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_proposal(exp_num, desc, old_line, new_line, status, val_bpb, vram, wall_time):
    entry = {
        "condition": "enhanced_v3",
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

def log_research(topic, findings):
    with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {time.strftime('%Y-%m-%d %H:%M')} | {topic}\n\n")
        f.write(findings[:4000])
        f.write("\n\n---")


# ---------------------------------------------------------------------------
# Stopping logic (identical to baseline)
# ---------------------------------------------------------------------------

def should_stop(results_history, total_experiments):
    if total_experiments >= MAX_EXPERIMENTS:
        return True, f"max_experiments ({MAX_EXPERIMENTS})"

    consec_discards = 0
    for status in reversed(results_history):
        if status in ("discard", "crash", "skip"):
            consec_discards += 1
        else:
            break
    if consec_discards >= CONSEC_DISCARD_LIMIT:
        return True, f"consecutive_discards ({consec_discards} >= {CONSEC_DISCARD_LIMIT})"

    if len(results_history) >= NO_KEEP_WINDOW:
        recent = results_history[-NO_KEEP_WINDOW:]
        if "keep" not in recent:
            return True, f"no_keep_in_window (last {NO_KEEP_WINDOW} experiments)"

    return False, ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LocalPilot -- Enhanced v3 (Condition C)")
    print("  Qwen3.5-9B orchestrator -> MolmoWeb -> Devstral -> Train")
    print("=" * 60)

    # Check models exist
    for model_name, model_path in [("Qwen3.5-9B", QWEN_MODEL),
                                    ("Devstral", DEVSTRAL_MODEL)]:
        if not os.path.exists(model_path):
            print(f"\nERROR: {model_name} not found at {model_path}")
            print("Download with: huggingface-cli download unsloth/Qwen3.5-9B-GGUF "
                  "Qwen3.5-9B-Q6_K.gguf --local-dir <dir>")
            sys.exit(1)

    # Initialize results file
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\twall_seconds\n")

    if not RESEARCH_LOG.exists():
        RESEARCH_LOG.write_text(
            "# LocalPilot v3 Research Log\n\nQwen3.5-9B orchestrated arXiv browsing sessions.\n",
            encoding="utf-8",
        )

    # Read existing results for resume support
    results_history = []
    tried_descs = []
    best_bpb = STARTING_BPB
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    results_history.append(row.get("status", "discard"))
                    tried_descs.append(row.get("description", ""))
                    if row.get("status") == "keep":
                        bpb = float(row["val_bpb"])
                        if bpb < best_bpb:
                            best_bpb = bpb
        except Exception:
            pass

    # Load queued patches from disk (resume support)
    queue = []
    if QUEUE_FILE.exists():
        try:
            queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    total_done = len(results_history)
    print(f"\nStarting from val_bpb={best_bpb:.6f}, {total_done} experiments done, "
          f"{len(queue)} patches queued")
    print(f"Stopping: {CONSEC_DISCARD_LIMIT} consecutive discards, "
          f"or no KEEP in {NO_KEEP_WINDOW}, or {MAX_EXPERIMENTS} total\n")

    run_start = time.time()

    try:
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

                with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "condition": "enhanced_v3",
                        "event": "stopped",
                        "reason": reason,
                        "total_experiments": total_done,
                        "best_bpb": best_bpb,
                        "total_wall_seconds": round(time.time() - run_start, 1),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }) + "\n")
                break

            # ── Refill queue when running low ──────────────────────────
            if len(queue) < REFILL_THRESHOLD:
                hp_lines = get_hyperparams_block()

                # Phase 1: Orchestrator plans search (loads Qwen)
                search_query = orchestrator_plan_search(
                    hp_lines, results_history, tried_descs, best_bpb
                )

                # Phase 2: MolmoWeb browses IN PARALLEL with Qwen still loaded
                # Qwen (llama-server, ~8GB) + MolmoWeb (transformers, ~8GB) = ~16GB
                # Both fit in 24GB VRAM simultaneously
                paper_ideas = research_session(search_query, parallel=True)
                log_research(search_query, paper_ideas)

                # Phase 3: Orchestrator reviews findings + proposes
                # Qwen is still loaded from Phase 1 -- no reload needed!
                new_proposals = orchestrator_propose(
                    hp_lines, paper_ideas, results_history,
                    tried_descs, best_bpb, n=PATCHES_PER_SESSION,
                )

                # Unload orchestrator before training (need full VRAM for train)
                stop_llama_server()

                queue.extend(new_proposals)
                QUEUE_FILE.write_text(
                    json.dumps(queue, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"\n  Queue refilled: {len(queue)} patches ready")

                if not queue:
                    print("  [No valid patches generated, retrying]")
                    continue

            # ── Run next experiment from queue ─────────────────────────
            patch = queue.pop(0)
            QUEUE_FILE.write_text(
                json.dumps(queue, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            old_line, new_line = devstral_refine(patch, get_hyperparams_block())
            tried_descs.append(patch["desc"])

            exp_num = total_done + 1
            best_bpb, val_bpb, status = run_experiment(
                patch["desc"], old_line, new_line, best_bpb, exp_num
            )
            results_history.append(status)

            keeps = sum(1 for s in results_history if s == "keep")
            consec = _count_tail_discards(results_history)
            print(f"\n  [{exp_num}] best={best_bpb:.6f}  keeps={keeps}  "
                  f"consec_discards={consec}")

    finally:
        stop_llama_server()

    print("\nEnhanced v3 complete.")


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
