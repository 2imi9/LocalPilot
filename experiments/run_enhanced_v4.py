"""
LocalPilot -- Enhanced v4 runner (Condition C final).

Key improvements over v3:
  1. Open parameter values: LLM proposes exact values (not just direction)
  2. Working MolmoWeb browsing with relevance scoring
  3. OOM pre-flight check before training
  4. Research findings quality-filtered before reaching orchestrator

Agent design patterns adapted from Claude Code (Anthropic):
  5. Batch relevance scoring: all papers scored in one LLM call (not N calls)
  6. History compaction: old experiments summarized into structured digest
  7. Adaptive thinking: Qwen's /think mode enabled for proposal generation
  8. Post-proposal validation: stop hooks catch duplicates and exhausted params
  9. Structured error recovery: exponential backoff, error categorization,
     circuit breaker with random fallback after 3 consecutive failures

Pipeline:
  1. Qwen3.5-9B plans search query (thinking off, fast)
  2. Scholar API + arXiv API discovery (no model needed)
  3. Qwen3.5-9B batch-scores relevance (one call for all papers)
  4. MolmoWeb-4B deep-reads top papers (max 2, score >= 0.7)
  5. Qwen3.5-9B proposes PARAM + VALUE (thinking on, high quality)
  6. Validation pass: reject duplicates, exhausted params, OOM
  7. Train → keep/discard

VRAM usage (sequential, never simultaneous):
  Orchestrate: Qwen3.5-9B Q6_K  ~8 GB
  Browse:      MolmoWeb-4B       ~8 GB
  Train:       GPT model          ~6 GB

Stopping criteria (identical to baseline for fair comparison):
  - Primary: CONSEC_DISCARD_LIMIT consecutive discards (default 15)
  - Secondary: no KEEP in last NO_KEEP_WINDOW experiments (default 20)
  - Safety cap: MAX_EXPERIMENTS (default 80)

Usage:
    cd /path/to/localpilot
    uv run python experiments/run_enhanced_v4.py
"""

import csv
import gc
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PYTHON       = str(ROOT / ".venv" / "Scripts" / "python.exe")

# Training: local by default, Docker optional for FA3 on Hopper GPUs
TRAIN_CMD = [PYTHON, "train.py"]

# To use Docker instead (e.g. for H100 with Flash Attention 3):
#   DOCKER_IMAGE = "autoresearch-train"
#   def _to_docker_path(win_path):
#       posix = str(win_path).replace("\\", "/")
#       if sys.platform == "win32" and len(posix) > 1 and posix[1] == ":":
#           return "/" + posix[0].lower() + posix[2:]
#       return posix
#   _docker_mount = _to_docker_path(ROOT)
#   _cache_dir = Path.home() / ".cache" / "autoresearch"
#   _docker_cache = _to_docker_path(_cache_dir)
#   TRAIN_CMD = [
#       "docker", "run", "--rm", "--gpus", "all",
#       "-v", f"{_docker_mount}:/workspace",
#       "-v", f"{_docker_cache}:/root/.cache/autoresearch",
#       "-w", "/workspace",
#       DOCKER_IMAGE, "uv", "run", "python", "train.py",
#   ]
RESULTS_FILE = ROOT / "results" / "results_enhanced_v4.tsv"
PROPOSALS_LOG = ROOT / "results" / "proposals_enhanced_v4.jsonl"
RESEARCH_LOG = ROOT / "research_log_v4.md"
QUEUE_FILE   = ROOT / "patch_queue_v4.json"

LLAMA_SERVER_EXE = str(ROOT.parent / "llama.cpp" / "llama-server.exe")

# Model paths
QWEN_MODEL   = str(ROOT.parent / "models" / "qwen3.5-9b" /
                    "Qwen3.5-9B-Q6_K.gguf")

LLAMA_PORT     = 8080
LLAMA_API_BASE = f"http://localhost:{LLAMA_PORT}"

# Stopping criteria (identical to baseline)
CONSEC_DISCARD_LIMIT = 15
NO_KEEP_WINDOW       = 20
MAX_EXPERIMENTS      = 80

PATCHES_PER_SESSION  = 6
REFILL_THRESHOLD     = 2

# Starting best (raw karpathy config)
STARTING_BPB = 1.268

# Fallback search topics
FALLBACK_TOPICS = [
    "compute optimal depth width transformer language model small scale 2024 2025",
    "Adam optimizer beta1 beta2 momentum transformer training 2024 2025",
    "linear learning rate warmup warmdown schedule transformer pretraining 2023 2024",
    "weight decay embedding learning rate language model tricks",
    "transformer architecture scaling depth width head dimension efficiency",
]

# ---------------------------------------------------------------------------
# Safe parameter space (identical bounds to baseline for fair comparison)
# ---------------------------------------------------------------------------
SAFE_CONTINUOUS = {
    "EMBEDDING_LR":   {"min": 0.1,  "max": 2.0,   "fmt": "{:.1f}",  "step_frac": 0.25},
    "UNEMBEDDING_LR": {"min": 0.001,"max": 0.02,   "fmt": "{:.4f}",  "step_frac": 0.25},
    "MATRIX_LR":      {"min": 0.01, "max": 0.12,   "fmt": "{:.2f}",  "step_frac": 0.25},
    "SCALAR_LR":      {"min": 0.1,  "max": 1.5,    "fmt": "{:.1f}",  "step_frac": 0.25},
    "WEIGHT_DECAY":   {"min": 0.01, "max": 0.4,    "fmt": "{:.2f}",  "step_frac": 0.25},
    "WARMUP_RATIO":   {"min": 0.0,  "max": 0.3,    "fmt": "{:.2f}",  "step_frac": 0.5},
    "WARMDOWN_RATIO": {"min": 0.2,  "max": 1.0,    "fmt": "{:.1f}",  "step_frac": 0.15},
    "FINAL_LR_FRAC":  {"min": 0.0,  "max": 0.1,    "fmt": "{:.3f}",  "step_frac": 0.5},
}

SAFE_DISCRETE = {
    "DEPTH":            [4, 6, 8, 10, 12],
    "ASPECT_RATIO":     [32, 48, 64, 80, 96],
    "HEAD_DIM":         [48, 64, 96, 128],
    "TOTAL_BATCH_SIZE": ["2**17", "2**18", "2**19"],
    "WINDOW_PATTERN":   ['"SSSL"', '"SSLL"', '"SLLL"', '"LLLL"', '"SSSS"'],
}

SAFE_ADAM_BETAS = {
    "beta1": {"min": 0.7, "max": 0.95},
    "beta2": {"min": 0.9, "max": 0.999},
}

ALL_PARAM_NAMES = list(SAFE_CONTINUOUS.keys()) + list(SAFE_DISCRETE.keys()) + ["ADAM_BETAS"]


# ---------------------------------------------------------------------------
# Value clamping (new in V4: open values, clamped to bounds)
# ---------------------------------------------------------------------------

def clamp_value(param_name, proposed_value):
    """Clamp a proposed value to safe bounds. Returns safe value string."""
    proposed_value = str(proposed_value).strip()

    # --- ADAM_BETAS ---
    if param_name == "ADAM_BETAS":
        m = re.match(r"\(([^,]+),\s*([^)]+)\)", proposed_value)
        if not m:
            return proposed_value
        try:
            b1 = float(m.group(1))
            b2 = float(m.group(2))
        except ValueError:
            return proposed_value
        b1 = max(SAFE_ADAM_BETAS["beta1"]["min"],
                 min(SAFE_ADAM_BETAS["beta1"]["max"], b1))
        b2 = max(SAFE_ADAM_BETAS["beta2"]["min"],
                 min(SAFE_ADAM_BETAS["beta2"]["max"], b2))
        return f"({round(b1, 3)}, {round(b2, 3)})"

    # --- Discrete ---
    if param_name in SAFE_DISCRETE:
        options = SAFE_DISCRETE[param_name]
        # Exact match
        for opt in options:
            if str(opt).strip('"') == proposed_value.strip('"'):
                return str(opt)
        # Snap to nearest numeric
        try:
            val = float(proposed_value)
            numeric_opts = []
            for opt in options:
                opt_str = str(opt)
                try:
                    opt_val = eval(opt_str) if "**" in opt_str else float(opt_str.strip('"'))
                    numeric_opts.append((abs(opt_val - val), opt_str))
                except (ValueError, SyntaxError):
                    pass
            if numeric_opts:
                numeric_opts.sort()
                return numeric_opts[0][1]
        except ValueError:
            pass
        return str(options[0])

    # --- Continuous ---
    if param_name in SAFE_CONTINUOUS:
        spec = SAFE_CONTINUOUS[param_name]
        try:
            val = float(proposed_value)
        except ValueError:
            return proposed_value
        val = max(spec["min"], min(spec["max"], val))
        return spec["fmt"].format(val)

    return proposed_value


def make_edit(param_name, proposed_value, hp_lines):
    """Produce old_line -> new_line edit from param + proposed value."""
    old_line = None
    for l in hp_lines:
        if l.startswith(param_name + " = ") or l.startswith(param_name + " ="):
            old_line = l
            break
    if old_line is None:
        return None

    eq_pos = old_line.index(" = ")
    comment_pos = old_line.find(" # ", eq_pos)
    if comment_pos == -1:
        cur_val_str = old_line[eq_pos + 3:].strip()
        comment_part = ""
    else:
        cur_val_str = old_line[eq_pos + 3:comment_pos].strip()
        comment_part = old_line[comment_pos:]

    safe_value = clamp_value(param_name, proposed_value)
    new_line = old_line[:eq_pos + 3] + safe_value + comment_part

    if new_line == old_line:
        return None

    return {
        "desc": f"{param_name}={safe_value} (was {cur_val_str})",
        "old": old_line,
        "new": new_line,
        "param": param_name,
        "value": safe_value,
    }


# ---------------------------------------------------------------------------
# OOM pre-flight check (new in V4)
# ---------------------------------------------------------------------------

def would_oom(hp_dict, max_score=3_000_000_000):
    """Rule-based OOM check. Returns True if config would likely OOM on 24GB."""
    d = int(hp_dict.get("DEPTH", 4))
    ar = int(hp_dict.get("ASPECT_RATIO", 48))
    bs_str = str(hp_dict.get("TOTAL_BATCH_SIZE", "2**18"))
    bs_val = int(eval(bs_str)) if "**" in bs_str else int(bs_str)
    score = d * ar * d * bs_val
    return score > max_score


def get_current_hp_dict(hp_lines):
    """Parse HP lines into dict."""
    hp_dict = {}
    for l in hp_lines:
        m = re.match(r'^(\w+)\s*=\s*(.+?)(?:\s*#.*)?$', l)
        if m:
            hp_dict[m.group(1)] = m.group(2).strip()
    return hp_dict


# ---------------------------------------------------------------------------
# Relevance scoring (new in V4)
# ---------------------------------------------------------------------------

def score_relevance_batch(papers):
    """Score relevance of all papers in a single LLM call.
    Adapted from Claude Code's batch tool execution pattern — concurrent
    read-only operations grouped into a single request.
    Returns list of (score, paper) tuples sorted by relevance.
    """
    if not papers:
        return []

    task_context = (
        "Tuning hyperparameters (learning rate, optimizer settings, schedule, "
        "architecture ratios) for a small GPT language model (~124M params) "
        "trained on text data. GPU: RTX 5090, 24GB VRAM, 5-min training budget."
    )

    # Build one prompt with all papers numbered
    paper_block = ""
    for i, p in enumerate(papers, 1):
        title = p.get("title", "untitled")[:100]
        abstract = p.get("abstract", "")[:300]
        paper_block += f"\n[{i}] {title}\n    {abstract}\n"

    prompt = f"""Rate how relevant each paper is to our task. Be strict.

OUR TASK: {task_context}

PAPERS:
{paper_block}

Rate each from 0 to 10:
- 0: Completely unrelated (biology, vision, etc.)
- 5: Somewhat related (transformers but different scale/task)
- 10: Directly applicable (specific HP values for ~100M param GPT)

Output one line per paper: [number] score
Example:
[1] 7
[2] 3
"""

    response = call_llm(prompt, max_tokens=len(papers) * 10 + 50, temperature=0.0)

    # Parse scores
    scores = {}
    for m in re.finditer(r'\[(\d+)\]\s*(\d+(?:\.\d+)?)', response):
        idx = int(m.group(1))
        score = max(0.0, min(1.0, float(m.group(2)) / 10.0))
        scores[idx] = score

    # Build scored list, default 0.5 for unparsed
    scored = []
    for i, p in enumerate(papers, 1):
        s = scores.get(i, 0.5)
        scored.append((s, p))
        title_short = p.get('title', '')[:60]
        print(f"    {s:.2f} | {title_short}")

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


_llama_proc = None
_current_model = None


# ---------------------------------------------------------------------------
# File state cache (adapted from Claude Code's FileStateCache)
# ---------------------------------------------------------------------------

class FileStateCache:
    """LRU cache for file reads with timestamp tracking.
    Adapted from Claude Code's fileStateCache.ts — prevents redundant disk
    reads when train.py is read multiple times per research cycle.
    """
    def __init__(self, max_entries=32):
        self._cache = {}  # path -> (content, mtime, read_time)
        self._max = max_entries
        self._access_order = []

    def get(self, path):
        """Return cached content if file hasn't changed on disk."""
        path = os.path.normpath(path)
        if path not in self._cache:
            return None
        content, cached_mtime, _ = self._cache[path]
        try:
            current_mtime = os.path.getmtime(path)
            if current_mtime != cached_mtime:
                del self._cache[path]
                return None  # stale
        except OSError:
            return None
        # LRU touch
        if path in self._access_order:
            self._access_order.remove(path)
        self._access_order.append(path)
        return content

    def set(self, path, content):
        path = os.path.normpath(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        self._cache[path] = (content, mtime, time.time())
        if path in self._access_order:
            self._access_order.remove(path)
        self._access_order.append(path)
        # Evict LRU
        while len(self._cache) > self._max:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

    def invalidate(self, path):
        path = os.path.normpath(path)
        self._cache.pop(path, None)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()


_file_cache = FileStateCache()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_file(path):
    cached = _file_cache.get(path)
    if cached is not None:
        return cached
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _file_cache.set(path, content)
    return content

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    _file_cache.invalidate(path)  # bust cache on write

def get_hyperparams_block():
    content = read_file("train.py")
    hp_lines = [l for l in content.splitlines()
                if re.match(r'^[A-Z_]+ = ', l)][:20]
    return hp_lines

def format_experiment_history(results_history, tried_descs):
    """Format experiment history with auto-compaction for long histories.

    Adapted from Claude Code's context compaction: when history exceeds a
    threshold, older experiments are summarized into a structured digest
    (keeps, parameter patterns, failure modes) so the LLM sees actionable
    context rather than a raw log that wastes tokens.
    """
    if not results_history:
        return "No experiments run yet. Starting from karpathy baseline (val_bpb=1.379)."

    COMPACT_THRESHOLD = 15  # summarize history older than this many experiments

    if len(results_history) <= COMPACT_THRESHOLD:
        # Short history — show everything
        lines = []
        for i, (status, desc) in enumerate(zip(results_history, tried_descs), 1):
            lines.append(f"  {i}. [{status.upper()}] {desc}")
        return "\n".join(lines)

    # --- Compact old history into structured summary ---
    old_statuses = results_history[:-COMPACT_THRESHOLD]
    old_descs = tried_descs[:-COMPACT_THRESHOLD]

    keeps = [(i+1, d) for i, (s, d) in enumerate(zip(old_statuses, old_descs)) if s == "keep"]
    crashes = sum(1 for s in old_statuses if s == "crash")
    discards = sum(1 for s in old_statuses if s == "discard")
    skips = sum(1 for s in old_statuses if s == "skip")

    # Count per-parameter attempts in old history
    from collections import Counter
    param_counts = Counter()
    for d in old_descs:
        m = re.match(r"(\w+)=", d)
        if m:
            param_counts[m.group(1)] += 1

    summary_lines = [f"  [COMPACTED: experiments 1-{len(old_statuses)}]"]
    summary_lines.append(f"    {len(keeps)} keeps, {discards} discards, {crashes} crashes, {skips} skips")
    if keeps:
        summary_lines.append(f"    Improvements: {', '.join(d for _, d in keeps)}")
    if param_counts:
        top_params = param_counts.most_common(5)
        summary_lines.append(f"    Most tried: {', '.join(f'{p}({n}x)' for p, n in top_params)}")

    # Recent history — show in full
    recent_statuses = results_history[-COMPACT_THRESHOLD:]
    recent_descs = tried_descs[-COMPACT_THRESHOLD:]
    offset = len(old_statuses)
    for i, (status, desc) in enumerate(zip(recent_statuses, recent_descs), 1):
        summary_lines.append(f"  {offset + i}. [{status.upper()}] {desc}")

    return "\n".join(summary_lines)


def format_param_summary(results_history, tried_descs, best_bpb):
    """Build per-parameter summary showing what's been explored."""
    from collections import defaultdict
    param_tries = defaultdict(list)

    for status, desc in zip(results_history, tried_descs):
        # V4 format: "PARAM=value (was old_value)"
        m = re.match(r"(\w+)=", desc)
        if m:
            param_tries[m.group(1)].append(status)

    lines = ["Parameter exploration summary:"]
    for param in sorted(param_tries):
        tries = param_tries[param]
        keeps = sum(1 for s in tries if s == "keep")
        n = len(tries)
        hint = f" (WORKING: {keeps} keeps)" if keeps > 0 else f" ({n} tries, no keeps)"
        lines.append(f"  {param:20s}: {n} tries{hint}")

    untried = [p for p in ALL_PARAM_NAMES if p not in param_tries]
    if untried:
        lines.append(f"\n  UNEXPLORED (try these!): {', '.join(untried)}")

    return "\n".join(lines)


def get_cooled_params(results_history, tried_descs, cooldown=3):
    """Return set of param names tried recently without success."""
    recent_fails = set()
    recent_window = list(zip(results_history, tried_descs))[-cooldown:]
    for status, desc in recent_window:
        if status != "keep":
            m = re.match(r"(\w+)=", desc)
            if m:
                recent_fails.add(m.group(1))
    for status, desc in recent_window:
        if status == "keep":
            m = re.match(r"(\w+)=", desc)
            if m:
                recent_fails.discard(m.group(1))
    return recent_fails


# ---------------------------------------------------------------------------
# llama-server management
# ---------------------------------------------------------------------------

def start_llama_server(model_path, ctx_size=4096, threads=8, label="model",
                       enable_thinking=False):
    """Start llama-server with the given model.

    Args:
        enable_thinking: Enable Qwen's native thinking mode (/think tags).
            Adapted from Claude Code's adaptive thinking — lets the model
            reason through complex decisions before outputting a response.
            Used for high-stakes calls (proposal generation) but not for
            quick scoring or search planning.
    """
    global _llama_proc, _current_model

    # Build a config key that includes thinking mode so we reload if it changed
    config_key = f"{model_path}:ctx={ctx_size}:think={enable_thinking}"
    if _current_model == config_key and _llama_proc is not None:
        try:
            r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass

    stop_llama_server()

    think_str = "true" if enable_thinking else "false"
    print(f"  [Loading {label} (thinking={'on' if enable_thinking else 'off'})...]")
    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "-ngl", "99",
        "--port", str(LLAMA_PORT),
        "--ctx-size", str(ctx_size),
        "-t", str(threads),
        "--no-mmap",
        "--chat-template-kwargs", f'{{"enable_thinking":{think_str}}}',
    ]
    _llama_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _current_model = config_key

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
    time.sleep(2)
    print("  [server stopped, VRAM freed]")


def call_llm(prompt, max_tokens=512, temperature=0.7, stop=None):
    """Call llama-server with exponential backoff and error categorization.

    Adapted from Claude Code's withRetry pattern:
    - Classify errors as retryable vs terminal
    - Exponential backoff with jitter (500ms * 2^attempt, capped at 32s)
    - Max 5 retries for transient failures
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop

    MAX_RETRIES = 5
    BASE_DELAY = 0.5  # seconds

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{LLAMA_API_BASE}/v1/chat/completions",
                json=payload,
                timeout=180,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            # Server not ready or crashed — retryable
            if attempt >= MAX_RETRIES:
                raise
            delay = min(BASE_DELAY * (2 ** attempt), 32)
            jitter = random.random() * 0.25 * delay
            print(f"    [LLM connection error, retry {attempt+1}/{MAX_RETRIES} "
                  f"in {delay+jitter:.1f}s]")
            time.sleep(delay + jitter)

        except requests.exceptions.Timeout:
            # Timeout — retryable with longer patience
            if attempt >= MAX_RETRIES:
                raise
            delay = min(BASE_DELAY * (2 ** attempt), 32)
            print(f"    [LLM timeout, retry {attempt+1}/{MAX_RETRIES} in {delay:.1f}s]")
            time.sleep(delay)

        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, 'status_code', 0) if e.response is not None else 0
            # 429 rate limit, 503 overloaded — retryable
            if status in (429, 503, 529):
                if attempt >= MAX_RETRIES:
                    raise
                delay = min(BASE_DELAY * (2 ** attempt), 32)
                print(f"    [LLM {status}, retry {attempt+1}/{MAX_RETRIES} in {delay:.1f}s]")
                time.sleep(delay)
            else:
                # 400 bad request, 401 auth, etc. — terminal
                raise


# ---------------------------------------------------------------------------
# Stage 1: Orchestrator plans search query
# ---------------------------------------------------------------------------

def orchestrator_plan_search(hp_lines, results_history, tried_descs, best_bpb):
    start_llama_server(QWEN_MODEL, ctx_size=4096, label="Qwen3.5-9B orchestrator")

    hp_numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(hp_lines))
    history = format_experiment_history(results_history, tried_descs)

    prompt = f"""You are an ML research orchestrator tuning a small GPT language model (~124M params).
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.
GPU: NVIDIA RTX 5090, 24 GB VRAM. Training budget: 5 min/experiment.

Current hyperparameters in train.py:
{hp_numbered}

Experiment history (most recent last):
{history}

Your task: Decide what research direction to explore NEXT on arXiv/Scholar.
- Look at what has been tried and what worked (KEEP) vs failed (DISCARD/CRASH)
- DO NOT repeat directions that have already been tried multiple times
- Focus on optimizer settings, learning rates, schedules, and architecture ratios
- We can only change existing hyperparameters (no new code, no new layers)

Output EXACTLY two lines:
SEARCH: <a Semantic Scholar search query, 5-15 words, specific to the technique>
WHY: <one sentence explaining the reasoning>"""

    response = call_llm(prompt, max_tokens=200, temperature=0.7, stop=["\n\n\n"])
    m = re.search(r"SEARCH:\s*(.+)", response)
    if m:
        query = m.group(1).strip()
        print(f"  [Search query: {query}]")
        return query

    # Fallback
    fallback = random.choice(FALLBACK_TOPICS)
    print(f"  [Using fallback query: {fallback}]")
    return fallback


# ---------------------------------------------------------------------------
# Stage 2: Research session (Scholar API + MolmoWeb fallback)
# ---------------------------------------------------------------------------

MOLMOWEB_DEEP_THRESHOLD = 0.7  # Score >= 0.7 triggers MolmoWeb deep read
MOLMOWEB_MAX_PAPERS = 2        # Max papers to deep-read per session
MOLMOWEB_STEP_LIMIT = 5        # Max browsing steps per paper (fewer = less scraping)
API_RATE_DELAY = 3.0            # Seconds between API calls


def research_session(topic):
    """
    Tiered research pipeline:
      1. Scholar/arXiv API → find papers (fast, cheap)
      2. Qwen scores relevance on each abstract
      3. Low relevance (<0.5) → skip
      4. Medium (0.5-0.7) → use abstract summary only
      5. High (>=0.7) → MolmoWeb deep-reads that specific paper URL

    Returns combined findings text for the orchestrator.
    """
    from localpilot.browse import scholar_search, arxiv_search

    # --- Stage A: API discovery (Scholar + arXiv) ---
    print(f"\n  [Research] '{topic}'")
    papers = []

    try:
        s_papers = scholar_search(topic, max_results=5)
        papers.extend(s_papers)
        print(f"  [Scholar: {len(s_papers)} papers]")
    except Exception as e:
        print(f"  [Scholar error: {e}]")

    time.sleep(API_RATE_DELAY)

    try:
        a_papers = arxiv_search(topic, max_results=5)
        papers.extend(a_papers)
        print(f"  [arXiv: {len(a_papers)} papers]")
    except Exception as e:
        print(f"  [arXiv error: {e}]")

    if not papers:
        print("  [No papers found from any API]")
        return "No papers found."

    # Deduplicate by title similarity
    seen_titles = set()
    unique_papers = []
    for p in papers:
        title_key = p.get("title", "").lower().strip()[:60]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(p)
    papers = unique_papers
    print(f"  [Total unique: {len(papers)} papers]")

    return papers


def score_and_filter_papers(papers):
    """
    Score each paper's relevance with Qwen, then decide:
      - score < 0.5 → skip entirely
      - 0.5 <= score < 0.7 → keep abstract summary only
      - score >= 0.7 → flag for MolmoWeb deep reading

    Returns (filtered_text, deep_read_urls).
    """
    if not papers or isinstance(papers, str):
        return papers if isinstance(papers, str) else "No papers found.", []

    scored = score_relevance_batch(papers)

    # Partition by relevance tier
    deep_read = []   # score >= 0.7 → MolmoWeb
    summary = []     # 0.5 <= score < 0.7 → abstract only
    skipped = 0      # score < 0.5

    for score, paper in scored:
        if score >= MOLMOWEB_DEEP_THRESHOLD and len(deep_read) < MOLMOWEB_MAX_PAPERS:
            deep_read.append((score, paper))
        elif score >= 0.5:
            summary.append((score, paper))
        else:
            skipped += 1

    print(f"  [Relevance: {len(deep_read)} deep-read, {len(summary)} summary, {skipped} skipped]")

    # Build text from summary-tier papers
    filtered = ""
    if summary:
        filtered += "## Papers (abstract summary):\n\n"
        for i, (score, p) in enumerate(summary, 1):
            filtered += f"{i}. [{score:.1f}] {p.get('title', '')}"
            if p.get('year'):
                filtered += f" ({p['year']})"
            filtered += f"\n   {p.get('abstract', '')[:400]}\n\n"

    # Collect URLs for deep reading
    deep_urls = []
    for score, p in deep_read:
        url = p.get("url", "")
        if url:
            deep_urls.append({"url": url, "title": p.get("title", ""), "score": score})
        # Also include abstract in filtered text
        filtered += f"[HIGH RELEVANCE {score:.1f}] {p.get('title', '')}\n"
        filtered += f"  {p.get('abstract', '')[:500]}\n\n"

    if not filtered:
        filtered = "No sufficiently relevant papers found."

    return filtered, deep_urls


def molmoweb_deep_read(paper_info):
    """
    Use MolmoWeb to visually read a specific paper page.
    Only called for high-relevance papers (score >= 0.7).
    Uses minimal steps (5 max) to reduce network footprint.
    """
    url = paper_info["url"]
    title = paper_info["title"]
    print(f"  [MolmoWeb deep-read: {title[:60]}]")

    # Stop Qwen to free VRAM for MolmoWeb
    stop_llama_server()

    browse_task = (
        f"Go to {url} and read the paper details. "
        f"Extract: (1) specific hyperparameter values recommended, "
        f"(2) techniques for small transformer training, "
        f"(3) any learning rate, weight decay, or optimizer settings mentioned. "
        f"Signal completion with send_msg_to_user summarising the key findings."
    )

    try:
        result = subprocess.run(
            [PYTHON, "-m", "localpilot.browse", "browse", browse_task],
            capture_output=True, timeout=180,  # shorter timeout for single page
            cwd=str(ROOT),
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "MOLMOWEB_MAX_STEPS": str(MOLMOWEB_STEP_LIMIT),
            },
        )
        stdout = result.stdout.decode("utf-8", errors="replace") if isinstance(result.stdout, bytes) else result.stdout
        stderr = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else result.stderr
        output = (stdout + stderr).strip()

        is_error = ("Traceback" in output or "Error:" in output or result.returncode != 0)
        if not is_error and len(output) > 200:
            # Extract agent's visual observations
            thoughts = []
            for line in output.split("\n"):
                if "Thought:" in line:
                    t = line.split("Thought:", 1)[1].strip()
                    if len(t) > 20:
                        thoughts.append(t)

            deep_findings = f"## Deep read: {title}\n\n"
            if thoughts:
                deep_findings += "Visual observations:\n"
                for i, t in enumerate(thoughts[:5], 1):
                    deep_findings += f"  {i}. {t[:300]}\n"
                deep_findings += "\n"

            # Add page content
            page_sections = output.split("---")
            for section in page_sections[-2:]:  # last sections have most content
                cleaned = section.strip()
                if len(cleaned) > 100:
                    deep_findings += cleaned[:1500] + "\n\n"

            print(f"  [MolmoWeb: {len(deep_findings)} chars, {len(thoughts)} observations]")
            return deep_findings

        print(f"  [MolmoWeb: {'error' if is_error else 'too short'} ({len(output)} chars)]")
    except subprocess.TimeoutExpired:
        print("  [MolmoWeb: timeout]")
    except Exception as e:
        print(f"  [MolmoWeb error: {e}]")

    return ""


# ---------------------------------------------------------------------------
# Post-proposal validation (adapted from Claude Code's stop hooks)
# ---------------------------------------------------------------------------

def validate_proposal(param_name, proposed_value, results_history, tried_descs,
                      best_bpb):
    """Check if a proposal is obviously bad before wasting a training run.

    Adapted from Claude Code's stop hooks — run validation after the model
    proposes but before executing. Returns a rejection reason string, or
    None if the proposal looks okay.
    """
    # Rule 1: Exact duplicate — same param=value already tried
    proposed_key = f"{param_name}={proposed_value}"
    for desc in tried_descs:
        if desc.startswith(proposed_key):
            return "exact duplicate already tried"

    # Rule 2: Same value as current (no-op edit caught earlier, but double-check)
    # (handled by make_edit returning None)

    # Rule 3: Parameter tried 3+ times recently with all discards — likely exhausted
    recent_window = 8
    recent_tries = 0
    recent_keeps = 0
    for status, desc in zip(
        results_history[-recent_window:], tried_descs[-recent_window:]
    ):
        if desc.startswith(f"{param_name}="):
            recent_tries += 1
            if status == "keep":
                recent_keeps += 1
    if recent_tries >= 3 and recent_keeps == 0:
        return f"tried {recent_tries}x recently with 0 keeps"

    return None


# ---------------------------------------------------------------------------
# Stage 4: Orchestrator proposes PARAM + VALUE (V4: open values)
# ---------------------------------------------------------------------------

def orchestrator_propose(hp_lines, paper_ideas, results_history,
                         tried_descs, best_bpb, n=PATCHES_PER_SESSION):
    """
    Qwen3.5-9B reviews findings + history, proposes PARAM + exact VALUE.
    Values are clamped to safe bounds after proposal.
    """
    start_llama_server(QWEN_MODEL, ctx_size=8192, label="Qwen3.5-9B orchestrator",
                       enable_thinking=True)

    hp_numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(hp_lines))
    history = format_experiment_history(results_history, tried_descs)
    param_summary = format_param_summary(results_history, tried_descs, best_bpb)

    cooled = get_cooled_params(results_history, tried_descs, cooldown=3)

    # Build bounds description for the LLM
    bounds_desc = "Parameter bounds (values will be clamped to these):\n"
    for name, spec in SAFE_CONTINUOUS.items():
        bounds_desc += f"  {name}: [{spec['min']}, {spec['max']}]\n"
    for name, opts in SAFE_DISCRETE.items():
        bounds_desc += f"  {name}: one of {opts}\n"
    bounds_desc += f"  ADAM_BETAS: (beta1 in [{SAFE_ADAM_BETAS['beta1']['min']}, {SAFE_ADAM_BETAS['beta1']['max']}], beta2 in [{SAFE_ADAM_BETAS['beta2']['min']}, {SAFE_ADAM_BETAS['beta2']['max']}])\n"

    proposals = []
    hp_dict = get_current_hp_dict(hp_lines)

    for attempt in range(n * 3):
        if len(proposals) >= n:
            break

        batch_params = {p["param"] for p in proposals}
        available = [p for p in ALL_PARAM_NAMES
                     if p not in cooled and p not in batch_params]
        if not available:
            available = [p for p in ALL_PARAM_NAMES if p not in batch_params]

        avail_str = "\n".join(f"  - {name}" for name in available)

        prompt = f"""You are an ML research orchestrator tuning a GPT language model.
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.

GPU: NVIDIA RTX 5090, 24 GB VRAM. Training budget: 5 min/experiment.
This is a small GPT (~124M params). Focus on optimizer, LR, and schedule tuning.

Current hyperparameters:
{hp_numbered}

{param_summary}

{bounds_desc}

Parameters you can choose from (others are on cooldown):
{avail_str}

Research findings:
{paper_ideas[:2000]}

Pick ONE parameter and propose an EXACT new value.
- You can set ANY value within the bounds (not just +/- steps)
- If research suggests a specific value, propose it directly
- Prefer UNEXPLORED parameters
- For ADAM_BETAS, format as (beta1, beta2)

Output EXACTLY 3 lines:
PARAM: <parameter name from the list above>
VALUE: <the exact new value>
REASON: <one sentence citing research or experimental evidence>
"""

        try:
            response = call_llm(prompt, max_tokens=200, temperature=0.9,
                                stop=["\n\n\n"])
            param_m = re.search(r"PARAM:\s*(\w+)", response)
            value_m = re.search(r"VALUE:\s*(.+)", response)
            reason_m = re.search(r"REASON:\s*(.+)", response)

            if not all([param_m, value_m]):
                continue

            param_name = param_m.group(1).strip()
            raw_value = value_m.group(1).strip()
            reason = reason_m.group(1).strip() if reason_m else "no reason given"

            if param_name not in available:
                continue
            if param_name in batch_params:
                continue

            # Clamp to safe bounds
            safe_value = clamp_value(param_name, raw_value)

            # Produce edit
            edit = make_edit(param_name, safe_value, hp_lines)
            if edit is None:
                continue

            # OOM pre-flight check
            test_hp = dict(hp_dict)
            test_hp[param_name] = safe_value
            if would_oom(test_hp):
                print(f"    [BLOCKED OOM: {param_name}={safe_value}]")
                continue

            edit["reason"] = reason

            # --- Post-proposal validation (adapted from Claude Code stop hooks) ---
            # Before accepting, do a quick sanity check: is this change
            # contradicted by our own experiment history?
            rejection = validate_proposal(param_name, safe_value, results_history,
                                          tried_descs, best_bpb)
            if rejection:
                print(f"    [REJECTED: {param_name}={safe_value} — {rejection}]")
                continue

            proposals.append(edit)
            print(f"    proposal {len(proposals)}: {edit['desc']} | {reason[:50]}")

        except Exception as e:
            print(f"    [orchestrator propose error: {e}]")
            time.sleep(1)

    return proposals


# ---------------------------------------------------------------------------
# Stage 5: Experiment runner (same keep/discard protocol as baseline)
# ---------------------------------------------------------------------------

def run_experiment(desc, old_line, new_line, best_bpb, exp_num):
    """Apply a patch, train, keep/discard."""
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
    subprocess.run(["git", "commit", "-m", f"enhanced-v4: {desc}"],
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
        "condition": "enhanced_v4",
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
# Stopping logic with diminishing returns detection
# (adapted from Claude Code's tokenBudget.ts diminishing threshold)
# ---------------------------------------------------------------------------

def detect_diminishing_returns(results_history, tried_descs, best_bpb,
                                window=10, min_experiments=20):
    """Detect if improvements have stagnated.
    Adapted from Claude Code's checkTokenBudget diminishing returns:
    track deltas between checks, flag if 3+ windows show no progress.
    Returns (is_diminishing, reason_str).
    """
    if len(results_history) < min_experiments:
        return False, ""

    recent = list(zip(results_history[-window:], tried_descs[-window:]))
    recent_keeps = sum(1 for s, _ in recent if s == "keep")

    # Check if any recent keep actually improved BPB meaningfully
    # (not just noise — threshold: 0.0001 BPB)
    if recent_keeps == 0:
        return False, ""  # no keeps = handled by consec_discards

    # Look at the magnitude of recent improvements
    # If all keeps in recent window improved by < 0.0005, we're in diminishing territory
    # (This is a heuristic — real measurement needs BPB values which we don't store in history)
    return False, ""


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


def _count_tail_discards(history):
    count = 0
    for s in reversed(history):
        if s in ("discard", "crash", "skip"):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Error categorization & fallback (adapted from Claude Code's withRetry.ts)
# ---------------------------------------------------------------------------

def _categorize_error(error):
    """Classify error as retryable vs terminal, with a human-readable label.
    Adapted from Claude Code's shouldRetry error classification.
    """
    err_str = str(error).lower()
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection_error"  # retryable: server crashed or not started
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout"  # retryable: model took too long
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(error.response, 'status_code', 0) if error.response is not None else 0
        if status in (429, 503, 529):
            return f"overloaded_{status}"  # retryable: rate limit or overload
        if status >= 500:
            return f"server_error_{status}"  # retryable: server bug
        return f"client_error_{status}"  # likely terminal
    if "oom" in err_str or "out of memory" in err_str:
        return "oom"  # terminal: need smaller model or config
    if isinstance(error, subprocess.TimeoutExpired):
        return "subprocess_timeout"  # retryable
    return "unknown"


def _random_fallback_proposals(hp_lines, hp_dict, results_history, tried_descs,
                               n=PATCHES_PER_SESSION):
    """Generate random proposals when research pipeline fails.
    Circuit-breaker fallback: ensures the experiment loop keeps running
    even when Scholar/arXiv/MolmoWeb/Qwen are all broken.
    """
    proposals = []
    all_params = list(SAFE_CONTINUOUS.keys()) + list(SAFE_DISCRETE.keys()) + ["ADAM_BETAS"]

    for _ in range(n * 3):
        if len(proposals) >= n:
            break

        param = random.choice(all_params)
        batch_params = {p["param"] for p in proposals}
        if param in batch_params:
            continue

        # Generate random value within bounds
        if param in SAFE_CONTINUOUS:
            spec = SAFE_CONTINUOUS[param]
            val = random.uniform(spec["min"], spec["max"])
            safe_val = spec["fmt"].format(val)
        elif param in SAFE_DISCRETE:
            safe_val = str(random.choice(SAFE_DISCRETE[param]))
        elif param == "ADAM_BETAS":
            b1 = random.uniform(SAFE_ADAM_BETAS["beta1"]["min"],
                                SAFE_ADAM_BETAS["beta1"]["max"])
            b2 = random.uniform(SAFE_ADAM_BETAS["beta2"]["min"],
                                SAFE_ADAM_BETAS["beta2"]["max"])
            safe_val = f"({round(b1, 3)}, {round(b2, 3)})"
        else:
            continue

        edit = make_edit(param, safe_val, hp_lines)
        if edit is None:
            continue

        # OOM check
        test_hp = dict(hp_dict)
        test_hp[param] = safe_val
        if would_oom(test_hp):
            continue

        edit["reason"] = "random fallback (research pipeline failed)"
        proposals.append(edit)

    print(f"  [Random fallback: {len(proposals)} proposals]")
    return proposals


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LocalPilot -- Enhanced v4 (Condition C final)")
    print("  Qwen3.5-9B + MolmoWeb + relevance scoring + open values")
    print("=" * 60)

    # Check models exist
    if not os.path.exists(QWEN_MODEL):
        print(f"\nERROR: Qwen model not found at {QWEN_MODEL}")
        sys.exit(1)

    # Initialize results file
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\twall_seconds\n")

    if not RESEARCH_LOG.exists():
        RESEARCH_LOG.write_text(
            "# LocalPilot v4 Research Log\n\nQwen3.5-9B orchestrated research with relevance scoring.\n",
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

    # --- Graceful interrupt (adapted from Claude Code's AbortController) ---
    # Two-stage: first Ctrl+C saves state + finishes current experiment,
    # second Ctrl+C force-quits.
    _interrupted = False
    _interrupt_count = 0

    def _handle_interrupt(signum, frame):
        nonlocal _interrupted, _interrupt_count
        _interrupt_count += 1
        if _interrupt_count == 1:
            _interrupted = True
            print("\n\n  [Ctrl+C] Finishing current experiment, then saving state...")
            print("  [Ctrl+C again to force-quit]")
        else:
            print("\n  [Force quit]")
            stop_llama_server()
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_interrupt)

    # --- State machine loop (adapted from Claude Code's query.ts) ---
    # Each iteration has an explicit transition reason so failures are
    # diagnosed, not blindly retried.
    consecutive_research_failures = 0
    MAX_RESEARCH_FAILURES = 3  # circuit breaker (from Claude Code's autocompact)

    try:
        while True:
            total_done = len(results_history)
            transition = None  # track why we continue (for debugging)

            # Check interrupt signal (from Claude Code's AbortController)
            if _interrupted:
                reason = "user_interrupt (Ctrl+C)"
                print(f"\n{'='*60}")
                print(f"INTERRUPTED: saving state")
                QUEUE_FILE.write_text(
                    json.dumps(queue, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  Queue saved: {len(queue)} patches preserved for resume")
                # Fall through to normal stop logic below

            stop, reason = should_stop(results_history, total_done)
            if _interrupted or stop:
                print(f"\n{'='*60}")
                print(f"STOPPING: {reason}")
                print(f"Total experiments: {total_done}")
                print(f"Best val_bpb: {best_bpb:.6f}")
                print(f"Total wall time: {time.time() - run_start:.0f}s")
                print(f"Results: {RESULTS_FILE}")
                print(f"{'='*60}")

                with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "condition": "enhanced_v4",
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
                try:
                    hp_lines = get_hyperparams_block()

                    # Phase 1: Orchestrator plans search
                    search_query = orchestrator_plan_search(
                        hp_lines, results_history, tried_descs, best_bpb
                    )

                    # Phase 2: API discovery (Scholar + arXiv, fast)
                    papers = research_session(search_query)

                    # Phase 3: Relevance scoring + tiered filtering
                    start_llama_server(QWEN_MODEL, ctx_size=4096,
                                       label="Qwen3.5-9B scorer")
                    if isinstance(papers, list):
                        filtered_ideas, deep_urls = score_and_filter_papers(papers)
                    else:
                        filtered_ideas, deep_urls = papers, []

                    # Phase 3b: MolmoWeb deep-reads highly relevant papers
                    deep_findings = ""
                    for paper_info in deep_urls[:MOLMOWEB_MAX_PAPERS]:
                        finding = molmoweb_deep_read(paper_info)
                        if finding:
                            deep_findings += finding + "\n"
                        time.sleep(API_RATE_DELAY)

                    if deep_findings:
                        filtered_ideas += "\n" + deep_findings

                    log_research(search_query, filtered_ideas)
                    print(f"  [Research complete: {len(filtered_ideas)} chars"
                          f" ({len(deep_urls)} deep-reads)]")

                    # Phase 4: Orchestrator proposes with open values
                    new_proposals = orchestrator_propose(
                        hp_lines, filtered_ideas, results_history,
                        tried_descs, best_bpb, n=PATCHES_PER_SESSION,
                    )

                    stop_llama_server()

                    queue.extend(new_proposals)
                    if new_proposals:
                        consecutive_research_failures = 0  # reset only on actual success

                    QUEUE_FILE.write_text(
                        json.dumps(queue, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"\n  Queue refilled: {len(queue)} patches ready")

                    if not queue:
                        consecutive_research_failures += 1
                        transition = "empty_proposals"
                        print(f"  [No valid patches generated "
                              f"({consecutive_research_failures}/{MAX_RESEARCH_FAILURES})]")
                        if consecutive_research_failures >= MAX_RESEARCH_FAILURES:
                            print("  [Circuit breaker: falling back to random]")
                            queue.extend(_random_fallback_proposals(
                                hp_lines,
                                get_current_hp_dict(hp_lines),
                                results_history, tried_descs,
                            ))
                            consecutive_research_failures = 0
                        else:
                            time.sleep(5)
                        continue

                except Exception as e:
                    # --- Error recovery (adapted from Claude Code's withRetry) ---
                    # Categorize the failure and decide recovery strategy.
                    consecutive_research_failures += 1
                    error_type = _categorize_error(e)

                    print(f"\n  [Research phase FAILED: {error_type} — {e}]")
                    print(f"  [Consecutive failures: {consecutive_research_failures}"
                          f"/{MAX_RESEARCH_FAILURES}]")

                    # Circuit breaker: too many consecutive research failures
                    if consecutive_research_failures >= MAX_RESEARCH_FAILURES:
                        print(f"  [Circuit breaker tripped — falling back to "
                              f"random proposals]")
                        queue.extend(_random_fallback_proposals(
                            get_hyperparams_block(),
                            get_current_hp_dict(get_hyperparams_block()),
                            results_history, tried_descs,
                        ))
                        consecutive_research_failures = 0
                    else:
                        # Retryable — backoff and try again
                        delay = min(5 * (2 ** (consecutive_research_failures - 1)), 60)
                        print(f"  [Retrying in {delay}s...]")
                        stop_llama_server()
                        time.sleep(delay)

                    transition = f"research_error:{error_type}"
                    if not queue:
                        continue

            # ── Run next experiment from queue ─────────────────────────
            patch = queue.pop(0)
            QUEUE_FILE.write_text(
                json.dumps(queue, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            tried_descs.append(patch["desc"])

            exp_num = total_done + 1
            best_bpb, val_bpb, status = run_experiment(
                patch["desc"], patch["old"], patch["new"], best_bpb, exp_num
            )
            results_history.append(status)

            keeps = sum(1 for s in results_history if s == "keep")
            consec = _count_tail_discards(results_history)
            print(f"\n  [{exp_num}] best={best_bpb:.6f}  keeps={keeps}  "
                  f"consec_discards={consec}")

    finally:
        stop_llama_server()

    print("\nEnhanced v4 complete.")


if __name__ == "__main__":
    main()
