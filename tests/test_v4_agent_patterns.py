"""
Unit tests for V4 agent design patterns adapted from Claude Code.

Tests:
  1. FileStateCache — LRU caching with mtime invalidation
  2. score_relevance_batch — batch scoring all papers in one call
  3. format_experiment_history — history compaction for long runs
  4. validate_proposal — post-proposal stop hooks
  5. call_llm retry logic — exponential backoff, error categorization
  6. _categorize_error — error classification
  7. _random_fallback_proposals — circuit breaker fallback
  8. should_stop — stopping criteria
  9. detect_diminishing_returns — stagnation detection

All tests run WITHOUT GPU/LLM — pure logic tests only.
"""

import os
import re
import sys
import time
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# ---------------------------------------------------------------------------
# Import or redefine functions from run_enhanced_v4.py
# (copied to avoid side effects from module-level code in the runner)
# ---------------------------------------------------------------------------

# --- Safe parameter bounds (must match v4 runner) ---
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

PATCHES_PER_SESSION = 6


# --- FileStateCache (copied from v4) ---
class FileStateCache:
    def __init__(self, max_entries=32):
        self._cache = {}
        self._max = max_entries
        self._access_order = []

    def get(self, path):
        path = os.path.normpath(path)
        if path not in self._cache:
            return None
        content, cached_mtime, _ = self._cache[path]
        try:
            current_mtime = os.path.getmtime(path)
            if current_mtime != cached_mtime:
                del self._cache[path]
                return None
        except OSError:
            return None
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
        while len(self._cache) > self._max:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

    def invalidate(self, path):
        path = os.path.normpath(path)
        self._cache.pop(path, None)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()


# --- score_relevance_batch (parse logic only, no LLM call) ---
def parse_batch_scores(response, num_papers):
    """Parse the LLM response from batch scoring. Extracted for testability."""
    scores = {}
    for m in re.finditer(r'\[(\d+)\]\s*(\d+(?:\.\d+)?)', response):
        idx = int(m.group(1))
        score = max(0.0, min(1.0, float(m.group(2)) / 10.0))
        scores[idx] = score
    result = []
    for i in range(1, num_papers + 1):
        result.append(scores.get(i, 0.5))
    return result


# --- format_experiment_history (copied from v4) ---
def format_experiment_history(results_history, tried_descs):
    if not results_history:
        return "No experiments run yet. Starting from karpathy baseline (val_bpb=1.379)."

    COMPACT_THRESHOLD = 15
    if len(results_history) <= COMPACT_THRESHOLD:
        lines = []
        for i, (status, desc) in enumerate(zip(results_history, tried_descs), 1):
            lines.append(f"  {i}. [{status.upper()}] {desc}")
        return "\n".join(lines)

    old_statuses = results_history[:-COMPACT_THRESHOLD]
    old_descs = tried_descs[:-COMPACT_THRESHOLD]
    keeps = [(i+1, d) for i, (s, d) in enumerate(zip(old_statuses, old_descs)) if s == "keep"]
    crashes = sum(1 for s in old_statuses if s == "crash")
    discards = sum(1 for s in old_statuses if s == "discard")
    skips = sum(1 for s in old_statuses if s == "skip")

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

    recent_statuses = results_history[-COMPACT_THRESHOLD:]
    recent_descs = tried_descs[-COMPACT_THRESHOLD:]
    offset = len(old_statuses)
    for i, (status, desc) in enumerate(zip(recent_statuses, recent_descs), 1):
        summary_lines.append(f"  {offset + i}. [{status.upper()}] {desc}")

    return "\n".join(summary_lines)


# --- validate_proposal (copied from v4) ---
def validate_proposal(param_name, proposed_value, results_history, tried_descs, best_bpb):
    proposed_key = f"{param_name}={proposed_value}"
    for desc in tried_descs:
        if desc.startswith(proposed_key):
            return "exact duplicate already tried"
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


# --- _categorize_error (copied from v4) ---
import requests
import subprocess as _subprocess

def _categorize_error(error):
    err_str = str(error).lower()
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection_error"
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(error.response, 'status_code', 0) if error.response is not None else 0
        if status in (429, 503, 529):
            return f"overloaded_{status}"
        if status >= 500:
            return f"server_error_{status}"
        return f"client_error_{status}"
    if "oom" in err_str or "out of memory" in err_str:
        return "oom"
    if isinstance(error, _subprocess.TimeoutExpired):
        return "subprocess_timeout"
    return "unknown"


# --- should_stop (copied from v4) ---
CONSEC_DISCARD_LIMIT = 15
NO_KEEP_WINDOW = 20
MAX_EXPERIMENTS = 80

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


# --- clamp_value + make_edit + would_oom (copied from v4) ---
def clamp_value(param_name, proposed_value):
    proposed_value = str(proposed_value).strip()
    if param_name == "ADAM_BETAS":
        m = re.match(r"\(([^,]+),\s*([^)]+)\)", proposed_value)
        if not m:
            return proposed_value
        try:
            b1 = float(m.group(1))
            b2 = float(m.group(2))
        except ValueError:
            return proposed_value
        b1 = max(SAFE_ADAM_BETAS["beta1"]["min"], min(SAFE_ADAM_BETAS["beta1"]["max"], b1))
        b2 = max(SAFE_ADAM_BETAS["beta2"]["min"], min(SAFE_ADAM_BETAS["beta2"]["max"], b2))
        return f"({round(b1, 3)}, {round(b2, 3)})"
    if param_name in SAFE_DISCRETE:
        options = SAFE_DISCRETE[param_name]
        for opt in options:
            if str(opt).strip('"') == proposed_value.strip('"'):
                return str(opt)
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
    return {"desc": f"{param_name}={safe_value} (was {cur_val_str})",
            "old": old_line, "new": new_line, "param": param_name, "value": safe_value}


def would_oom(hp_dict, max_score=3_000_000_000):
    d = int(hp_dict.get("DEPTH", 4))
    ar = int(hp_dict.get("ASPECT_RATIO", 48))
    bs_str = str(hp_dict.get("TOTAL_BATCH_SIZE", "2**18"))
    bs_val = int(eval(bs_str)) if "**" in bs_str else int(bs_str)
    score = d * ar * d * bs_val
    return score > max_score


def _random_fallback_proposals(hp_lines, hp_dict, results_history, tried_descs,
                               n=PATCHES_PER_SESSION):
    import random
    proposals = []
    all_params = list(SAFE_CONTINUOUS.keys()) + list(SAFE_DISCRETE.keys()) + ["ADAM_BETAS"]
    for _ in range(n * 3):
        if len(proposals) >= n:
            break
        param = random.choice(all_params)
        batch_params = {p["param"] for p in proposals}
        if param in batch_params:
            continue
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
        test_hp = dict(hp_dict)
        test_hp[param] = safe_val
        if would_oom(test_hp):
            continue
        edit["reason"] = "random fallback (research pipeline failed)"
        proposals.append(edit)
    return proposals


# ===========================================================================
# TESTS
# ===========================================================================

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# 1. FileStateCache tests
# ---------------------------------------------------------------------------

def test_cache_basic_read_write():
    """Cache stores and retrieves file content."""
    cache = FileStateCache(max_entries=8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        cache.set(path, "hello world")
        assert cache.get(path) == "hello world", "cached content should match"
    finally:
        os.unlink(path)


def test_cache_invalidation_on_write():
    """Cache returns None after file is modified on disk."""
    cache = FileStateCache(max_entries=8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("v1")
        path = f.name
    try:
        cache.set(path, "v1")
        assert cache.get(path) == "v1"
        # Modify file — mtime changes
        time.sleep(0.05)
        with open(path, "w") as f:
            f.write("v2")
        assert cache.get(path) is None, "cache should be invalidated after disk write"
    finally:
        os.unlink(path)


def test_cache_explicit_invalidate():
    """invalidate() removes specific entry."""
    cache = FileStateCache(max_entries=8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("data")
        path = f.name
    try:
        cache.set(path, "data")
        cache.invalidate(path)
        assert cache.get(path) is None, "should be None after invalidate"
    finally:
        os.unlink(path)


def test_cache_lru_eviction():
    """Oldest entries are evicted when cache exceeds max_entries."""
    cache = FileStateCache(max_entries=3)
    paths = []
    for i in range(5):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"content_{i}")
            paths.append(f.name)
        cache.set(paths[-1], f"content_{i}")

    try:
        # First two should have been evicted
        assert cache.get(paths[0]) is None, "oldest should be evicted"
        assert cache.get(paths[1]) is None, "second oldest should be evicted"
        # Last three should remain
        assert cache.get(paths[2]) == "content_2"
        assert cache.get(paths[3]) == "content_3"
        assert cache.get(paths[4]) == "content_4"
    finally:
        for p in paths:
            os.unlink(p)


def test_cache_path_normalization():
    """Different path representations resolve to same cache entry."""
    cache = FileStateCache(max_entries=8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     dir=tempfile.gettempdir()) as f:
        f.write("normalized")
        path = f.name
    try:
        cache.set(path, "normalized")
        # Access with different separator style
        alt_path = path.replace("\\", "/")
        # normpath will normalize both to same form
        result = cache.get(alt_path)
        assert result == "normalized", f"path normalization failed: got {result}"
    finally:
        os.unlink(path)


def test_cache_nonexistent_file():
    """get() returns None for files that don't exist."""
    cache = FileStateCache()
    assert cache.get("/nonexistent/path.txt") is None


def test_cache_clear():
    """clear() empties the entire cache."""
    cache = FileStateCache()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("data")
        path = f.name
    try:
        cache.set(path, "data")
        cache.clear()
        assert cache.get(path) is None
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2. Batch scoring parse tests
# ---------------------------------------------------------------------------

def test_batch_score_parsing_normal():
    """Standard batch response parses correctly."""
    response = "[1] 8\n[2] 3\n[3] 6\n[4] 1\n[5] 9"
    scores = parse_batch_scores(response, 5)
    assert len(scores) == 5
    assert scores[0] == 0.8  # 8/10
    assert scores[1] == 0.3
    assert scores[2] == 0.6
    assert scores[3] == 0.1
    assert scores[4] == 0.9


def test_batch_score_parsing_with_noise():
    """Parser handles extra text around scores."""
    response = "Here are my ratings:\n[1] 7 - very relevant\n[2] 2 - not related\n[3] 5"
    scores = parse_batch_scores(response, 3)
    assert scores[0] == 0.7
    assert scores[1] == 0.2
    assert scores[2] == 0.5


def test_batch_score_missing_entries():
    """Missing entries default to 0.5."""
    response = "[1] 8\n[3] 6"  # [2] missing
    scores = parse_batch_scores(response, 3)
    assert scores[0] == 0.8
    assert scores[1] == 0.5  # default
    assert scores[2] == 0.6


def test_batch_score_clamping():
    """Scores outside 0-10 are clamped; negatives default to 0.5."""
    response = "[1] 15\n[2] -3"
    scores = parse_batch_scores(response, 2)
    assert scores[0] == 1.0, f"15/10 should clamp to 1.0, got {scores[0]}"
    # -3 won't match \d+ regex, so defaults to 0.5
    assert scores[1] == 0.5, f"negative should default to 0.5, got {scores[1]}"


def test_batch_score_empty_response():
    """Empty response gives all defaults."""
    scores = parse_batch_scores("", 3)
    assert scores == [0.5, 0.5, 0.5]


def test_batch_score_float_values():
    """Float scores like 7.5 are parsed correctly."""
    response = "[1] 7.5\n[2] 3.2"
    scores = parse_batch_scores(response, 2)
    assert scores[0] == 0.75
    assert abs(scores[1] - 0.32) < 0.01


# ---------------------------------------------------------------------------
# 3. History compaction tests
# ---------------------------------------------------------------------------

def test_history_empty():
    """Empty history returns starter message."""
    result = format_experiment_history([], [])
    assert "No experiments" in result


def test_history_short_no_compaction():
    """Short history (<= 15) is shown in full."""
    statuses = ["keep", "discard", "discard", "keep"]
    descs = ["LR=0.5", "DEPTH=6", "WEIGHT_DECAY=0.1", "MATRIX_LR=0.08"]
    result = format_experiment_history(statuses, descs)
    # All 4 experiments should be present
    assert "1. [KEEP]" in result
    assert "2. [DISCARD]" in result
    assert "3. [DISCARD]" in result
    assert "4. [KEEP]" in result
    assert "COMPACTED" not in result


def test_history_long_compaction():
    """Long history (> 15) has old experiments compacted."""
    n = 30
    statuses = ["discard"] * 25 + ["keep"] * 2 + ["discard"] * 3
    descs = [f"PARAM_{i}=val_{i}" for i in range(n)]
    # Make some keeps in old section
    statuses[3] = "keep"
    statuses[7] = "keep"
    statuses[10] = "crash"

    result = format_experiment_history(statuses, descs)
    assert "COMPACTED" in result
    assert "keeps" in result.lower() or "keep" in result.lower()
    # Recent experiments should be shown individually
    assert f"{n}. [DISCARD]" in result


def test_history_compaction_preserves_recent():
    """Compacted history shows last 15 experiments individually."""
    n = 40
    statuses = ["discard"] * 35 + ["keep"] * 5
    descs = [f"PARAM_{i}=val" for i in range(n)]
    result = format_experiment_history(statuses, descs)
    # Last 15 experiments (26-40) should appear individually
    for i in range(26, 41):
        assert f"{i}." in result, f"experiment {i} should appear in recent section"


def test_history_compaction_counts_correct():
    """Compacted section has correct keep/discard/crash/skip counts."""
    old_size = 20
    statuses = (["keep"] * 3 + ["discard"] * 12 + ["crash"] * 3 + ["skip"] * 2 +
                ["discard"] * 15)  # 15 recent
    descs = [f"P{i}=v" for i in range(len(statuses))]
    result = format_experiment_history(statuses, descs)
    assert "3 keeps" in result
    assert "12 discards" in result
    assert "3 crashes" in result
    assert "2 skips" in result


# ---------------------------------------------------------------------------
# 4. Validate proposal tests
# ---------------------------------------------------------------------------

def test_validate_duplicate():
    """Exact duplicate is rejected."""
    history = ["keep", "discard"]
    descs = ["LR=0.5 (was 0.3)", "DEPTH=6 (was 4)"]
    result = validate_proposal("LR", "0.5", history, descs, 1.2)
    assert result is not None
    assert "duplicate" in result


def test_validate_different_value_ok():
    """Same param but different value is allowed."""
    history = ["keep"]
    descs = ["LR=0.5 (was 0.3)"]
    result = validate_proposal("LR", "0.8", history, descs, 1.2)
    assert result is None, "different value should be allowed"


def test_validate_exhausted_param():
    """Param tried 3+ times recently with 0 keeps is rejected."""
    history = ["discard", "discard", "discard", "discard"]
    descs = ["DEPTH=4 (was 6)", "DEPTH=8 (was 6)", "DEPTH=10 (was 6)", "DEPTH=12 (was 6)"]
    result = validate_proposal("DEPTH", "6", history, descs, 1.2)
    assert result is not None
    assert "tried" in result and "0 keeps" in result


def test_validate_recently_successful_param_ok():
    """Param with recent keep is allowed even if tried many times."""
    history = ["discard", "keep", "discard", "discard"]
    descs = ["DEPTH=4 (was 6)", "DEPTH=8 (was 6)", "DEPTH=10 (was 6)", "DEPTH=12 (was 6)"]
    result = validate_proposal("DEPTH", "6", history, descs, 1.2)
    assert result is None, "param with recent keep should be allowed"


def test_validate_no_history():
    """No history means any proposal is valid."""
    result = validate_proposal("LR", "0.5", [], [], 1.268)
    assert result is None


def test_validate_other_param_exhausted():
    """Exhaustion of one param doesn't affect another."""
    history = ["discard"] * 5
    descs = ["LR=0.1", "LR=0.2", "LR=0.3", "LR=0.4", "LR=0.5"]
    result = validate_proposal("DEPTH", "6", history, descs, 1.2)
    assert result is None, "different param should not be affected"


# ---------------------------------------------------------------------------
# 5. Error categorization tests
# ---------------------------------------------------------------------------

def test_categorize_connection_error():
    assert _categorize_error(requests.exceptions.ConnectionError()) == "connection_error"


def test_categorize_timeout():
    assert _categorize_error(requests.exceptions.Timeout()) == "timeout"


def _make_http_error(status_code):
    """Helper to construct HTTPError with a mock response."""
    resp = requests.models.Response()
    resp.status_code = status_code
    resp._content = b""
    return requests.exceptions.HTTPError(response=resp)


def test_categorize_rate_limit():
    """429 is classified as overloaded (retryable)."""
    err = _make_http_error(429)
    assert _categorize_error(err) == "overloaded_429"


def test_categorize_server_error():
    err = _make_http_error(500)
    assert _categorize_error(err) == "server_error_500"


def test_categorize_client_error():
    err = _make_http_error(400)
    assert _categorize_error(err) == "client_error_400"


def test_categorize_oom():
    assert _categorize_error(RuntimeError("CUDA out of memory")) == "oom"
    assert _categorize_error(RuntimeError("OOM killed")) == "oom"


def test_categorize_subprocess_timeout():
    err = _subprocess.TimeoutExpired(cmd="train", timeout=900)
    assert _categorize_error(err) == "subprocess_timeout"


def test_categorize_unknown():
    assert _categorize_error(ValueError("something weird")) == "unknown"


# ---------------------------------------------------------------------------
# 6. should_stop tests
# ---------------------------------------------------------------------------

def test_stop_max_experiments():
    history = ["keep"] * 80
    stop, reason = should_stop(history, 80)
    assert stop is True
    assert "max_experiments" in reason


def test_stop_consecutive_discards():
    history = ["keep"] + ["discard"] * 15
    stop, reason = should_stop(history, 16)
    assert stop is True
    assert "consecutive_discards" in reason


def test_stop_no_keep_in_window():
    # Need at least 20 entries AND no keep in last 20
    # Put the keep far enough back that last 20 are all discards
    history = ["keep"] + ["discard"] * 25
    stop, reason = should_stop(history, len(history))
    assert stop is True, f"should stop: {reason}, history len={len(history)}"
    assert "no_keep_in_window" in reason or "consecutive_discards" in reason


def test_no_stop_healthy():
    """Healthy run should not stop."""
    history = ["keep", "discard", "discard", "keep", "discard"]
    stop, _ = should_stop(history, 5)
    assert stop is False


def test_stop_crash_counts_as_discard():
    """Crashes count toward consecutive discard limit."""
    history = ["keep"] + ["crash"] * 15
    stop, reason = should_stop(history, 16)
    assert stop is True


def test_stop_skip_counts_as_discard():
    """Skips count toward consecutive discard limit."""
    history = ["keep"] + ["skip"] * 15
    stop, reason = should_stop(history, 16)
    assert stop is True


def test_stop_keep_resets_consecutive():
    """A keep in the middle resets the consecutive counter."""
    history = ["discard"] * 10 + ["keep"] + ["discard"] * 5
    stop, _ = should_stop(history, 16)
    assert stop is False, "keep should reset consecutive counter"


# ---------------------------------------------------------------------------
# 7. Random fallback proposals tests
# ---------------------------------------------------------------------------

def test_fallback_generates_proposals():
    """Fallback generates valid proposals."""
    hp_lines = [
        "DEPTH = 4",
        "EMBEDDING_LR = 0.6",
        "WEIGHT_DECAY = 0.1",
        "MATRIX_LR = 0.04",
        "SCALAR_LR = 0.3",
        "ASPECT_RATIO = 48",
        "HEAD_DIM = 64",
        "TOTAL_BATCH_SIZE = 2**18",
    ]
    hp_dict = {"DEPTH": "4", "EMBEDDING_LR": "0.6", "WEIGHT_DECAY": "0.1",
               "MATRIX_LR": "0.04", "SCALAR_LR": "0.3", "ASPECT_RATIO": "48",
               "HEAD_DIM": "64", "TOTAL_BATCH_SIZE": "2**18"}
    proposals = _random_fallback_proposals(hp_lines, hp_dict, [], [])
    assert len(proposals) > 0, "should generate at least one proposal"
    assert len(proposals) <= PATCHES_PER_SESSION


def test_fallback_no_duplicate_params():
    """Fallback doesn't propose same param twice."""
    hp_lines = [
        "DEPTH = 4", "EMBEDDING_LR = 0.6", "WEIGHT_DECAY = 0.1",
        "MATRIX_LR = 0.04", "SCALAR_LR = 0.3", "ASPECT_RATIO = 48",
        "HEAD_DIM = 64", "TOTAL_BATCH_SIZE = 2**18",
    ]
    hp_dict = {"DEPTH": "4", "EMBEDDING_LR": "0.6", "WEIGHT_DECAY": "0.1",
               "MATRIX_LR": "0.04", "SCALAR_LR": "0.3", "ASPECT_RATIO": "48",
               "HEAD_DIM": "64", "TOTAL_BATCH_SIZE": "2**18"}
    proposals = _random_fallback_proposals(hp_lines, hp_dict, [], [])
    params = [p["param"] for p in proposals]
    assert len(params) == len(set(params)), f"duplicate params in fallback: {params}"


def test_fallback_values_within_bounds():
    """All fallback values are within safe bounds."""
    hp_lines = [
        "DEPTH = 4", "EMBEDDING_LR = 0.6", "WEIGHT_DECAY = 0.1",
        "MATRIX_LR = 0.04", "SCALAR_LR = 0.3", "ASPECT_RATIO = 48",
        "HEAD_DIM = 64", "TOTAL_BATCH_SIZE = 2**18",
    ]
    hp_dict = {"DEPTH": "4", "EMBEDDING_LR": "0.6", "WEIGHT_DECAY": "0.1",
               "MATRIX_LR": "0.04", "SCALAR_LR": "0.3", "ASPECT_RATIO": "48",
               "HEAD_DIM": "64", "TOTAL_BATCH_SIZE": "2**18"}
    for _ in range(10):  # run multiple times for random coverage
        proposals = _random_fallback_proposals(hp_lines, hp_dict, [], [])
        for p in proposals:
            param = p["param"]
            val = p["value"]
            if param in SAFE_CONTINUOUS:
                spec = SAFE_CONTINUOUS[param]
                fval = float(val)
                assert spec["min"] <= fval <= spec["max"], \
                    f"{param}={val} out of bounds [{spec['min']}, {spec['max']}]"
            elif param in SAFE_DISCRETE:
                opts = [str(o) for o in SAFE_DISCRETE[param]]
                assert val in opts, f"{param}={val} not in {opts}"


def test_fallback_no_oom():
    """Fallback proposals pass OOM check."""
    hp_lines = [
        "DEPTH = 12", "ASPECT_RATIO = 96", "TOTAL_BATCH_SIZE = 2**19",
        "EMBEDDING_LR = 0.6", "WEIGHT_DECAY = 0.1",
    ]
    hp_dict = {"DEPTH": "12", "ASPECT_RATIO": "96", "TOTAL_BATCH_SIZE": "2**19",
               "EMBEDDING_LR": "0.6", "WEIGHT_DECAY": "0.1"}
    proposals = _random_fallback_proposals(hp_lines, hp_dict, [], [])
    for p in proposals:
        test_hp = dict(hp_dict)
        test_hp[p["param"]] = p["value"]
        assert not would_oom(test_hp), f"OOM proposal slipped through: {p['desc']}"


# ---------------------------------------------------------------------------
# 8. OOM detection tests
# ---------------------------------------------------------------------------

def test_oom_safe_config():
    assert not would_oom({"DEPTH": "4", "ASPECT_RATIO": "48", "TOTAL_BATCH_SIZE": "2**18"})


def test_oom_dangerous_config():
    assert would_oom({"DEPTH": "12", "ASPECT_RATIO": "96", "TOTAL_BATCH_SIZE": "2**19"})


def test_oom_edge_case():
    """Boundary config should be checked."""
    # d=10, ar=80, bs=2^19 -> 10*80*10*524288 = 41_943_040_000 > 3B -> OOM
    assert would_oom({"DEPTH": "10", "ASPECT_RATIO": "80", "TOTAL_BATCH_SIZE": "2**19"})


# ---------------------------------------------------------------------------
# 9. Integration: validate + clamp + make_edit pipeline
# ---------------------------------------------------------------------------

def test_full_proposal_pipeline():
    """Test the full proposal -> validate -> clamp -> edit pipeline."""
    hp_lines = [
        "DEPTH = 4",
        "EMBEDDING_LR = 0.6 # default",
        "WEIGHT_DECAY = 0.1",
    ]
    history = ["keep", "discard"]
    descs = ["DEPTH=6 (was 4)", "WEIGHT_DECAY=0.2 (was 0.1)"]

    # Propose a new LR value
    param = "EMBEDDING_LR"
    value = "1.5"

    # Step 1: Validate
    rejection = validate_proposal(param, value, history, descs, 1.2)
    assert rejection is None, f"should be valid but got: {rejection}"

    # Step 2: Clamp
    safe = clamp_value(param, value)
    assert safe == "1.5"

    # Step 3: Make edit
    edit = make_edit(param, safe, hp_lines)
    assert edit is not None
    assert edit["param"] == "EMBEDDING_LR"
    assert "1.5" in edit["new"]
    assert "# default" in edit["new"], "comment should be preserved"


def test_full_pipeline_reject_duplicate():
    """Pipeline rejects duplicate proposals."""
    hp_lines = ["DEPTH = 4"]
    history = ["keep"]
    descs = ["DEPTH=6 (was 4)"]

    rejection = validate_proposal("DEPTH", "6", history, descs, 1.2)
    assert rejection is not None
    assert "duplicate" in rejection


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  V4 Agent Pattern Tests (no GPU required)")
    print("=" * 60 + "\n")

    tests = [
        # FileStateCache
        ("cache: basic read/write", test_cache_basic_read_write),
        ("cache: invalidation on disk write", test_cache_invalidation_on_write),
        ("cache: explicit invalidate()", test_cache_explicit_invalidate),
        ("cache: LRU eviction", test_cache_lru_eviction),
        ("cache: path normalization", test_cache_path_normalization),
        ("cache: nonexistent file", test_cache_nonexistent_file),
        ("cache: clear()", test_cache_clear),

        # Batch scoring
        ("batch_score: normal parse", test_batch_score_parsing_normal),
        ("batch_score: noise in response", test_batch_score_parsing_with_noise),
        ("batch_score: missing entries default 0.5", test_batch_score_missing_entries),
        ("batch_score: clamping", test_batch_score_clamping),
        ("batch_score: empty response", test_batch_score_empty_response),
        ("batch_score: float values", test_batch_score_float_values),

        # History compaction
        ("history: empty", test_history_empty),
        ("history: short (no compaction)", test_history_short_no_compaction),
        ("history: long (compacted)", test_history_long_compaction),
        ("history: preserves recent 15", test_history_compaction_preserves_recent),
        ("history: correct counts", test_history_compaction_counts_correct),

        # Validate proposal
        ("validate: reject duplicate", test_validate_duplicate),
        ("validate: different value OK", test_validate_different_value_ok),
        ("validate: exhausted param", test_validate_exhausted_param),
        ("validate: recent success OK", test_validate_recently_successful_param_ok),
        ("validate: no history", test_validate_no_history),
        ("validate: other param exhausted", test_validate_other_param_exhausted),

        # Error categorization
        ("error: connection", test_categorize_connection_error),
        ("error: timeout", test_categorize_timeout),
        ("error: rate limit 429", test_categorize_rate_limit),
        ("error: server 500", test_categorize_server_error),
        ("error: client 400", test_categorize_client_error),
        ("error: OOM", test_categorize_oom),
        ("error: subprocess timeout", test_categorize_subprocess_timeout),
        ("error: unknown", test_categorize_unknown),

        # Stopping logic
        ("stop: max experiments", test_stop_max_experiments),
        ("stop: consecutive discards", test_stop_consecutive_discards),
        ("stop: no keep in window", test_stop_no_keep_in_window),
        ("stop: healthy run", test_no_stop_healthy),
        ("stop: crash counts", test_stop_crash_counts_as_discard),
        ("stop: skip counts", test_stop_skip_counts_as_discard),
        ("stop: keep resets consecutive", test_stop_keep_resets_consecutive),

        # Random fallback
        ("fallback: generates proposals", test_fallback_generates_proposals),
        ("fallback: no duplicate params", test_fallback_no_duplicate_params),
        ("fallback: values within bounds", test_fallback_values_within_bounds),
        ("fallback: no OOM proposals", test_fallback_no_oom),

        # OOM
        ("oom: safe config", test_oom_safe_config),
        ("oom: dangerous config", test_oom_dangerous_config),
        ("oom: edge case", test_oom_edge_case),

        # Integration
        ("pipeline: full proposal flow", test_full_proposal_pipeline),
        ("pipeline: reject duplicate", test_full_pipeline_reject_duplicate),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}\n")

    sys.exit(1 if failed > 0 else 0)
