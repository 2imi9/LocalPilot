"""
Unit test: Relevance scoring for research findings.

V4 needs to score how relevant each paper/finding is to our specific task
(tuning a small GPT ~124M params on Shakespeare-like data).

Tests:
  1. Scoring function produces scores in [0, 1] range
  2. Highly relevant papers score higher than irrelevant ones
  3. Qwen-based scoring works via llama-server
  4. Edge cases: empty input, very long input, non-English text
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LLAMA_SERVER_EXE = str(ROOT.parent / "llama.cpp" / "llama-server.exe")
QWEN_MODEL = str(ROOT.parent / "models" / "qwen3.5-9b" / "Qwen3.5-9B-Q6_K.gguf")
LLAMA_PORT = 8080
LLAMA_API_BASE = f"http://localhost:{LLAMA_PORT}"

import requests


def start_qwen():
    try:
        r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
        if r.status_code == 200:
            print("[Qwen already running]")
            return None
    except Exception:
        pass

    print("[Starting Qwen3.5-9B...]")
    proc = subprocess.Popen(
        [LLAMA_SERVER_EXE, "-m", QWEN_MODEL, "-ngl", "99",
         "--port", str(LLAMA_PORT), "--ctx-size", "4096",
         "-t", "8", "--no-mmap",
         "--chat-template-kwargs", '{"enable_thinking":false}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
            if r.status_code == 200:
                print("[Qwen ready]")
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError("Qwen failed to start")


def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)
        print("[Server stopped]")


def call_qwen(prompt, max_tokens=100, temperature=0.0):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{LLAMA_API_BASE}/v1/chat/completions",
                      json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ===========================================================================
# Relevance scoring function (will be used in V4)
# ===========================================================================
def score_relevance(finding: str, task_context: str = "") -> float:
    """
    Score how relevant a research finding is to our task.
    Uses Qwen to rate relevance on a 0-10 scale, normalized to [0, 1].

    Args:
        finding: Paper title + abstract or extracted text
        task_context: Description of what we're optimizing

    Returns:
        float in [0, 1] where 1 = highly relevant
    """
    if not finding or len(finding.strip()) < 10:
        return 0.0

    if not task_context:
        task_context = (
            "Tuning hyperparameters (learning rate, optimizer settings, schedule, "
            "architecture ratios) for a small GPT language model (~124M params) "
            "trained on text data. GPU: RTX 5090, 24GB VRAM, 5-min training budget."
        )

    prompt = f"""Rate how relevant this research finding is to our task.

OUR TASK: {task_context}

RESEARCH FINDING:
{finding[:1500]}

Rate relevance from 0 to 10:
- 0: Completely unrelated (e.g., biology, image classification)
- 3: Tangentially related (e.g., general ML but not transformers)
- 5: Somewhat related (e.g., transformer training but different scale)
- 7: Quite relevant (e.g., small transformer hyperparameter tuning)
- 10: Directly applicable (e.g., specific HP values for ~100M param GPT)

Output ONLY a single number (0-10):"""

    response = call_qwen(prompt, max_tokens=10, temperature=0.0)

    # Extract number from response
    m = re.search(r'(\d+(?:\.\d+)?)', response)
    if m:
        score = float(m.group(1))
        return max(0.0, min(1.0, score / 10.0))
    return 0.5  # default if parsing fails


def score_relevance_batch(findings: list[str], task_context: str = "") -> list[dict]:
    """Score multiple findings, return sorted by relevance."""
    scored = []
    for f in findings:
        score = score_relevance(f, task_context)
        scored.append({"finding": f[:200], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ===========================================================================
# Tests
# ===========================================================================

# Test papers with known relevance levels
HIGHLY_RELEVANT = {
    "title": "Optimal Learning Rate Schedules for Small Transformer Language Models",
    "abstract": "We study learning rate scheduling for GPT-style transformers with "
    "100M-200M parameters. We find that a warmup ratio of 0.05 followed by cosine "
    "decay to 0.01x the peak LR yields 3% lower perplexity. Adam beta2=0.95 "
    "outperforms the default 0.999 for models under 500M params. Weight decay of "
    "0.1 is optimal across all tested scales.",
}

SOMEWHAT_RELEVANT = {
    "title": "Scaling Laws for Neural Language Models",
    "abstract": "We study the scaling behavior of language model performance as a "
    "function of model size, dataset size, and compute budget. We find power-law "
    "relationships between these quantities and cross-entropy loss. Our experiments "
    "span models from 768 parameters to 1.5 billion parameters.",
}

NOT_RELEVANT = {
    "title": "Deep Learning for Protein Folding: A Survey",
    "abstract": "We survey recent advances in protein structure prediction using "
    "deep learning, including AlphaFold2 and ESMFold. These methods use "
    "attention mechanisms to predict 3D protein structures from amino acid "
    "sequences with near-experimental accuracy.",
}

EDGE_CASE_EMPTY = {"title": "", "abstract": ""}
EDGE_CASE_LONG = {
    "title": "A" * 500,
    "abstract": "We propose a novel approach to training transformers. " * 100,
}


def test_score_range():
    """Test that scores are always in [0, 1]."""
    print("\n" + "="*60)
    print("TEST 1: Score range [0, 1]")
    print("="*60)

    for paper in [HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, NOT_RELEVANT]:
        text = f"{paper['title']}. {paper['abstract']}"
        score = score_relevance(text)
        assert 0.0 <= score <= 1.0, f"FAIL: score {score} out of range for {paper['title'][:50]}"
        print(f"  {paper['title'][:50]:50s} -> {score:.2f}")

    print("  PASS: All scores in [0, 1]")


def test_relative_ordering():
    """Test that highly relevant > somewhat > not relevant."""
    print("\n" + "="*60)
    print("TEST 2: Relative ordering")
    print("="*60)

    high = score_relevance(f"{HIGHLY_RELEVANT['title']}. {HIGHLY_RELEVANT['abstract']}")
    medium = score_relevance(f"{SOMEWHAT_RELEVANT['title']}. {SOMEWHAT_RELEVANT['abstract']}")
    low = score_relevance(f"{NOT_RELEVANT['title']}. {NOT_RELEVANT['abstract']}")

    print(f"  Highly relevant:   {high:.2f}")
    print(f"  Somewhat relevant: {medium:.2f}")
    print(f"  Not relevant:      {low:.2f}")

    assert high > low, f"FAIL: highly relevant ({high}) should score higher than irrelevant ({low})"
    assert high >= medium, f"FAIL: highly relevant ({high}) should score >= somewhat ({medium})"
    # Note: medium vs low may be close, so we only assert high > low strictly

    print("  PASS: Ordering correct (high > low)")


def test_batch_scoring():
    """Test batch scoring and sorting."""
    print("\n" + "="*60)
    print("TEST 3: Batch scoring")
    print("="*60)

    findings = [
        f"{NOT_RELEVANT['title']}. {NOT_RELEVANT['abstract']}",
        f"{HIGHLY_RELEVANT['title']}. {HIGHLY_RELEVANT['abstract']}",
        f"{SOMEWHAT_RELEVANT['title']}. {SOMEWHAT_RELEVANT['abstract']}",
    ]

    scored = score_relevance_batch(findings)

    print("  Sorted results:")
    for item in scored:
        print(f"    {item['score']:.2f} | {item['finding'][:80]}")

    # The highest scored should be the highly relevant one
    assert "Learning Rate" in scored[0]["finding"] or "Optimal" in scored[0]["finding"], \
        f"FAIL: top result should be the HP tuning paper, got: {scored[0]['finding'][:80]}"

    print("  PASS: Batch scoring works, most relevant ranked first")


def test_edge_cases():
    """Test edge cases: empty input, very long input."""
    print("\n" + "="*60)
    print("TEST 4: Edge cases")
    print("="*60)

    # Empty input
    score_empty = score_relevance("")
    assert score_empty == 0.0, f"FAIL: empty input should score 0, got {score_empty}"
    print(f"  Empty input:  {score_empty:.2f} (expected 0.00)")

    # Very short input
    score_short = score_relevance("hi")
    assert score_short == 0.0, f"FAIL: very short input should score 0, got {score_short}"
    print(f"  Short input:  {score_short:.2f} (expected 0.00)")

    # Very long input (should be truncated, not crash)
    long_text = f"{EDGE_CASE_LONG['title']}. {EDGE_CASE_LONG['abstract']}"
    score_long = score_relevance(long_text)
    assert 0.0 <= score_long <= 1.0, f"FAIL: long input score {score_long} out of range"
    print(f"  Long input:   {score_long:.2f} (in range)")

    print("  PASS: Edge cases handled correctly")


if __name__ == "__main__":
    print("Relevance Scoring Tests")
    print("="*60)

    proc = start_qwen()
    passed = 0
    failed = 0

    for test_fn in [test_score_range, test_relative_ordering, test_batch_scoring, test_edge_cases]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    stop_server(proc)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
