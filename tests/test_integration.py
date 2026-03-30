"""
Integration test: Full V4 pipeline dry run.

Tests the complete loop WITHOUT training (mocks train.py execution).
Verifies that all stages connect correctly:
  1. Orchestrator plans search query
  2. Research session (Scholar API, no MolmoWeb for speed)
  3. Relevance scoring filters findings
  4. Orchestrator proposes PARAM + VALUE
  5. Value clamped to bounds
  6. Edit produced for train.py
  7. OOM check passes
  8. (Training would happen here - mocked)
  9. Keep/discard decision

This is a fast smoke test (~2 min with Qwen).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LLAMA_SERVER_EXE = str(ROOT.parent / "llama.cpp" / "llama-server.exe")
QWEN_MODEL = str(ROOT.parent / "models" / "qwen3.5-9b" / "Qwen3.5-9B-Q6_K.gguf")
LLAMA_PORT = 8080
LLAMA_API_BASE = f"http://localhost:{LLAMA_PORT}"

import requests
from test_devstral_refine import clamp_value, make_edit, SAFE_CONTINUOUS, SAFE_DISCRETE
from test_relevance_scoring import score_relevance

# Parameter bounds
ALL_PARAM_NAMES = list(SAFE_CONTINUOUS.keys()) + list(SAFE_DISCRETE.keys()) + ["ADAM_BETAS"]


def start_qwen():
    try:
        r = requests.get(f"{LLAMA_API_BASE}/health", timeout=2)
        if r.status_code == 200:
            return None
    except Exception:
        pass
    print("[Starting Qwen3.5-9B...]")
    proc = subprocess.Popen(
        [LLAMA_SERVER_EXE, "-m", QWEN_MODEL, "-ngl", "99",
         "--port", str(LLAMA_PORT), "--ctx-size", "8192",
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


def call_qwen(prompt, max_tokens=300, temperature=0.7):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{LLAMA_API_BASE}/v1/chat/completions",
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def get_hp_lines():
    """Read actual HP lines from train.py."""
    content = (ROOT / "train.py").read_text(encoding="utf-8")
    return [l for l in content.splitlines() if re.match(r'^[A-Z_]+ = ', l)][:20]


def test_full_pipeline():
    """Run one complete V4 experiment loop (dry run, no actual training)."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full V4 Pipeline Dry Run")
    print("="*60)

    hp_lines = get_hp_lines()
    best_bpb = 1.1507
    results_history = ["keep", "discard", "keep"]
    tried_descs = [
        "EMBEDDING_LR=1.5 (was 0.6): Paper suggests higher embedding LR",
        "DEPTH=6 (was 4): Scaling laws paper",
        "TOTAL_BATCH_SIZE=2**17 (was 2**18): Reduce batch for faster iteration",
    ]

    # -------------------------------------------------------
    # Stage 1: Orchestrator plans search query
    # -------------------------------------------------------
    print("\n--- Stage 1: Plan search query ---")
    hp_numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(hp_lines))
    history = "\n".join(f"  {i+1}. [{s.upper()}] {d}"
                        for i, (s, d) in enumerate(zip(results_history, tried_descs), 1))

    search_prompt = f"""You are an ML research orchestrator tuning a small GPT language model (~124M params).
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.

Current hyperparameters:
{hp_numbered}

Experiment history:
{history}

Decide what research direction to explore NEXT.
Output EXACTLY one line:
SEARCH: <5-15 word query for Semantic Scholar>"""

    search_response = call_qwen(search_prompt, max_tokens=100, temperature=0.7)
    search_m = re.search(r"SEARCH:\s*(.+)", search_response)
    assert search_m, f"Stage 1 FAIL: no SEARCH in response: {search_response}"
    query = search_m.group(1).strip()
    print(f"  Query: {query}")

    # -------------------------------------------------------
    # Stage 2: Research session (Semantic Scholar only for speed)
    # -------------------------------------------------------
    print("\n--- Stage 2: Research session ---")
    from localpilot.browse import scholar_search
    try:
        papers = scholar_search(query, max_results=5)
        print(f"  Found {len(papers)} papers")
        for p in papers[:3]:
            print(f"    - {p['title'][:80]}")
    except Exception as e:
        print(f"  Scholar search failed: {e}")
        papers = []

    # -------------------------------------------------------
    # Stage 3: Relevance scoring
    # -------------------------------------------------------
    print("\n--- Stage 3: Relevance scoring ---")
    scored_papers = []
    for p in papers[:5]:
        text = f"{p['title']}. {p.get('abstract', '')}"
        score = score_relevance(text)
        scored_papers.append({"paper": p, "score": score})
        print(f"  {score:.2f} | {p['title'][:60]}")

    # Filter: only keep papers with score >= 0.3
    relevant = [sp for sp in scored_papers if sp["score"] >= 0.3]
    print(f"  Relevant papers (score >= 0.3): {len(relevant)}/{len(scored_papers)}")

    # Build research summary from relevant papers
    research_summary = ""
    if relevant:
        for sp in relevant[:3]:
            p = sp["paper"]
            research_summary += f"- {p['title']} (score: {sp['score']:.1f})\n"
            if p.get("abstract"):
                research_summary += f"  {p['abstract'][:300]}\n\n"
    else:
        research_summary = "No sufficiently relevant papers found."

    # -------------------------------------------------------
    # Stage 4: Orchestrator proposes PARAM + VALUE
    # -------------------------------------------------------
    print("\n--- Stage 4: Orchestrator proposal ---")
    available = [p for p in ALL_PARAM_NAMES]
    avail_str = "\n".join(f"  - {name}" for name in available)

    proposal_prompt = f"""You are an ML research orchestrator tuning a GPT language model.
Current best val_bpb: {best_bpb:.4f} (lower is better). Starting was 1.379.

Current hyperparameters:
{hp_numbered}

Parameters you can choose from:
{avail_str}

Research findings:
{research_summary[:2000]}

Based on the research findings, pick ONE parameter and propose a specific new value.

Output EXACTLY 3 lines:
PARAM: <parameter name>
VALUE: <the exact new value>
REASON: <one sentence citing research>
"""

    proposal_response = call_qwen(proposal_prompt, max_tokens=200, temperature=0.7)
    param_m = re.search(r"PARAM:\s*(\w+)", proposal_response)
    value_m = re.search(r"VALUE:\s*(.+)", proposal_response)
    reason_m = re.search(r"REASON:\s*(.+)", proposal_response)

    assert param_m, f"Stage 4 FAIL: no PARAM: {proposal_response}"
    assert value_m, f"Stage 4 FAIL: no VALUE: {proposal_response}"

    param = param_m.group(1).strip()
    value = value_m.group(1).strip()
    reason = reason_m.group(1).strip() if reason_m else "no reason"

    print(f"  PARAM:  {param}")
    print(f"  VALUE:  {value}")
    print(f"  REASON: {reason}")

    # -------------------------------------------------------
    # Stage 5: Clamp value + produce edit
    # -------------------------------------------------------
    print("\n--- Stage 5: Clamp + edit ---")
    safe_value = clamp_value(param, value)
    print(f"  Clamped: {value} -> {safe_value}")

    edit = make_edit(param, safe_value, hp_lines)
    if edit:
        print(f"  Old: {edit['old']}")
        print(f"  New: {edit['new']}")
    else:
        print(f"  No edit produced (value unchanged or param not found)")
        print("  This is acceptable -- LLM proposed same value or invalid param")

    # -------------------------------------------------------
    # Stage 6: OOM check
    # -------------------------------------------------------
    print("\n--- Stage 6: OOM check ---")
    # Build current HP dict
    hp_dict = {}
    for l in hp_lines:
        m = re.match(r'^(\w+)\s*=\s*(.+?)(?:\s*#.*)?$', l)
        if m:
            hp_dict[m.group(1)] = m.group(2).strip()
    # Apply proposed change
    if edit:
        hp_dict[param] = safe_value

    # Simple OOM check
    d = int(hp_dict.get("DEPTH", 4))
    ar = int(hp_dict.get("ASPECT_RATIO", 48))
    bs_str = str(hp_dict.get("TOTAL_BATCH_SIZE", "2**18"))
    bs_val = int(eval(bs_str)) if "**" in bs_str else int(bs_str)
    oom_score = d * ar * d * bs_val
    would_oom = oom_score > 3_000_000_000

    print(f"  Config: DEPTH={d}, AR={ar}, BS={bs_str}")
    print(f"  OOM score: {oom_score:,}")
    print(f"  Would OOM: {would_oom}")

    if would_oom:
        print("  [BLOCKED: would OOM, skip this experiment]")
    else:
        print("  [OK: safe to train]")

    # -------------------------------------------------------
    # Stage 7: Mock training result
    # -------------------------------------------------------
    print("\n--- Stage 7: Training (mocked) ---")
    if edit and not would_oom:
        # Simulate a training result
        import random
        mock_bpb = best_bpb + random.uniform(-0.01, 0.02)
        if mock_bpb < best_bpb:
            status = "keep"
            print(f"  KEEP: {mock_bpb:.6f} (was {best_bpb:.6f})")
        else:
            status = "discard"
            print(f"  DISCARD: {mock_bpb:.6f} (best: {best_bpb:.6f})")
    else:
        status = "skip"
        print(f"  SKIP (no valid edit or OOM)")

    print(f"\n--- Pipeline result: {status.upper()} ---")
    print("PASS: Full pipeline completed without errors")
    return True


if __name__ == "__main__":
    print("V4 Integration Test")
    print("="*60)

    proc = start_qwen()
    try:
        success = test_full_pipeline()
        print(f"\n{'='*60}")
        print(f"Integration test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_server(proc)
