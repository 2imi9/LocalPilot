# AutoResearch — LocalPilot

**An autonomous research agent that visually browses arXiv papers, reasons about what to try, and trains models — every experiment cites a paper, every improvement is explainable.**

![teaser](figures/fig_teaser.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## Why LocalPilot?

Most autoresearch systems use **random perturbation** — blindly tweak a number, train, keep if better. This works, but you learn nothing about *why* something worked, and every failed experiment is wasted compute with no insight.

LocalPilot is different: it has a **visual web browsing agent** ([MolmoWeb-4B](https://huggingface.co/allenai/MolmoWeb-4B-0225)) that reads real arXiv papers — not just titles or abstracts, but full figures, tables, and methods sections — then reasons about what to try next.

| | Random perturbation | LocalPilot |
|---|---|---|
| Browses papers | No | Yes — MolmoWeb visually reads full PDFs |
| Searches literature | No | Yes — arXiv via visual browsing |
| Proposals are explainable | No | Yes — every change cites a paper |
| Learns from failures | No | Yes — LLM sees full history |
| Scales to larger search spaces | Poorly | Naturally |
| Runs fully local | Yes | Yes — no cloud APIs needed |

## What it does

```
  You run it                                   It does this, autonomously
  ─────────                                    ──────────────────────────
  uv run python experiments/run_enhanced_v3.py ─>  1. Reads train.py + past results
                                                 2. Qwen3.5-9B plans what to search
                                                 3. MolmoWeb-4B browses arXiv visually
                                                 4. Devstral-24B proposes a HP change, citing why
                                                 5. Edits train.py, trains locally
                                                 6. Keeps if val_bpb improves, reverts if not
                                                 7. Loops — gets smarter each iteration
```

The key innovation is **step 3**: MolmoWeb-4B is a visual web agent that takes screenshots of web pages and interacts with them like a human would. It navigates to arXiv papers, scrolls through figures and tables, and extracts specific techniques — not just keyword matches from abstracts.

All models run locally (Qwen3.5-9B for orchestration, MolmoWeb-4B for browsing, Devstral-24B for code). No API keys, no cloud bills.

## Results

### karpathy/autoresearch benchmark

Starting from the karpathy baseline config (val_bpb ~1.268), LocalPilot found **11 paper-traceable improvements** reaching **1.1507 BPB** in 64 experiments:

```
Experiment #50: WINDOW_PATTERN "SL" → "L"
  Reason: "Switching to full-context pattern mitigates instabilities in
           small-scale proxies where limited context exacerbates noise"
  Result: val_bpb 1.1510 → 1.1507 ✓ kept
```

### Surrogate benchmark validation (YAHPO LCBench)

To validate with proper statistics, we ran both methods on [YAHPO Gym](https://github.com/slds-lmu/yahpo_gym) (LCBench: 7 HPs, neural net tuning, instant surrogate evaluations) with **500 seeds each**:

![yahpo](figures/fig6_yahpo.png)

| | Random search | Informed search |
|---|---|---|
| Median val_cross_entropy | 0.1485 | **0.1124** |
| Improvement | — | **24% better** |
| Seeds | 500 | 500 |

With enough seeds and a larger search space, informed search clearly dominates random perturbation. The karpathy benchmark (13 bounded HPs) is deliberately constrained — see [Limitations](#limitations).

## Quick start

**Requirements:** Windows with NVIDIA GPU (24+ GB VRAM for default models, or 12+ GB with Q4 quants — see `localpilot.yaml`), Python 3.11+, [uv](https://docs.astral.sh/uv/), Git, CMake, CUDA toolkit, Docker (optional, for FA3 training)

> **Note:** Currently tested on Windows. Linux/macOS support is planned — the main blockers are hardcoded `.exe` paths in the runner scripts.

### Step 1: Clone and install

```bash
git clone https://github.com/2imi9/autoresearch.git
cd autoresearch
uv sync
```

### Step 2: Download training data

```bash
uv run python prepare.py
```

This downloads FineWeb-Edu shards and trains a BPE tokenizer (~2 min, cached in `~/.cache/autoresearch/`).

### Step 3: Build llama.cpp

The research agent uses [llama.cpp](https://github.com/ggerganov/llama.cpp) to run local LLMs. Build it as a sibling directory:

```bash
cd ..
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

After building, copy (or symlink) the server binary to the repo root:

```bash
# Windows (adjust path if your build config differs)
copy build\bin\Release\llama-server.exe llama-server.exe

# The runner expects it at: ../llama.cpp/llama-server.exe
# (i.e., autoresearch/ and llama.cpp/ are sibling directories)
```

```bash
cd ../autoresearch
```

### Step 4: Download GGUF models

Create a `models/` directory next to `autoresearch/` and download these GGUF files:

```bash
# Create model directories (from autoresearch/ parent)
cd ..
mkdir -p models/qwen3.5-9b models/devstral

# Qwen3.5-9B — code/orchestration agent (~6 GB)
# Download from: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q6_K.gguf --local-dir models/qwen3.5-9b

# Devstral-24B — experiment proposal agent (~19 GB)
# Download from: https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF
huggingface-cli download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf --local-dir models/devstral

cd autoresearch
```

MolmoWeb-4B (the visual web agent) must also be pre-downloaded. Use `huggingface-cli download allenai/MolmoWeb-4B-0225 --local-dir models/MolmoWeb-4B` or download via HuggingFace transformers' `from_pretrained()` caching before running offline.

**Expected directory layout after setup:**
```
parent/
├── autoresearch/       # this repo
├── llama.cpp/          # built with CUDA, llama-server.exe at root
└── models/
    ├── qwen3.5-9b/
    │   └── Qwen3.5-9B-Q6_K.gguf
    ├── devstral/
    │   └── Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf
    └── MolmoWeb-4B/    # visual web agent (pre-download required)
```

### Step 5: Set up the Python environment

The runner scripts expect a local `.venv` created by `uv`. If `uv sync` (Step 1) completed successfully, this is already done. Verify:

```bash
# Should print the Python path inside .venv
python -c "import sys; print(sys.executable)"
```

> **Optional:** A Dockerfile is included for running training inside a Linux container (useful for Flash Attention 3 which requires Linux CUDA). Build with `docker build -t autoresearch-train .` if needed.

### Step 6: Run it

```bash
# Run the autonomous research agent (reads papers, proposes experiments)
uv run python experiments/run_enhanced_v3.py

# Or run the random baseline for comparison (no LLMs needed)
uv run python experiments/run_baseline_v2.py
```

The enhanced runner will pre-flight check that all models exist and print download commands if anything is missing.

### Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: llama-server.exe` | Copy the built binary to `../llama.cpp/llama-server.exe` (see Step 3) |
| `FileNotFoundError: ...Q6_K.gguf` | Download the GGUF model files (see Step 4) |
| `uv sync` fails on torch | Ensure CUDA toolkit is installed; `uv sync` pulls PyTorch with CUDA 13.0 |
| Docker build fails (optional) | Ensure Docker Desktop has WSL2 backend + GPU access enabled |
| Out of VRAM | Edit `localpilot.yaml` to select smaller model variants (Q4 instead of Q6) |

## How the research pipeline works

### V3 (current)

Qwen3.5-9B orchestrates the loop — it decides what to search, MolmoWeb-4B browses arXiv visually, and Devstral-24B writes the code patch:

```
  Qwen3.5-9B orchestrator               Plans search direction from history
         │
         ▼
  MolmoWeb-4B visual browser             Browses arXiv, takes screenshots,
                                         reads figures/tables/methods
         │
         ▼
  Devstral-24B code agent                Writes train.py patch citing papers
```

**Why visual browsing matters:** API-only approaches see titles and abstracts. MolmoWeb sees the actual paper — training curves, architecture diagrams, ablation tables. It can tell the difference between a paper that *mentions* learning rate scheduling and one that *demonstrates* a specific schedule that works for shallow transformers.

### V4 (WIP): tiered pipeline + agent-grade resilience

V4 adds Semantic Scholar + arXiv API search with batch relevance scoring, plus agent design patterns adapted from [Claude Code](https://docs.anthropic.com/en/docs/claude-code):

```
  Semantic Scholar + arXiv API          Fast, free, ~50 papers/query
         │
         ▼
  Qwen batch scoring (0-10)            One LLM call scores ALL papers
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Skip  Summary  Deep-read             Only top papers get browsed
  (<5)  (5-7)    (≥7)
                   │
                   ▼
              MolmoWeb-4B               Takes screenshots, clicks through
                                        figures/tables, extracts techniques
         │
         ▼
  Qwen proposals (thinking mode)       /think enabled for high-quality reasoning
         │
         ▼
  Validation → OOM check → Train       Stop hooks catch bad proposals early
```

**Patterns borrowed from Claude Code's agent framework:**
- **Batch scoring** — all papers scored in one LLM call instead of N sequential calls (~60s faster)
- **History compaction** — old experiments summarized into a structured digest (keeps, failure patterns, parameter coverage) so the LLM gets actionable context, not a raw log
- **Adaptive thinking** — Qwen3.5's native `/think` mode enabled for proposal generation (the highest-stakes decision), disabled for fast scoring/planning
- **Post-proposal validation** — stop hooks reject duplicates, recently-exhausted parameters, and OOM configs before wasting a training run
- **Structured error recovery** — exponential backoff with jitter, error categorization (retryable vs terminal), and a circuit breaker that falls back to random proposals after 3 consecutive research failures

This solves the rate-limiting problem — raw MolmoWeb browsing triggered CDN bans (~1500 HTTP requests per session). Tiered research cuts web requests by ~90%.

## Adapting to your own project

LocalPilot isn't locked to karpathy's train.py. To use it on your own training script:

1. Define your hyperparameters and bounds in `constants.py`
2. Point the runner at your training script
3. Define your evaluation metric (val_bpb, accuracy, loss, etc.)

The LLM reads papers relevant to **your** task and proposes changes specific to **your** setup.

## File structure

```
autoresearch/
├── train.py                  # The file the agent edits
├── prepare.py                # One-time data prep
├── localpilot.yaml           # Model selection config
├── Dockerfile                # CUDA 13.0 + FA3 training image (optional)
│
├── experiments/
│   ├── run_baseline_v2.py    # Random perturbation (Condition A)
│   ├── run_enhanced_v3.py    # Paper-grounded search (Condition B)
│   ├── run_enhanced_v4.py    # V4 (WIP): open values + OOM pre-flight
│   └── run_both.py           # Run both conditions back-to-back
│
├── localpilot/
│   ├── browse.py             # MolmoWeb visual web agent
│   ├── config.py             # Hardware-aware model selection
│   ├── constants.py          # HP bounds and parameter definitions
│   └── analyze.py            # Result analysis + figures
│
├── results/
│   ├── results_baseline_v2.tsv    # Baseline experiment log (45 experiments)
│   ├── results_enhanced_v3.tsv    # Enhanced experiment log (64 experiments)
│   ├── proposals_baseline_v2.jsonl
│   ├── proposals_enhanced_v3.jsonl
│   ├── make_figures.py            # Generates all figures from result data
│   └── analysis.ipynb             # Exploratory analysis notebook
│
├── figures/                  # Publication figures
└── tests/                    # Unit + integration tests
```

## Models and VRAM

All phases are sequential — models load/unload, never run simultaneously:

| Phase | Model | VRAM |
|---|---|---|
| Research | MolmoWeb-4B or 8B | ~8–18 GB |
| Orchestrate + Propose | Qwen3.5-9B + Devstral-24B | 8–25 GB |
| Train | train.py (local or Docker) | ~6–12 GB |

A single 24+ GB GPU handles the full pipeline with default Q6 models (or 12+ GB with Q4 quants). Override model selection via `localpilot.yaml` or environment variables.

## Cost

| | Per experiment | 64-experiment run |
|---|---|---|
| Local GPU (electricity) | ~$0.002 | **~$0.10** |
| Cloud H100 ($2.49/hr) | ~$0.23 | ~$14.70 |

**~150x cheaper** than cloud. Calculated at $0.13/kWh, RTX 5090 Laptop at 150W.

## Limitations

**Benchmark scope:** The karpathy benchmark (13 bounded HPs, 5-min training runs) is a single run (n=1). We don't yet have multi-seed variance measurements for the full agent pipeline, so we can't claim statistical significance on this benchmark alone.

**YAHPO validation:** The surrogate benchmark (500 seeds, 24% improvement) validates that *informed search beats random search in principle* — but it uses a simulated informed strategy on a different task (LCBench), not our actual MolmoWeb + Qwen + Devstral pipeline. It's evidence for the approach, not a direct measurement of our system.

**Paper citations:** The LLM generates references to justify its proposals (e.g., "[Smith2026]"). These citations reflect the LLM's training data, not verified literature lookups — though MolmoWeb does browse real papers during the search phase.

The real value of paper-grounded search emerges with:

- **Larger search spaces** — architecture choices, data mixing, training schedules
- **Expensive training** — when each failed experiment costs hours, not minutes
- **Structural changes** — new attention patterns, optimizer variants, positional embeddings

We chose this constrained benchmark to validate the system end-to-end. More rigorous multi-seed evaluation and larger-scale benchmarks are future work.

## Contributing / Future work

Good first issues for contributors:

- **Linux/macOS support** — remove hardcoded `.exe` paths in runner scripts (straightforward)
- **More benchmarks** — try it on fine-tuning, RLHF, or vision models and share results

Bigger research directions:

- **Unbounded parameter search** — V4 allows free values (still clamped to safe bounds). Next step: let the LLM propose entirely new parameters or architectural changes beyond the predefined search space
- **Multi-objective optimization** — optimize for speed + quality, not just val_bpb
- **Smarter paper selection** — V4's tiered pipeline (Scholar + arXiv + relevance scoring) needs testing and tuning
- **Multi-seed evaluation** — run more seeds on the karpathy benchmark to measure variance

PRs welcome. If you try it on your own training setup, open an issue — we'd love to hear what works.

## Based on

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous research framework
- [MolmoWeb-4B](https://huggingface.co/allenai/MolmoWeb-4B-0225) — visual web agent for paper reading
- [Qwen3.5-9B](https://huggingface.co/Qwen) — local orchestrator for search planning
- [Devstral-24B](https://huggingface.co/mistralai) — local code agent for experiment proposals

## License

MIT
