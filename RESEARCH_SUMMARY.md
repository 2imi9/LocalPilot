# LocalPilot: Research Summary

**Date:** 2026-03-27
**Status:** Pilot study complete. Proper A/B run (v2) in progress — both conditions from same raw config, fully automated with MolmoWeb + Devstral.

---

## Experimental Design

Two-condition comparison of autonomous ML hyperparameter search:

- **Condition A (Baseline):** Random/greedy hill-climbing, no external knowledge
- **Condition B (Enhanced):** Ideas sourced from arXiv papers via MolmoWeb visual web browsing + Devstral code agent

The **only difference** between conditions is how experiments are proposed. Hardware, time budget (5 min/exp), training data, and keep/discard logic are identical.

### Design versions

| Version | Baseline start | Enhanced start | Status |
|---|---|---|---|
| **Pilot (v1)** | 1.379 (raw config) | 1.122538 (pre-tuned) | Complete — different starting points, not directly comparable |
| **Proper A/B (v2)** | 1.379 (raw config) | 1.379 (raw config) | **In progress** — `experiments/run_automated_v2.py` |

> The pilot v1 enhanced condition started from an already-tuned config (after baseline's architectural search), so the starting points were not equal. Results are still meaningful as a fine-tuning zone comparison but cannot be directly compared as a full A/B test.

---

## Pilot Results (v1) — Fine-tuning Zone Comparison

Both conditions measured from the shared reference point of val_bpb = 1.122538:

| | Condition A (Baseline) | Condition B (Enhanced, pilot) |
|---|---|---|
| Total experiments | 78 (from raw config) | 53 (from pre-tuned config) |
| Keeps | 22 | 5 |
| Keep rate | 28.2% | 9.4% |
| Reference val_bpb | 1.122538 | 1.122538 |
| **Best val_bpb** | **1.122049** | **1.118972** |
| Improvement from reference | −0.000489 | **−0.003566** |
| Paper-traceable improvements | 0 | **5** |

**Enhanced achieves 7.3x more improvement from the same reference point.** All 5 improvements are traceable to specific arXiv papers (2023–2026).

---

## Condition B Pilot: Kept Experiments

| Exp # | val_bpb   | Delta     | Change | Paper Source |
|-------|-----------|-----------|--------|--------------|
| 7 | 1.121538 | −0.001000 | `ADAM_BETAS=(0.9,0.95)` | Li et al. 2026 (AGGC) |
| 8 | 1.121178 | −0.000360 | `WARMUP=0.03+WARMDOWN=0.97+FINAL_LR=0.001` | Defazio et al. 2023 |
| 15 | 1.119922 | −0.001256 | `MATRIX_LR=0.06` | SpectralClipping 2026 |
| 18 | 1.119175 | −0.000747 | `muon_beta2=0.85` | Qian et al. 2025 (MVR) |
| 20 | 1.118972 | −0.000203 | `MATRIX_LR=0.07` | Qiu et al. 2026 (HT) |

---

## Current Best Config (train.py HEAD — enhanced pilot final state)

```python
EMBEDDING_LR    = 1.0
UNEMBEDDING_LR  = 0.008
MATRIX_LR       = 0.07        # Qiu 2026 HT
SCALAR_LR       = 0.5
WEIGHT_DECAY    = 0.15
ADAM_BETAS      = (0.9, 0.95) # Li 2026 AGGC
WARMUP_RATIO    = 0.03        # Defazio 2023
WARMDOWN_RATIO  = 0.97        # Defazio 2023
FINAL_LR_FRAC   = 0.001       # Defazio 2023
DEPTH           = 4
# Muon optimizer:
momentum        = 0.97
ns_steps        = 5
beta2           = 0.85        # Qian 2025 MVR
```

> **Note:** For the v2 run, train.py will be reset to karpathy's original raw config before starting.

---

## Known Negative Results (from pilot)

**LR Schedule:** Extreme warmdown ratios (0.20, 0.40, 1.0) all worse. FINAL_LR=0.0 worse than 0.001.

**Muon tuning:** momentum={0.95,0.96,0.975,0.98} all worse than 0.97. ns_steps={3,6,10} worse than 5. beta2={0.75,0.80,0.90} all worse than 0.85.

**MATRIX_LR:** Sweet spot is 0.07. Both 0.08 and 0.09 are worse.

**Weight decay:** 0.10, 0.12, 0.18, 0.20 all worse than 0.15.

**Architecture:** n_kv_head=1 (MQA) significantly worse (1.131954).

---

## Infrastructure (v2 run)

- **MolmoWeb-4B**: downloaded to `models/MolmoWeb-4B/`, transformers==4.57.6 compatible
- **Devstral-Small-2-24B Q6_K** (19 GB): at `../models/devstral/`
- **llama-server.exe** (CUDA 13.1): at `../llama.cpp/`
- **Automated runner**: `experiments/run_automated_v2.py` (MolmoWeb -> Devstral -> train loop)

---

## Paper Claim (pending v2 results)

> LocalPilot demonstrates that grounding autonomous hyperparameter search in recent ML literature — via a local visual web agent (MolmoWeb) and local code agent (Devstral) — achieves significantly greater improvement than random greedy search, starting from the same configuration, on the same hardware, with the same compute budget per experiment.
