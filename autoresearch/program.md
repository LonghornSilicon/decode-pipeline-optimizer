# HADES Speculative Decoding Autoresearch

You are an autonomous research agent proving that speculative decoding gives 2–4× decode
throughput in the context of the HADES LLM accelerator chip architecture.

You modify `train.py`, run the benchmark, check if `tokens_per_second` improved, keep or
discard, and repeat. You run forever until manually stopped.

---

## Setup

1. **Agree on a run tag** with the user (e.g. `specdec-apr26`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3. **Read in-scope files**:
   - `program.md` — these instructions
   - `train.py` — **the only file you edit**. Contains the speculative decode benchmark.
   - `prepare.py` — model/tokenizer setup, do NOT modify.
4. **Initialize results.tsv** with just the header row.
5. **Confirm** setup, then immediately kick off the experiment loop.

---

## The Problem

**Goal**: Prove speculative decoding with a small draft model achieves ≥2× tokens/sec
vs. standard autoregressive decoding, while:
- Keeping KV cache total ≤ 8 MB (HADES 16nm on-chip SRAM budget)
- Maintaining output quality (val_bpb within 2% of baseline)
- Running a draft model small enough to fit a dedicated hardware lane

**Primary metric**: `tokens_per_second` (higher is better)
**Secondary**: `acceptance_rate` (target ≥ 0.70), `kv_memory_mb` (must be ≤ 8.0)
**Quality gate**: `val_bpb` must be ≤ `baseline_bpb * 1.02` (within 2% degradation)

---

## Speedup Formula (Your North Star)

```
speedup = (gamma + 1) / (gamma * (1 - alpha) + 1)
```

- alpha = acceptance rate (0 to 1)
- gamma = draft length (tokens proposed per step)

To get ≥ 2×: need alpha ≥ 0.70 with gamma ≥ 4, OR alpha ≥ 0.80 with gamma ≥ 3.
The theoretical max at alpha=0.85, gamma=8 is 4.24×.

---

## HADES Context (Why This Matters)

The HADES chip has:
- **Step 1 (KV Cache Engine)**: 4-bit quantized KV, 8–32 MB on-chip SRAM
- **Step 2 (Token Importance Unit)**: Attention-weight-based importance scores
- **Step 3 (Memory Hierarchy)**: L1 SRAM → L2 eDRAM → off-chip DRAM
- **Step 5 (This experiment)**: Dedicated draft model lane + merge unit

Key constraint: draft KV + main KV must fit in 8 MB with 4-bit compression.
Key opportunity: token importance scores (Step 2) may predict which tokens will be accepted.

---

## What You Can Modify in train.py

Everything in `train.py` is fair game:
- Draft model architecture (num_layers, num_heads, d_model — keep params ≤ 200M)
- Draft length gamma (try 2 to 10)
- Acceptance strategy: standard (threshold), top-k filtered, temperature-adjusted
- KV quantization bits (2, 4, 8) for budget simulation
- Whether to use token importance scores to skip speculation on high-entropy positions
- Draft model initialization (random, distilled from target, shared embeddings)

## What You Cannot Do

- Modify `prepare.py`
- Install new packages beyond: `torch`, `numpy`, `transformers`, `datasets`, `math`, `random`
- Hardcode outputs — the benchmark must run live inference

---

## Proven Strategies (in order of expected impact)

### Tier 1 — Quick wins (first ~20 experiments)
1. **Tune gamma**: Try draft lengths 2, 3, 4, 5, 6, 8. Find the sweet spot where
   acceptance rate doesn't drop faster than the parallelism benefit.
2. **Acceptance threshold**: Standard spec-decode uses threshold τ=1.0 (pure sampling match).
   Try τ=0.8, 0.9 — slightly lower bar → higher acceptance at tiny quality cost.
3. **Greedy draft**: Draft model uses greedy decoding (argmax), main model uses sampling.
   Often increases acceptance rate by 5–10 percentage points.
4. **Temperature-adjusted acceptance**: Scale draft logits by 0.8× before acceptance check —
   makes draft tokens more confident and increases acceptance on repetitive/formulaic text.

### Tier 2 — Architecture (experiments 20–50)
5. **Shrink draft to 50M params**: Smaller draft → faster draft pass → more speculative steps
   per wall-clock second. Try (4 layers, 8 heads, d_model=256).
6. **Shared token embeddings**: Use target model's embedding layer in the draft model.
   This aligns vocabulary distributions and raises acceptance rate by ~3–8%.
7. **Single-layer draft**: Ultra-minimal draft (1 layer, 8 heads, d_model=512).
   Extremely fast draft pass. Acceptance rate ~55–65% but draft overhead is negligible.
8. **Draft with KV reuse**: Draft model attends to a compressed subset of the main model's
   KV cache instead of maintaining its own. Saves memory at the cost of some accuracy.

### Tier 3 — HADES-Specific (experiments 50–80)
9. **Simulate 4-bit KV compression**: Quantize both draft and main model KV to 4-bit during
   the benchmark. Measure if acceptance rate changes with quantized KV (it should stay within
   2–3% with GEAR-style error correction). Show total KV fits ≤ 8 MB.
10. **Token importance gating**: Compute per-position entropy of the target model's logits.
    On high-entropy positions (uncertain tokens), skip speculation and decode directly.
    This is a proxy for Step 2 (Token Importance Unit) in hardware — may raise α on the
    remaining speculative positions.
11. **Adaptive gamma**: Instead of fixed draft length, dynamically extend draft if recent
    acceptance rate is high, shorten if acceptance rate drops. Models what hardware could do
    with a feedback loop from the merge unit.
12. **Verify-in-parallel**: Batch verification of all γ draft tokens in a single forward pass
    (this is how the hardware works). Ensure train.py is using batch verification, not
    sequential token-by-token verification.

---

## Running an Experiment

**On Brev (A100 GPU):**
```bash
python train.py > run.log 2>&1
```

**Locally (small test, CPU):**
```bash
python train.py --quick > run.log 2>&1
```

Read results:
```bash
grep "^tokens_per_second:\|^acceptance_rate:\|^kv_memory_mb:\|^val_bpb:\|^speedup:" run.log
```

---

## Output Format

Every run must print (exactly this format, one per line):
```
---
tokens_per_second:  423.7
baseline_tps:       198.2
speedup:            2.14x
acceptance_rate:    0.743
kv_memory_mb:       3.81
val_bpb:            1.023
baseline_bpb:       1.009
quality_ok:         YES
budget_ok:          YES
total_seconds:      287.3
```

`quality_ok: YES` means val_bpb ≤ baseline_bpb × 1.02
`budget_ok: YES` means kv_memory_mb ≤ 8.0

**Discard if**: `quality_ok: NO` OR `budget_ok: NO` (regardless of speedup)

---

## Logging

Log every run to `results.tsv` (tab-separated). Do NOT commit it.

Header:
```
commit	tokens_per_second	speedup	acceptance_rate	kv_memory_mb	quality_ok	budget_ok	status	description
```

Status: `keep`, `discard`, `crash`

---

## Experiment Loop

LOOP FOREVER:

1. Read current git state (branch + commit).
2. Pick next experiment from the strategy list (or extend a winning idea).
3. Modify `train.py`.
4. `git commit -m "experiment: <short description>"`
5. Run: `python train.py > run.log 2>&1`
6. `grep "^tokens_per_second:\|^speedup:\|^quality_ok:\|^budget_ok:" run.log`
7. If nothing returned → crash. `tail -n 30 run.log`. Fix and retry up to 3×.
8. If `quality_ok: NO` or `budget_ok: NO` → discard regardless of speedup.
9. Log to `results.tsv`.
10. If `tokens_per_second` improved AND both gates pass → **keep**. New baseline.
11. Else → `git reset --hard HEAD~1`. Discard.

Every 5 experiments: print best result so far (tokens_per_second, speedup, acceptance_rate).

**PROOF THRESHOLD**: When you achieve `speedup ≥ 2.00×` with `quality_ok: YES` and
`budget_ok: YES`, flag it clearly:

```
*** PROOF POINT REACHED: 2x speedup proven ***
tokens_per_second: XXX  speedup: X.XXx  alpha: 0.XX  kv_mb: X.XX
```

Keep running to find even better configurations (toward 4×).

---

## Key Numbers

| Baseline (autoregressive) | ~200 tokens/sec (GPT-2 medium, A100) |
| 2× target | ~400 tokens/sec |
| 4× target | ~800 tokens/sec |
| KV budget | ≤ 8 MB (conservative 16nm SRAM) |
| Quality gate | val_bpb ≤ baseline × 1.02 |
| Alpha threshold for 2× (γ=4) | α ≥ 0.70 |
