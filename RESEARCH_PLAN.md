# Research Plan: Proving Speculative Decoding Throughput for HADES

## Thesis

Speculative decoding with a dedicated hardware draft-model lane (as described in the HADES
Step 5 architecture) yields 2–4× decode throughput, and the KV compression + token importance
infrastructure from Steps 1–4 mitigates the memory overhead that would otherwise make this
impractical.

---

## How Steps 1–4 Set Up Step 5

This is important context. The PDF shows HADES is not just "add speculative decoding" in
isolation. Steps 1–4 create the hardware conditions that make Step 5 viable:

| Step | Component | What it gives Step 5 |
|------|-----------|----------------------|
| 1 | KV Cache Engine (2–4 bit quantization, GEAR/RotateKV) | Draft model KV fits in the on-chip SRAM budget without evicting main model KV |
| 2 | Token Importance Unit (mixed-precision retention) | Importance scores predict which speculative tokens will be accepted — can prefetch/precompute smarter |
| 3 | Memory Hierarchy Controller (L1 SRAM → L2 eDRAM → DRAM) | Separate memory lanes for draft vs. main model; draft KV lives in warm eDRAM tier |
| 4 | (Attention Core / Main Decode Path) | The batched verification pass is the same hardware path, just run over γ+1 tokens instead of 1 |
| 5 | **Speculative Decode Assist** | Ties everything together: draft lane runs in parallel, merge unit reconciles, memory hierarchy absorbs both KV sets |

Without Step 1 compression, the draft model's KV cache would blow the SRAM budget and force
off-chip traffic that erases the speedup. This is the crux of why the PDF marks Step 5 optional
for v1 (no compression yet) but viable for v2.

---

## What We Need to Prove

1. **Speedup is real**: Speculative decoding with realistic acceptance rates gives ≥2× throughput
2. **Memory overhead is acceptable**: Draft model KV + 4-bit compression from Step 1 stays within
   the 8–32 MB on-chip SRAM budget at 16nm
3. **Acceptance rate is achievable**: A small draft model achieves ≥70% acceptance on realistic
   workloads (the threshold for ≥2× gain)
4. **Best draft model config**: AutoResearch finds the draft architecture that maximizes
   acceptance × throughput under HADES memory constraints

---

## Hardware Requirements

**You need Brev.** AutoResearch runs 80–100 experiments overnight and requires 20+ GB VRAM.
Your laptop is fine for Phases 1 and 2 (no GPU needed). Use Brev starting Phase 3.

Recommended instance: **A100 40GB or RTX 4090 24GB**

- Phase 1–2: Laptop (CPU only)
- Phase 3: Brev A100 (overnight AutoResearch runs)
- Phase 4: Laptop (analysis of results)

---

## Phase 1 — Analytical Model (Laptop, ~1 day)

**Goal**: Derive the exact conditions under which 2–4× speedup holds. This gives us the
theoretical bound before we run a single experiment.

### The Speedup Equation

For draft length γ and per-token acceptance rate α, the expected speedup is:

```
speedup(α, γ) = (γ + 1) / (γ · (1 - α) + 1)
```

Key values:
| α (acceptance rate) | γ=3 | γ=4 | γ=6 | γ=8 |
|---------------------|-----|-----|-----|-----|
| 0.60 | 1.71× | 1.92× | 2.28× | 2.56× |
| 0.70 | 1.88× | 2.14× | 2.63× | 3.00× |
| 0.80 | 2.08× | 2.45× | 3.13× | 3.67× |
| 0.85 | 2.19× | 2.63× | 3.50× | 4.24× |

The 2–4× claim is valid when α ≥ 0.70 and γ ≥ 4. Phase 3 proves this is achievable.

### KV Memory Budget Analysis

At 16nm, 8–32 MB SRAM on-chip. With Step 1 compression (4-bit quantization):

```
kv_size_per_token = num_heads × head_dim × 2 (K+V) × bits/8
```

For a 7B target model (32 heads, 128 dim, 4-bit): ~1KB/token
For a 125M draft model (12 heads, 64 dim, 4-bit): ~96B/token

Combined at 2048 context: ~2.2 MB — fits in 8 MB SRAM with room for activations.
Without Step 1 compression (16-bit): ~8.8 MB — tight. This is why Steps 1 and 5 are coupled.

**Deliverable**: `analysis/speedup_model.py` — plots speedup surface, KV budget curves.

---

## Phase 2 — Simulation Harness (Laptop, ~2 days)

**Goal**: Build the `train.py` that AutoResearch will iterate on. This needs to run a
speculative decoding benchmark and output a single throughput metric in ~5 minutes.

### Benchmark Design

```
train.py measures:
  - tokens_per_second (main metric, higher is better)
  - acceptance_rate (diagnostic)
  - kv_memory_mb (must stay under budget)
  - output_quality_bpb (must not regress)
```

The agent modifies:
- Draft model architecture (layers, heads, dim)
- Draft length γ
- Acceptance strategy (standard, temperature-adjusted, top-k filtered)
- KV cache allocation ratio (draft vs. main model)

### Model Setup

- **Target model**: GPT-2 Medium (345M) — representative, fits in <24GB VRAM
- **Draft model**: starts as GPT-2 Small (117M), agent evolves it
- **Dataset**: TinyStories (same as autoresearch default — no new data download)
- **Metric**: tokens/sec at iso-quality (val_bpb within 2% of baseline)

**Deliverable**: `autoresearch/train.py`, `autoresearch/program.md`

---

## Phase 3 — AutoResearch Overnight Runs (Brev, 1–2 nights)

**Goal**: Let the agent autonomously discover the optimal draft model configuration,
running ~80–100 experiments and keeping only improvements.

### What the Agent Optimizes

Starting from the Phase 2 baseline, the agent modifies `train.py` to try:

- **Tier 1** (quick wins):
  - Tune γ (draft length 2 → 8)
  - Adjust acceptance temperature
  - Try greedy vs. sampled draft

- **Tier 2** (architecture):
  - Shrink draft model (fewer layers, smaller dim)
  - Shared embedding with target model
  - Distillation-tuned draft (pre-tuned to mimic target's distribution)

- **Tier 3** (HADES-specific):
  - Simulate Step 1 KV compression in the draft model path
  - Use token importance scores (Step 2) to skip speculative steps on high-entropy tokens
  - Model memory hierarchy latency (SRAM hit vs. eDRAM fallback)

### Success Criteria

The run proves the claim if we find a config where:
- `tokens_per_second ≥ 2× baseline` (proves 2× throughput)
- `acceptance_rate ≥ 0.70`
- `kv_memory_mb ≤ 8 MB` (fits in-SRAM at 16nm)
- `val_bpb` within 2% of baseline (quality preserved)

---

## Phase 4 — HADES Integration Report (Laptop, ~2 days)

**Goal**: Translate AutoResearch findings into concrete HADES v2 specs.

### Deliverables

1. **Speedup proof**: Table of acceptance rates found empirically, mapped to the analytic
   speedup formula from Phase 1. If α=0.78 found in Phase 3 → 2.73× proven.

2. **KV budget report**: Show that with Step 1 compression, both draft + main model KV
   fit in the 16nm on-chip SRAM budget (8–32 MB).

3. **Draft lane spec**: Based on best-found config, specify:
   - Draft model parameter count
   - Required SRAM allocation
   - γ (draft length)
   - Expected speedup range (with uncertainty from acceptance rate variance)

4. **Interaction with Token Importance Unit (Step 2)**: Whether attention scores predict
   token-level acceptance rates (if yes, Step 2 hardware can act as a speculative skip gate).

5. **Go/No-Go for v2 tapeout**: Concrete recommendation with supporting data.

---

## Repository Structure

```
decode-pipeline-optimizer/
├── RESEARCH_PLAN.md          ← this file
├── README.md                 ← overview (already exists)
├── analysis/
│   ├── speedup_model.py      ← Phase 1: analytic speedup curves
│   └── kv_budget.py          ← Phase 1: KV memory budget analysis
├── autoresearch/
│   ├── program.md            ← Phase 2: AutoResearch research directions
│   ├── train.py              ← Phase 2: speculative decode benchmark
│   └── prepare.py            ← Phase 2: dataset/model setup
└── results/
    ├── results.tsv           ← Phase 3: AutoResearch log (not committed)
    └── final_report.md       ← Phase 4: HADES v2 recommendation
```

---

## Timeline

| Phase | Where | Duration | Blocker |
|-------|-------|----------|---------|
| 1 — Analytical model | Laptop | 1 day | None |
| 2 — Simulation harness | Laptop | 2 days | Phase 1 done |
| 3 — AutoResearch runs | Brev | 1–2 nights | Phase 2 + Brev provisioned |
| 4 — Integration report | Laptop | 2 days | Phase 3 results |

**Total**: ~1 week to a publishable-quality proof.
