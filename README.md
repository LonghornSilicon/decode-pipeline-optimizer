# Decode Pipeline Optimizer — Speculative Decoding Assist

**HADES Step 5 | Optional v1, Target v2**

## What It Is

The Decode Pipeline Optimizer implements speculative decoding at the hardware level. Instead of generating one token at a time (sequential, memory-bandwidth-bound), a small draft model runs in parallel guessing the next N tokens. The main model then verifies all guesses in a single batched pass — effectively yielding **2–4× throughput** when the draft model's accuracy is high.

## Hardware Architecture

| Component | Description |
|---|---|
| Draft Model Lane | Dedicated hardware thread / secondary compute path running the small draft model |
| Draft KV Cache | Small, separate KV cache for speculative token state |
| Merge Unit | Reconciles speculative tokens with main model output; flushes on misprediction |
| Verification Path | Batched verification pass on the main model accelerator |

## Why It's Optional for v1

Speculative decoding doubles control complexity:

- Two models must be managed simultaneously
- Mispredictions require speculative state to be flushed and rolled back
- Draft model sizing must be tuned per workload to maximize acceptance rate

The v1 tapeout wisely defers this. At **16nm**, the transistor budget is sufficient to include a draft-model lane in v2 without significant area penalty.

## Target Metrics (v2)

- **Throughput gain**: 2–4× decode tokens/sec at iso-quality
- **Acceptance rate target**: ≥ 70% speculative token acceptance
- **Draft model size**: ~100M–1B parameters (fits in dedicated SRAM lane)
- **Flush latency**: < 1 decode cycle penalty on misprediction

## Relationship to HADES

This module is the fifth and final component of the HADES architecture plan. It sits at the end of the decode pipeline and is gated on successful v1 silicon validation of steps 1–4.

## Status

- [ ] Draft model lane microarchitecture
- [ ] KV cache sizing and SRAM allocation
- [ ] Merge unit logic (accept / flush / rollback)
- [ ] Verification batching protocol
- [ ] Acceptance rate simulation
- [ ] RTL implementation
- [ ] Integration with main decode pipeline
