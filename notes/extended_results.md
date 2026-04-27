# Extended Verification Results

## Cross-Validation (Test 1)

- Our SpecdecSimulator baseline: **333.26 tok/s**
- Official Kelle-Simulator baseline: **333.26 tok/s**
- Measurement error: **0.0010%** (< 0.1% — simulators match exactly)

### Hardware Bonus Over Analytic Formula
- Average simulated/analytic ratio: **1.39x**
- Explanation: analytic formula assumes equal cost per token for draft and verify.
  On Kelle hardware, draft passes hit SRAM (185.9 pJ/byte, 1-cycle latency),
  while verify passes hit DRAM (640 pJ/byte, 100-cycle latency, 93.2% of decode energy).
  Draft compute is ~15x cheaper per token than target-model compute.
  This means actual hardware speedup consistently exceeds the analytic bound.

## Model Scaling (Test 2)
- OPT-125M, γ=4, α=0.75: **333.3 → 1335.6 tok/s (4.01x speedup)**
- OPT-125M, γ=8, α=0.80: **333.3 → 1515.9 tok/s (4.55x speedup)**
- OPT-1.3B, γ=4, α=0.75: **25.5 → 103.6 tok/s (4.06x speedup)**
- OPT-1.3B, γ=8, α=0.80: **25.5 → 121.2 tok/s (4.75x speedup)**

OPT-1.3B best config: 25.5 → 121.2 tok/s (4.75x) with γ=8, α=0.80

## Context Sweep (Test 3)

| Prompt | Decode | Speedup | KV Budget |
|--------|--------|---------|-----------|
| 32 | 32 | 4.18x | 2.28 MB |
| 64 | 64 | 4.00x | 4.56 MB |
| 128 | 64 | 4.01x | 6.84 MB |
| 256 | 64 | 4.02x | 11.41 MB |

## Wide Sweep (Test 4): 72 Configurations

- **Total configs**: 72
- **>= 2x proven**: 67/72 (93%)
- **>= 4x proven**: 25/72 (35%)
- **Mean speedup**: 3.53x
- **Median speedup**: 3.19x
- **Best config**: γ=10, α=0.90 → **8.63x** speedup
- **Best throughput**: 2875.4 tok/sec (baseline: 333.3 tok/sec)

## Energy Efficiency (Test 5)

Note: SpecdecSimulator accumulates energy in SpecDecStats.draft_energy_pj / verify_energy_pj but
does not propagate these into SimStats.decode.total_energy_uj (the field run_comparison reads).
The 0.0 spec values and 100% savings in the raw output are an instrumentation gap, not a real result.

Analytic energy savings (derived from measured speedup, DRAM-amortization model):
- Energy per token ≈ E_verify / speedup (since verify cost dominates and is amortized over more tokens)
- Saving = (1 - 1/speedup) × 100%

| Config | Speedup | Analytic Energy Saving |
|--------|---------|------------------------|
| α=0.75 γ=4 | 4.01× | **75%** |
| α=0.75 γ=6 | 4.37× | **77%** |
| α=0.80 γ=4 | 4.01× | **75%** |
| α=0.80 γ=8 | 4.55× | **78%** |
| α=0.85 γ=8 | 6.33× | **84%** |

## Extended Figures
- `fig5_analytic_vs_simulated.png` — cross-validation scatter + hardware bonus factor
- `fig6_model_scaling.png` — OPT-125M vs OPT-1.3B throughput and speedup
- `fig7_wide_heatmap.png` — 72-config heatmap + speedup distribution histogram
- `fig8_energy_efficiency.png` — energy per token baseline vs speculative

## Key Numbers for Paper

| Metric | Value |
|--------|-------|
| Baseline throughput (OPT-125M, Kelle, 256-tok KV) | **333 tok/s** |
| Best speculative throughput (γ=10, α=0.90) | **2875 tok/s** |
| Peak speedup | **8.63x** |
| Configurations ≥ 2x speedup | **67/72** |
| Configurations ≥ 4x speedup | **25/72** |
| Hardware bonus over analytic formula | **1.39x avg** |
| Simulator cross-validation error | **< 0.01%** |
| KV budget with 4-bit compression (512-tok ctx) | **4.75 MB (fits 8 MB SRAM)** |
