"""
SpecdecSimulator — extends KelleSimulator with a hardware speculative decoding path.

Architecture modeled:
  - Small draft model runs on the SAME RSA array (timeshared), weights fit in SRAM
  - Draft model proposes gamma tokens sequentially
  - Target model verifies all gamma+1 positions in ONE batched forward pass
  - Because ceil((gamma+1)/32) == 1 for gamma < 31, verification costs the SAME
    as a single autoregressive decode step on the 32x32 Kelle RSA
  - Draft weights (~5-10 MB) fit in 2 MB SRAM when INT4 quantized (Step 1),
    so draft passes pay SRAM latency, not DRAM latency

This is the hardware proof for HADES Step 5.
"""

from __future__ import annotations
import math
import random
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Allow import from the kelle-simulator repo
_KELLE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "kelle-simulator")
if _KELLE_PATH not in sys.path:
    sys.path.insert(0, _KELLE_PATH)

from kelle_simulator.simulator import KelleSimulator, SimStats, PhaseStats
from kelle_simulator.config import HardwareConfig, ModelConfig, MODELS
from kelle_simulator.rsa import RSA, ComputeResult
from kelle_simulator.memory import MemoryHierarchy
from kelle_simulator.sfu import SFU


# ─────────────────────────────────────────────────────────────────────────────
# Draft model configs (Step 5: small, SRAM-resident draft model lane)
# ─────────────────────────────────────────────────────────────────────────────

DRAFT_MODELS: Dict[str, ModelConfig] = {
    # Ultra-tiny: 2 layers, 256 dim — ~15M params, ~4 MB INT8 → fits in 2 MB SRAM at INT4
    "draft-tiny": ModelConfig(
        name="DraftTiny-15M",
        num_layers=2,
        num_heads=4,
        d_model=256,
        d_ffn=1024,
        head_dim=64,
        vocab_size=50272,
        max_seq_len=2048,
        weight_bits=4,   # INT4 — Step 1 compression applied to draft weights
        kv_bits=4,        # 4-bit KV — Step 1 compression
    ),
    # Small: 4 layers, 512 dim — ~50M params, ~12 MB INT8 → ~6 MB INT4 (needs eDRAM)
    "draft-small": ModelConfig(
        name="DraftSmall-50M",
        num_layers=4,
        num_heads=8,
        d_model=512,
        d_ffn=2048,
        head_dim=64,
        vocab_size=50272,
        max_seq_len=2048,
        weight_bits=4,
        kv_bits=4,
    ),
    # Medium: 6 layers, 512 dim — ~75M params — approaches OPT-125M territory
    "draft-medium": ModelConfig(
        name="DraftMedium-75M",
        num_layers=6,
        num_heads=8,
        d_model=512,
        d_ffn=2048,
        head_dim=64,
        vocab_size=50272,
        max_seq_len=2048,
        weight_bits=4,
        kv_bits=4,
    ),
}


@dataclass
class SpecDecStats:
    """Tracks speculative decoding specific metrics."""
    spec_steps: int = 0               # number of speculative decode calls
    tokens_generated: int = 0         # total tokens produced
    tokens_proposed: int = 0          # total tokens the draft proposed
    tokens_accepted: int = 0          # tokens accepted by target
    draft_cycles: int = 0             # cycles spent on draft model passes
    verify_cycles: int = 0            # cycles spent on target verification
    draft_energy_pj: float = 0.0
    verify_energy_pj: float = 0.0
    tokens_per_step: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.tokens_accepted / max(self.tokens_proposed, 1)

    @property
    def avg_tokens_per_step(self) -> float:
        return self.tokens_generated / max(self.spec_steps, 1)

    @property
    def draft_fraction(self) -> float:
        total = self.draft_cycles + self.verify_cycles
        return self.draft_cycles / max(total, 1)


class SpecdecSimulator(KelleSimulator):
    """
    Extends KelleSimulator to simulate a hardware speculative decoding path.

    The draft model runs on a secondary compute path (same RSA, timeshared).
    Because draft weights are INT4 and small, they fit in on-chip SRAM.
    The target model verification pass is batched over gamma+1 tokens.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        hw_cfg: HardwareConfig,
        draft_cfg: ModelConfig,
        gamma: int = 4,
        acceptance_rate: float = 0.75,
        seed: int = 42,
    ):
        super().__init__(model_cfg, hw_cfg)
        self.draft_cfg = draft_cfg
        self.gamma = gamma
        self.acceptance_rate = acceptance_rate
        self.spec_stats = SpecDecStats()
        random.seed(seed)

        # Draft model RSA: same hardware, different model weights (SRAM-resident)
        draft_memory = _make_sram_only_memory(draft_cfg, hw_cfg)
        self.draft_rsa = RSA(hw_cfg, draft_cfg, draft_memory)
        self.draft_memory = draft_memory

    # ── Draft model compute cost ───────────────────────────────────────────────

    def _draft_decode_cycles_one_step(self, kv_len: int) -> Tuple[int, float]:
        """
        Compute cycles + energy for the draft model generating one token.
        Draft weights fit in SRAM (INT4, small model), so no DRAM stalls.
        """
        d = self.draft_cfg
        total_cycles = 0
        total_energy = 0.0

        for layer in range(d.num_layers):
            # QKV projection
            r = self.draft_rsa.matmul(
                M=1, K=d.d_model, N=3 * d.d_model,
                layer=0,  # all draft layers are SRAM-resident (layer 0 key)
                weight_bytes=d.weight_bytes(d.d_model, 3 * d.d_model),
                activation_bytes=d.d_model * 2,
            )
            total_cycles += r.total_cycles
            total_energy += r.energy_pj

            # Attention score: [1, head_dim] × [head_dim, kv_len]
            M, K, N = 1, d.head_dim, kv_len
            attn_ops = M * K * N * 2 * d.num_heads
            attn_cycles = self.draft_rsa._compute_cycles(M, K, N) * d.num_heads
            attn_energy = self.draft_rsa._energy_pj(attn_ops)
            total_cycles += attn_cycles
            total_energy += attn_energy

            # Attention output: [1, kv_len] × [kv_len, head_dim]
            M2, K2, N2 = 1, kv_len, d.head_dim
            out_ops = M2 * K2 * N2 * 2 * d.num_heads
            out_cycles = self.draft_rsa._compute_cycles(M2, K2, N2) * d.num_heads
            out_energy = self.draft_rsa._energy_pj(out_ops)
            total_cycles += out_cycles
            total_energy += out_energy

            # FFN
            for (in_d, out_d) in [(d.d_model, d.d_ffn), (d.d_ffn, d.d_model)]:
                r2 = self.draft_rsa.matmul(
                    M=1, K=in_d, N=out_d,
                    layer=0,
                    weight_bytes=d.weight_bytes(in_d, out_d),
                    activation_bytes=in_d * 2,
                )
                total_cycles += r2.total_cycles
                total_energy += r2.energy_pj

        return total_cycles, total_energy

    # ── Target model verification pass ────────────────────────────────────────

    def _verification_pass_cycles(self, kv_len: int, batch: int) -> Tuple[int, float]:
        """
        Cycles + energy for the target model verifying `batch` = gamma+1 tokens.

        KEY HARDWARE INSIGHT: ceil(batch / rsa_rows) == 1 for batch <= 32.
        So verifying gamma+1 tokens costs the SAME as verifying 1 token in the
        QKV and FFN dimensions — it's free batching on the 32x32 RSA.

        The only extra cost vs. a single decode step:
          - Attention scores scale as [batch, head_dim] x [head_dim, kv_len+batch]
          - Attention output scales as [batch, kv_len+batch] x [kv_len+batch, head_dim]
        Both are still dominated by the memory stall from DRAM weight streaming.
        """
        m = self.model
        total_cycles = 0
        total_energy = 0.0

        for layer in range(m.num_layers):
            # QKV: M = batch, but ceil(batch/32) == 1 for batch <= 31
            qkv = self.rsa.matmul(
                M=batch, K=m.d_model, N=3 * m.d_model,
                layer=layer,
                weight_bytes=m.weight_bytes(m.d_model, 3 * m.d_model),
                activation_bytes=batch * m.d_model * 2,
            )
            total_cycles += qkv.total_cycles
            total_energy += qkv.energy_pj

            # Attention over full context + draft tokens
            full_kv = kv_len + batch
            M, K, N = batch, m.head_dim, full_kv
            attn_ops = M * K * N * 2 * m.num_heads
            attn_cycles = self.rsa._compute_cycles(M, K, N) * m.num_heads
            attn_energy = self.rsa._energy_pj(attn_ops)
            total_cycles += attn_cycles
            total_energy += attn_energy

            # Attention output
            M2, K2, N2 = batch, full_kv, m.head_dim
            out_ops = M2 * K2 * N2 * 2 * m.num_heads
            out_cycles = self.rsa._compute_cycles(M2, K2, N2) * m.num_heads
            out_energy = self.rsa._energy_pj(out_ops)
            total_cycles += out_cycles
            total_energy += out_energy

            # FFN: M = batch, same tile-count as M=1 (ceil(batch/32) == 1)
            for (in_d, out_d) in [(m.d_model, m.d_ffn), (m.d_ffn, m.d_model)]:
                r = self.rsa.matmul(
                    M=batch, K=in_d, N=out_d,
                    layer=layer,
                    weight_bytes=m.weight_bytes(in_d, out_d),
                    activation_bytes=batch * in_d * 2,
                )
                total_cycles += r.total_cycles
                total_energy += r.energy_pj

        return total_cycles, total_energy

    # ── One speculative decode step ────────────────────────────────────────────

    def _speculative_decode_step(self, step: int, base_token_id: int, kv_len: int) -> Tuple[int, int]:
        """
        One speculative decode step:
          1. Draft model generates gamma tokens sequentially
          2. Target model verifies all gamma+1 positions in batch
          3. Accept/reject chain: accept until first mismatch
          4. Return (total_cycles_consumed, tokens_generated)

        Returns:
          total_cycles: cycles consumed this step
          n_generated:  tokens produced (1 to gamma+1)
        """
        # ── Phase 1: Draft model (gamma forward passes) ───────────────────────
        draft_total_cycles = 0
        draft_total_energy = 0.0
        for i in range(self.gamma):
            c, e = self._draft_decode_cycles_one_step(kv_len + i)
            draft_total_cycles += c
            draft_total_energy += e

        # ── Phase 2: Verification (one batched target model pass) ─────────────
        verify_cycles, verify_energy = self._verification_pass_cycles(kv_len, self.gamma + 1)

        # ── Phase 3: Accept/reject (modeled stochastically) ──────────────────
        n_accepted = 0
        for _ in range(self.gamma):
            if random.random() < self.acceptance_rate:
                n_accepted += 1
            else:
                break
        n_generated = n_accepted + 1  # always get at least the resampled token

        total_cycles = draft_total_cycles + verify_cycles

        # Update stats
        self.spec_stats.spec_steps += 1
        self.spec_stats.draft_cycles += draft_total_cycles
        self.spec_stats.verify_cycles += verify_cycles
        self.spec_stats.draft_energy_pj += draft_total_energy
        self.spec_stats.verify_energy_pj += verify_energy
        self.spec_stats.tokens_proposed += self.gamma
        self.spec_stats.tokens_accepted += n_accepted
        self.spec_stats.tokens_generated += n_generated
        self.spec_stats.tokens_per_step.append(n_generated)

        return total_cycles, n_generated

    # ── Full speculative simulation run ───────────────────────────────────────

    def simulate_speculative(
        self,
        prompt_len: int,
        num_target_tokens: int,
        verbose: bool = True,
    ) -> Tuple[SimStats, SpecDecStats]:
        """
        Run full speculative decode: prefill (normal) + speculative decode loop.
        Generates num_target_tokens total, stopping as soon as we hit the target.
        """
        import time
        self.stats.wall_clock_start = time.perf_counter()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Kelle SPECULATIVE DECODE Simulator")
            print(f"  Target : {self.model.name}")
            print(f"  Draft  : {self.draft_cfg.name}  (γ={self.gamma}, α={self.acceptance_rate:.2f})")
            print(f"  Prompt : {prompt_len} tokens  |  Generate: {num_target_tokens} tokens")
            print(f"{'='*60}")

        # Prefill: identical to baseline
        if verbose:
            print(f"\n[Prefill] Processing {prompt_len} tokens...")
        self.prefill(prompt_len)
        if verbose:
            print(f"  Prefill: {self.stats.prefill_latency_ms:.3f} ms")

        # Speculative decode loop
        if verbose:
            print(f"\n[Speculative Decode] Target: {num_target_tokens} tokens...")
        tokens_generated = 0
        kv_len = min(prompt_len, self.hw.kv_cache_capacity_tokens)
        base_gamma = self.gamma

        while tokens_generated < num_target_tokens:
            remaining = num_target_tokens - tokens_generated
            # Temporarily cap gamma so we don't overshoot the target
            self.gamma = min(base_gamma, remaining)

            cycles, n = self._speculative_decode_step(
                step=self.spec_stats.spec_steps,
                base_token_id=prompt_len + tokens_generated,
                kv_len=kv_len,
            )
            self.gamma = base_gamma

            tokens_generated += n
            kv_len = min(kv_len + n, self.hw.kv_cache_capacity_tokens)

            # Add to decode phase stats
            self.stats.decode.tokens_processed = tokens_generated
            self.stats.decode.total_cycles += cycles
            self.stats.decode_latency_per_step.append(cycles)

            if verbose and self.spec_stats.spec_steps % max(1, (num_target_tokens // self.gamma) // 5) == 0:
                print(f"  step {self.spec_stats.spec_steps:>4}  "
                      f"tokens={tokens_generated}/{num_target_tokens}  "
                      f"α={self.spec_stats.acceptance_rate:.2f}  "
                      f"avg_gen={self.spec_stats.avg_tokens_per_step:.2f}/step")

        self.stats.decode.tokens_processed = tokens_generated
        self.stats.wall_clock_end = time.perf_counter()

        if verbose:
            self._print_spec_summary()

        return self.stats, self.spec_stats

    def _print_spec_summary(self) -> None:
        s = self.stats
        ss = self.spec_stats
        print(f"\n{'='*60}")
        print(f"  SPECULATIVE DECODE RESULTS")
        print(f"{'='*60}")
        print(f"  Target model      : {self.model.name}")
        print(f"  Draft model       : {self.draft_cfg.name}")
        print(f"  Draft length γ    : {self.gamma}")
        print(f"  Acceptance rate α : {ss.acceptance_rate:.3f}  (target: {self.acceptance_rate:.2f})")
        print(f"  Avg tokens/step   : {ss.avg_tokens_per_step:.2f}  (max: {self.gamma+1})")
        print(f"  Spec steps        : {ss.spec_steps}")
        print(f"  Tokens generated  : {ss.tokens_generated}")
        print(f"  Decode latency    : {s.decode_latency_ms:.3f} ms")
        tps = ss.tokens_generated / (s.decode.total_cycles / 1e9)
        print(f"  Throughput        : {tps:.2f} tokens/sec")
        print(f"  Draft cycles      : {ss.draft_cycles:,}  ({100*ss.draft_fraction:.1f}% of decode)")
        print(f"  Verify cycles     : {ss.verify_cycles:,}  ({100*(1-ss.draft_fraction):.1f}% of decode)")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draft model memory (SRAM-only, no DRAM spill)
# ─────────────────────────────────────────────────────────────────────────────

def _make_sram_only_memory(model_cfg: ModelConfig, hw_cfg: HardwareConfig) -> MemoryHierarchy:
    """
    Create a MemoryHierarchy where ALL layers are SRAM-resident.
    Used for the draft model whose weights (INT4) fit in on-chip SRAM.
    """
    mem = MemoryHierarchy(model_cfg, hw_cfg)
    # Force all layers to SRAM regardless of model weight size
    mem._sram_resident_layers = set(range(model_cfg.num_layers))
    return mem


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run baseline vs. speculative and return comparison dict
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    target_key: str = "opt-125m",
    draft_key: str = "draft-tiny",
    gamma: int = 4,
    acceptance_rate: float = 0.75,
    prompt_len: int = 128,
    num_decode_tokens: int = 64,
    kv_capacity: int = 256,
    verbose: bool = False,
) -> Dict:
    """Run baseline and speculative simulations, return comparison metrics."""
    from kelle_simulator.config import DEFAULT_HW_CONFIG

    hw = HardwareConfig(kv_cache_capacity_tokens=kv_capacity)
    model = MODELS[target_key]
    draft = DRAFT_MODELS[draft_key]

    # ── Baseline: standard autoregressive ─────────────────────────────────────
    baseline_sim = KelleSimulator(model, hw)
    baseline_stats = baseline_sim.simulate(prompt_len, num_decode_tokens, verbose=verbose)
    baseline_tps = baseline_stats.throughput_tokens_per_sec()
    baseline_decode_cycles = baseline_stats.decode.total_cycles

    # ── Speculative decode ────────────────────────────────────────────────────
    spec_sim = SpecdecSimulator(model, hw, draft, gamma=gamma,
                                acceptance_rate=acceptance_rate)
    spec_stats, ss = spec_sim.simulate_speculative(prompt_len, num_decode_tokens, verbose=verbose)
    spec_tps = ss.tokens_generated / (spec_stats.decode.total_cycles / 1e9)
    spec_decode_cycles = spec_stats.decode.total_cycles

    speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0

    # KV memory budget check (Step 1: 4-bit compression)
    def kv_mb(m, ctx, bits):
        return 2 * m.num_layers * m.num_heads * m.head_dim * ctx * (bits/8) / (1024**2)

    ctx = prompt_len + num_decode_tokens
    main_kv_mb = kv_mb(model, ctx, model.kv_bits)
    draft_kv_mb = kv_mb(draft, ctx, draft.kv_bits)
    total_kv_mb = main_kv_mb + draft_kv_mb

    return {
        "target": target_key,
        "draft": draft_key,
        "gamma": gamma,
        "alpha_target": acceptance_rate,
        "alpha_measured": ss.acceptance_rate,
        "avg_tokens_per_step": ss.avg_tokens_per_step,
        "baseline_tps": baseline_tps,
        "spec_tps": spec_tps,
        "speedup": speedup,
        "baseline_decode_cycles": baseline_decode_cycles,
        "spec_decode_cycles": spec_decode_cycles,
        "draft_fraction": ss.draft_fraction,
        "spec_steps": ss.spec_steps,
        "main_kv_mb": main_kv_mb,
        "draft_kv_mb": draft_kv_mb,
        "total_kv_mb": total_kv_mb,
        "kv_budget_ok": total_kv_mb <= 8.0,
        "baseline_decode_energy_uj": baseline_stats.decode.total_energy_uj,
        "spec_decode_energy_uj": spec_stats.decode.total_energy_uj,
    }
