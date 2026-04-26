"""
Phase 1: KV cache memory budget analysis for HADES at 16nm.
Proves that Step 1 compression (4-bit KV) makes dual-model (draft + main) fit in 8-32 MB SRAM.
Run on laptop (CPU only).
"""

import numpy as np
import matplotlib.pyplot as plt


def kv_size_mb(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    context_len: int,
    bits: int,
) -> float:
    """KV cache size in MB for one model."""
    bytes_per_entry = bits / 8
    # K and V, both stored
    total_bytes = 2 * num_layers * num_heads * head_dim * context_len * bytes_per_entry
    return total_bytes / (1024 ** 2)


MODELS = {
    # (name, layers, heads, head_dim, param_count_M)
    "GPT-2 small (117M)":   (12, 12, 64,  117),
    "GPT-2 medium (345M)":  (24, 16, 64,  345),
    "GPT-2 large (774M)":   (36, 20, 64,  774),
    "Llama-68M":            (4,  8,  64,   68),
    "Llama-160M":           (12, 12, 64,  160),
    "Llama-7B":             (32, 32, 128, 7000),
    "Llama-13B":            (40, 40, 128, 13000),
    "Custom draft 100M":    (8,  8,  64,  100),
}

HADES_SRAM_MB = 8   # conservative 16nm SRAM budget
HADES_SRAM_MAX_MB = 32  # generous 16nm budget


def print_budget_table(context_len: int = 2048):
    print(f"\n=== KV Cache Budget at context={context_len} tokens ===")
    print(f"{'Model':<26} {'Params':>8} {'16-bit MB':>12} {'8-bit MB':>10} {'4-bit MB':>10} {'2-bit MB':>10}")
    print("-" * 80)
    for name, (layers, heads, hdim, params) in MODELS.items():
        sizes = {b: kv_size_mb(layers, heads, hdim, context_len, b) for b in [16, 8, 4, 2]}
        print(f"{name:<26} {params:>7}M {sizes[16]:>11.2f} {sizes[8]:>10.2f} {sizes[4]:>10.2f} {sizes[2]:>10.2f}")


def dual_model_budget(
    draft_name: str,
    main_name: str,
    context_len: int = 2048,
    compression_bits: int = 4,
):
    """Compute combined KV footprint for draft + main model with Step 1 compression."""
    draft = MODELS[draft_name]
    main = MODELS[main_name]

    draft_kv = kv_size_mb(draft[0], draft[1], draft[2], context_len, compression_bits)
    main_kv = kv_size_mb(main[0], main[1], main[2], context_len, compression_bits)
    total = draft_kv + main_kv

    fits_conservative = total <= HADES_SRAM_MB
    fits_generous = total <= HADES_SRAM_MAX_MB

    print(f"\n  Draft: {draft_name}")
    print(f"  Main:  {main_name}")
    print(f"  Context: {context_len} tokens | Compression: {compression_bits}-bit (Step 1)")
    print(f"  Draft KV: {draft_kv:.2f} MB | Main KV: {main_kv:.2f} MB | Total: {total:.2f} MB")
    print(f"  Fits in 8MB SRAM:  {'YES ✓' if fits_conservative else 'NO — needs eDRAM'}")
    print(f"  Fits in 32MB SRAM: {'YES ✓' if fits_generous else 'NO — needs off-chip'}")
    return total


def plot_budget_vs_context():
    context_lengths = np.arange(256, 4097, 128)
    pairs = [
        ("Custom draft 100M", "GPT-2 medium (345M)", 4),
        ("Llama-68M", "Llama-7B", 4),
        ("GPT-2 small (117M)", "GPT-2 medium (345M)", 4),
        ("Llama-68M", "Llama-7B", 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for draft_name, main_name, bits in pairs:
        draft = MODELS[draft_name]
        main = MODELS[main_name]
        totals = [
            kv_size_mb(draft[0], draft[1], draft[2], c, bits) +
            kv_size_mb(main[0], main[1], main[2], c, bits)
            for c in context_lengths
        ]
        label = f"{draft_name.split('(')[0].strip()} + {main_name.split('(')[0].strip()} ({bits}-bit)"
        ax.plot(context_lengths, totals, linewidth=2, label=label)

    ax.axhline(HADES_SRAM_MB, color="red", linestyle="--", linewidth=1.5, label="8 MB SRAM (conservative)")
    ax.axhline(HADES_SRAM_MAX_MB, color="orange", linestyle="--", linewidth=1.5, label="32 MB SRAM (generous)")
    ax.fill_between(context_lengths, 0, HADES_SRAM_MB, alpha=0.08, color="green")
    ax.fill_between(context_lengths, HADES_SRAM_MB, HADES_SRAM_MAX_MB, alpha=0.05, color="yellow")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Combined KV Cache (MB)", fontsize=12)
    ax.set_title("HADES: Dual-Model KV Budget vs Context Length\n(Step 1 compression applied to both models)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig("analysis/kv_budget.png", dpi=150, bbox_inches="tight")
    print("\nSaved: analysis/kv_budget.png")
    plt.show()


if __name__ == "__main__":
    print("=== HADES KV Cache Memory Budget Analysis ===")
    print_budget_table(context_len=2048)

    print("\n--- Dual-Model Budget (draft + main, Step 1 compression) ---")
    dual_model_budget("Custom draft 100M", "GPT-2 medium (345M)", 2048, bits=4)
    dual_model_budget("Llama-68M", "Llama-7B", 2048, bits=4)
    dual_model_budget("GPT-2 small (117M)", "GPT-2 medium (345M)", 2048, bits=4)

    print("\n--- Without Step 1 (16-bit, to show why Steps 1+5 are coupled) ---")
    dual_model_budget("Custom draft 100M", "GPT-2 medium (345M)", 2048, bits=16)

    plot_budget_vs_context()
