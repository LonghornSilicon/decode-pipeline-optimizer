"""
Prepare environment for HADES speculative decoding autoresearch.
Downloads and caches GPT-2 models. Do NOT modify.
Run once before starting the autoresearch loop.
"""

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

print("Downloading GPT-2 tokenizer...")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
print(f"  Vocab size: {tok.vocab_size}")

print("Downloading GPT-2 medium (target model)...")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Parameters: {params:.0f}M")
del model

print("Verifying torch + CUDA:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nSetup complete. Run `python train.py` to test baseline.")
print("Run `python train.py --quick` for a fast CPU smoke test.")
