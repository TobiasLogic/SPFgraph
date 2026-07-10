"""
Smoke test: verify the inference.py loader handles all expected checkpoint formats.

Synthesizes tiny checkpoints in each format, loads them, and runs a forward
pass. Doesn't validate output quality (the weights are random) — only that the
load path completes and produces a tensor of the right shape.

Run:
    cd python
    python test_load.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference

def _expect_load(path: str, label: str, vocab_size: int):
    """Load a checkpoint via inference.load_checkpoint and run a forward pass."""
    print(f"\n[{label}]  {path}")
    try:
        model, cfg, device, n_params, missing, unexpected, arch, _ = inference.load_checkpoint(
            path, fp16=False, force_cpu=True, ctx_override=None,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        traceback.print_exc()
        return False

    print(f"  arch={arch}  params={n_params:,}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape[-1] == vocab_size, f"expected vocab_size={vocab_size}, got {logits.shape[-1]}"
    assert torch.isfinite(logits).all(), "non-finite logits — broken load"
    print(f"  forward OK: logits {tuple(logits.shape)}, finite={torch.isfinite(logits).all().item()}")
    return True

def make_gpt2_ckpt(path: str):
    """Standard GPT-2 sph format: keys like transformer.wte, transformer.h.0.attn.c_attn."""
    cfg = inference.GPTConfig(
        block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16,
    )
    model = inference.GPT(cfg)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, path)

def make_llama_sph_ckpt(path: str):
    """Standard sph LLaMA format: embed_tokens / input_layernorm / self_attn / etc.
    Untied lm_head, interleaved RoPE."""
    cfg = inference.LlamaConfig(
        block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16,
        num_key_value_heads=1, intermediate_size=32,
        rope_layout="interleaved", tie_word_embeddings=False,
    )
    model = inference.LlamaModel(cfg)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, path)

def make_llama_compact_ckpt(path: str):
    """ZeroShot trainer format: tok_emb / attn_norm / mlp_norm / attn.q_proj.
    Tied embedding (no separate lm_head), split-halves RoPE."""
    cfg = inference.LlamaConfig(
        block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16,
        num_key_value_heads=1, intermediate_size=32,
        rope_layout="split_halves", tie_word_embeddings=True,
    )
    model = inference.LlamaModel(cfg)
    sd = model.state_dict()

    head_dim = cfg.n_embd // cfg.n_head
    n_kv = cfg.num_key_value_heads or cfg.n_head

    def inv_permute(w, n_h, hd):
        return w.view(n_h, hd // 2, 2, -1).transpose(1, 2).contiguous().view(n_h * hd, -1)

    for i in range(cfg.n_layer):
        qk = f"layers.{i}.self_attn.q_proj.weight"
        kk = f"layers.{i}.self_attn.k_proj.weight"
        sd[qk] = inv_permute(sd[qk], cfg.n_head, head_dim)
        sd[kk] = inv_permute(sd[kk], n_kv, head_dim)

    compact = {}
    compact["tok_emb.weight"] = sd["embed_tokens.weight"]
    compact["norm.weight"] = sd["norm.weight"]
    for i in range(cfg.n_layer):
        p = f"layers.{i}"
        compact[f"{p}.attn_norm.weight"]       = sd[f"{p}.input_layernorm.weight"]
        compact[f"{p}.mlp_norm.weight"]        = sd[f"{p}.post_attention_layernorm.weight"]
        compact[f"{p}.attn.q_proj.weight"]     = sd[f"{p}.self_attn.q_proj.weight"]
        compact[f"{p}.attn.k_proj.weight"]     = sd[f"{p}.self_attn.k_proj.weight"]
        compact[f"{p}.attn.v_proj.weight"]     = sd[f"{p}.self_attn.v_proj.weight"]
        compact[f"{p}.attn.o_proj.weight"]     = sd[f"{p}.self_attn.o_proj.weight"]
        compact[f"{p}.mlp.gate_proj.weight"]   = sd[f"{p}.mlp.gate_proj.weight"]
        compact[f"{p}.mlp.up_proj.weight"]     = sd[f"{p}.mlp.up_proj.weight"]
        compact[f"{p}.mlp.down_proj.weight"]   = sd[f"{p}.mlp.down_proj.weight"]

    compact_cfg = {
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
        "num_key_value_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size,
        "block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
        "rope_theta": cfg.rope_theta, "norm_eps": cfg.norm_eps,
        "arch": "llama",
    }
    torch.save({"model": compact, "config": compact_cfg}, path)

def make_llama_hf_prefix_ckpt(path: str):
    """HF transformers format: keys prefixed with `model.` (e.g. model.embed_tokens)."""
    cfg = inference.LlamaConfig(
        block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16,
        num_key_value_heads=1, intermediate_size=32,
        rope_layout="interleaved", tie_word_embeddings=False,
    )
    model = inference.LlamaModel(cfg)
    sd = model.state_dict()
    prefixed = {f"model.{k}" if not k.startswith("lm_head") else k: v for k, v in sd.items()}
    torch.save({"model": prefixed, "config": asdict(cfg)}, path)

def make_compiled_save_ckpt(path: str):
    """Simulate a torch.compile()-wrapped save: keys prefixed with `_orig_mod.`."""
    cfg = inference.GPTConfig(block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16)
    model = inference.GPT(cfg)
    sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    torch.save({"model": sd, "config": asdict(cfg)}, path)

def main():
    cases = [
        ("gpt2 (standard)",          make_gpt2_ckpt,          128),
        ("llama (sph standard)",     make_llama_sph_ckpt,     128),
        ("llama (compact ZeroShot trainer)", make_llama_compact_ckpt, 128),
        ("llama (HF transformers `model.` prefix)", make_llama_hf_prefix_ckpt, 128),
        ("gpt2 (saved while torch.compile-wrapped)", make_compiled_save_ckpt, 128),
    ]

    results = []
    with tempfile.TemporaryDirectory() as td:
        for label, factory, vocab in cases:
            path = os.path.join(td, label.replace(" ", "_").replace("(", "").replace(")", "") + ".pt")
            torch.manual_seed(0)
            factory(path)
            ok = _expect_load(path, label, vocab)
            results.append((label, ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    failed = 0
    for label, ok in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}]  {label}")
        if not ok:
            failed += 1
    if failed:
        print(f"\n{failed}/{len(results)} cases failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} cases passed.")

if __name__ == "__main__":
    main()
