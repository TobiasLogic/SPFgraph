"""
End-to-end numerical correctness test.

Builds a sph-format LLaMA model with random weights, runs a forward pass to
get a reference output, then converts the weights to the compact ZeroShot
training format, saves a checkpoint, loads it through the patched
inference.load_checkpoint, runs the same forward pass, and asserts the
outputs match.

If this passes, the entire pipeline (key remap, RoPE permutation, weight
tying) is mathematically equivalent to a hand-converted checkpoint loaded
into the standard sph LLaMA path.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference

def test_compact_format_round_trip():
    torch.manual_seed(42)

    cfg_ref = inference.LlamaConfig(
        block_size=128, vocab_size=256,
        n_layer=3, n_head=4, n_embd=64,
        num_key_value_heads=2, intermediate_size=128,
        rope_theta=10000.0, norm_eps=1e-5,
        rope_layout="interleaved", tie_word_embeddings=False,
    )
    ref_model = inference.LlamaModel(cfg_ref)
    ref_model.eval()

    with torch.no_grad():
        ref_model.lm_head.weight.copy_(ref_model.embed_tokens.weight)

    idx = torch.randint(0, cfg_ref.vocab_size, (2, 17))
    with torch.no_grad():
        ref_logits = ref_model(idx)

    sd = ref_model.state_dict()
    head_dim = cfg_ref.n_embd // cfg_ref.n_head
    n_kv = cfg_ref.num_key_value_heads

    def interleaved_to_split_halves(w, n_h, hd):
        return w.view(n_h, hd // 2, 2, -1).transpose(1, 2).contiguous().view(n_h * hd, -1)

    compact = {
        "tok_emb.weight": sd["embed_tokens.weight"],
        "norm.weight": sd["norm.weight"],
    }
    for i in range(cfg_ref.n_layer):
        p = f"layers.{i}"
        compact[f"{p}.attn_norm.weight"] = sd[f"{p}.input_layernorm.weight"]
        compact[f"{p}.mlp_norm.weight"]  = sd[f"{p}.post_attention_layernorm.weight"]
        compact[f"{p}.attn.q_proj.weight"] = interleaved_to_split_halves(
            sd[f"{p}.self_attn.q_proj.weight"], cfg_ref.n_head, head_dim
        )
        compact[f"{p}.attn.k_proj.weight"] = interleaved_to_split_halves(
            sd[f"{p}.self_attn.k_proj.weight"], n_kv, head_dim
        )
        compact[f"{p}.attn.v_proj.weight"] = sd[f"{p}.self_attn.v_proj.weight"]
        compact[f"{p}.attn.o_proj.weight"] = sd[f"{p}.self_attn.o_proj.weight"]
        compact[f"{p}.mlp.gate_proj.weight"] = sd[f"{p}.mlp.gate_proj.weight"]
        compact[f"{p}.mlp.up_proj.weight"]   = sd[f"{p}.mlp.up_proj.weight"]
        compact[f"{p}.mlp.down_proj.weight"] = sd[f"{p}.mlp.down_proj.weight"]

    compact_cfg = {
        "arch": "llama",
        "n_layer": cfg_ref.n_layer, "n_head": cfg_ref.n_head, "n_embd": cfg_ref.n_embd,
        "num_key_value_heads": cfg_ref.num_key_value_heads,
        "intermediate_size": cfg_ref.intermediate_size,
        "block_size": cfg_ref.block_size, "vocab_size": cfg_ref.vocab_size,
        "rope_theta": cfg_ref.rope_theta, "norm_eps": cfg_ref.norm_eps,
    }

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "compact.pt")
        torch.save({"model": compact, "config": compact_cfg}, path)

        loaded_model, _, _, _, _, _, arch, _ = inference.load_checkpoint(
            path, fp16=False, force_cpu=True, ctx_override=None,
        )
        loaded_model.eval()

    assert arch == "llama"

    with torch.no_grad():
        loaded_logits = loaded_model(idx)

    max_diff = (ref_logits - loaded_logits).abs().max().item()
    print(f"  ref_logits  shape={tuple(ref_logits.shape)}  range=[{ref_logits.min():.4f}, {ref_logits.max():.4f}]")
    print(f"  load_logits shape={tuple(loaded_logits.shape)}  range=[{loaded_logits.min():.4f}, {loaded_logits.max():.4f}]")
    print(f"  max abs diff: {max_diff:.2e}")

    assert max_diff < 1e-4, (
        f"compact-format round-trip diverged from reference (max diff {max_diff:.2e}). "
        f"This means the loader's transformations are not mathematically equivalent "
        f"to the reference path."
    )
    print("  PASS — compact format round-trips to within numerical roundoff")

def test_sph_format_unchanged():
    """Loading a standard sph-format checkpoint should be bit-identical to before
    the patches (no unexpected transformations)."""
    torch.manual_seed(7)
    cfg = inference.LlamaConfig(
        block_size=64, vocab_size=128,
        n_layer=2, n_head=2, n_embd=32,
        num_key_value_heads=1, intermediate_size=64,
        rope_layout="interleaved", tie_word_embeddings=False,
    )
    ref_model = inference.LlamaModel(cfg)
    ref_model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref_logits = ref_model(idx)

    from dataclasses import asdict
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sph.pt")
        torch.save({"model": ref_model.state_dict(), "config": asdict(cfg)}, path)

        loaded_model, _, _, _, _, _, _, _ = inference.load_checkpoint(
            path, fp16=False, force_cpu=True, ctx_override=None,
        )
        loaded_model.eval()

    with torch.no_grad():
        loaded_logits = loaded_model(idx)

    max_diff = (ref_logits - loaded_logits).abs().max().item()
    print(f"  max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"sph-format reload diverged ({max_diff:.2e}) — regression!"
    print("  PASS — sph format reload is bit-identical")

def test_gpt2_format_unchanged():
    """GPT-2 path should be untouched by these LLaMA patches."""
    torch.manual_seed(99)
    cfg = inference.GPTConfig(block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=32)
    ref_model = inference.GPT(cfg)
    ref_model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref_logits = ref_model(idx)

    from dataclasses import asdict
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gpt2.pt")
        torch.save({"model": ref_model.state_dict(), "config": asdict(cfg)}, path)

        loaded_model, _, _, _, _, _, _, _ = inference.load_checkpoint(
            path, fp16=False, force_cpu=True, ctx_override=None,
        )
        loaded_model.eval()

    with torch.no_grad():
        loaded_logits = loaded_model(idx)

    max_diff = (ref_logits - loaded_logits).abs().max().item()
    print(f"  max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"gpt2 reload diverged ({max_diff:.2e}) — regression!"
    print("  PASS — gpt2 format reload is bit-identical")

if __name__ == "__main__":
    print("\n[test 1] gpt2 format reload (regression check)")
    test_gpt2_format_unchanged()
    print("\n[test 2] sph LLaMA format reload (regression check)")
    test_sph_format_unchanged()
    print("\n[test 3] compact LLaMA format → numerical equivalence to sph reference")
    test_compact_format_round_trip()
    print("\nAll correctness tests passed.")
