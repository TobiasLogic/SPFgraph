"""
zeroshot-run inference script.

Two modes:

1. One-shot CLI:
       python inference.py --checkpoint model.pt --prompt "Hello" \
                           --max_tokens 128 --temperature 0.8 --fp16

2. Server mode (used by the Node bridge):
       python inference.py --checkpoint model.pt --server [--fp16] [--cpu] [--ctx N]
                           [--tokenizer ./tokenizer.json] [--arch gpt2|llama]

   Server mode reads line-delimited JSON commands from stdin and writes
   line-delimited JSON events to stdout.

Checkpoint format (custom):
    torch.save({'model': model.state_dict(), 'config': cfg_dict}, path)

Required config keys: n_layer, n_head, n_embd
Optional:             block_size, vocab_size, bias (gpt2), arch
                      rope_theta, num_key_value_heads, intermediate_size (llama)

Architectures supported:
    gpt2  — GPT-2 decoder-only with weight tying, learned position embeddings,
            LayerNorm, GELU MLP, packed QKV. Default.
    llama — LLaMA-style: RMSNorm, RoPE, SwiGLU MLP, separate Q/K/V projections,
            optional grouped-query attention. Experimental — no shipping
            checkpoint of this format yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar et al., 2019).

    For each previously-generated token: if its score is positive, divide by
    the penalty (push toward zero); if negative, multiply (push more negative).
    Both reduce the probability of repeating that token.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = list(set(generated_ids))
    idx = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)
    selected = logits[..., idx]
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits[..., idx] = selected
    return logits

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    cutoff = v[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling (Holtzman et al., 2019)."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumprobs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, remove)
    return torch.where(mask, torch.full_like(logits, -float("inf")), logits)

def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-P sampling (Nguyen et al., 2024).

    Drops tokens whose probability is below `min_p × p_max`. The threshold
    scales with the model's confidence: when the top token is highly probable
    the cutoff is strict; when the model is unsure it keeps more candidates.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * p_max
    return torch.where(probs < threshold, torch.full_like(logits, -float("inf")), logits)

def sample_next(
    logits: torch.Tensor,
    generated_ids: List[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply the full sampling pipeline and return the next token id (shape [B, 1])."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = logits / max(temperature, 1e-5)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    logits = apply_min_p(logits, min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

class TokenizerAdapter:
    """Common interface: encode(str) → list[int], decode(list[int]) → str."""

    def encode(self, text: str) -> List[int]: raise NotImplementedError
    def decode(self, ids: List[int]) -> str:  raise NotImplementedError

    @property
    def vocab_size(self) -> int: raise NotImplementedError

    @property
    def kind(self) -> str: raise NotImplementedError

class TiktokenAdapter(TokenizerAdapter):
    def __init__(self, enc):
        self.enc = enc

    def encode(self, text):
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids):
        return self.enc.decode(ids)

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    @property
    def kind(self):
        return "tiktoken-gpt2"

class HFTokenizerAdapter(TokenizerAdapter):
    def __init__(self, tok, path: str):
        self.tok = tok
        self.path = path

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    @property
    def kind(self):
        return f"hf:{os.path.basename(self.path)}"

def get_tokenizer(checkpoint_path: str, override: Optional[str], cfg: dict) -> TokenizerAdapter:
    """Pick a tokenizer in priority order:
       1. --tokenizer override
       2. tokenizer.json sibling of the checkpoint
       3. cfg['tokenizer'] path
       4. tiktoken GPT-2 (default)
    """
    candidate = override or _sibling_tokenizer(checkpoint_path) or cfg.get("tokenizer")
    if candidate and os.path.exists(candidate):
        try:
            from tokenizers import Tokenizer
            return HFTokenizerAdapter(Tokenizer.from_file(candidate), candidate)
        except ImportError:
            log_err(f"  tokenizer.json found at {candidate} but `tokenizers` package is not installed; "
                    "falling back to tiktoken (run: pip install tokenizers)")
    import tiktoken
    return TiktokenAdapter(tiktoken.get_encoding("gpt2"))

def _sibling_tokenizer(checkpoint_path: str) -> Optional[str]:
    if not checkpoint_path:
        return None
    sibling = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "tokenizer.json")
    return sibling if os.path.exists(sibling) else None

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    dropout: float = 0.0
    arch: str = "gpt2"

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    arch_name = "gpt2"

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def retie_weights(self):
        self.transformer.wte.weight = self.lm_head.weight

@dataclass
class LlamaConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    rope_layout: str = "interleaved"
    tie_word_embeddings: bool = False
    arch: str = "llama"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(orig_dtype) * self.weight)

def precompute_rope(dim: int, max_seq_len: int, base: float, device, dtype):
    """Standard RoPE frequency table, cached as (cos, sin)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               layout: str = "interleaved") -> torch.Tensor:
    """Rotary position embedding.

    Two pair conventions exist in the wild:
      - "interleaved" (LLaMA reference, llama.cpp): pairs are (x[2i], x[2i+1])
      - "split_halves" (GPT-NeoX, HF transformers, nanoGPT-llama, our ZeroShot trainer):
        pairs are (x[i], x[i+D/2])
    They are NOT mathematically equivalent — Q/K weights trained for one convention
    produce garbage under the other unless explicitly permuted.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    if layout == "split_halves":
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)

class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.num_key_value_heads or cfg.n_head
        assert cfg.n_embd % cfg.n_head == 0
        assert self.n_head % self.n_kv == 0, "n_head must be divisible by num_key_value_heads"
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rope_layout = cfg.rope_layout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T], self.rope_layout)
        k = apply_rope(k, cos[:T], sin[:T], self.rope_layout)

        if self.n_kv != self.n_head:
            repeat = self.n_head // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        ff = cfg.intermediate_size or (4 * cfg.n_embd)
        self.gate_proj = nn.Linear(cfg.n_embd, ff, bias=False)
        self.up_proj   = nn.Linear(cfg.n_embd, ff, bias=False)
        self.down_proj = nn.Linear(ff, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class LlamaModel(nn.Module):
    arch_name = "llama"

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._cos = None
        self._sin = None

    def _ensure_rope(self, T, device, dtype):
        if self._cos is None or self._cos.size(0) < T or self._cos.device != device:
            head_dim = self.config.n_embd // self.config.n_head
            self._cos, self._sin = precompute_rope(head_dim, max(T, self.config.block_size), self.config.rope_theta, device, dtype)

    def forward(self, idx):
        B, T = idx.size()
        x = self.embed_tokens(idx)
        self._ensure_rope(T, x.device, x.dtype)
        for block in self.layers:
            x = block(x, self._cos, self._sin)
        x = self.norm(x)
        return self.lm_head(x)

    def retie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def transformer(self):
        class _Proxy:
            def __init__(_self, mdl): _self.wte = mdl.embed_tokens
        return _Proxy(self)

ARCH_REGISTRY = {
    "gpt2":  (GPT,         GPTConfig),
    "llama": (LlamaModel,  LlamaConfig),
}

def detect_arch(cfg_dict: dict) -> str:
    if cfg_dict.get("arch"):
        return cfg_dict["arch"]
    if "rope_theta" in cfg_dict or "num_key_value_heads" in cfg_dict or "intermediate_size" in cfg_dict:
        return "llama"
    return "gpt2"

LLAMA_KEY_ALIASES = {
    "tok_emb.weight":        "embed_tokens.weight",
    "attn_norm.weight":             "input_layernorm.weight",
    "mlp_norm.weight":              "post_attention_layernorm.weight",
    "attn.q_proj.weight":           "self_attn.q_proj.weight",
    "attn.k_proj.weight":           "self_attn.k_proj.weight",
    "attn.v_proj.weight":           "self_attn.v_proj.weight",
    "attn.o_proj.weight":           "self_attn.o_proj.weight",
}

def remap_llama_state_dict(state: dict) -> tuple:
    """Return (remapped_state, was_remapped). Detects the compact custom-trainer
    format by the presence of `tok_emb.weight` and renames keys to sph's layout.
    Also strips a leading `model.` prefix if present (HF transformers format)."""
    if any(k.startswith("model.") for k in state):
        new = {}
        for k, v in state.items():
            new[k[len("model."):] if k.startswith("model.") else k] = v
        state = new

    if "tok_emb.weight" not in state and "embed_tokens.weight" in state:
        return state, False
    if "tok_emb.weight" not in state:
        return state, False

    new = {}
    for k, v in state.items():
        nk = k
        if k == "tok_emb.weight":
            nk = "embed_tokens.weight"
        elif k.startswith("layers.") and ".attn_norm." in k:
            nk = k.replace(".attn_norm.", ".input_layernorm.")
        elif k.startswith("layers.") and ".mlp_norm." in k:
            nk = k.replace(".mlp_norm.", ".post_attention_layernorm.")
        elif k.startswith("layers.") and ".attn." in k:
            nk = k.replace(".attn.", ".self_attn.")
        new[nk] = v
    return new, True

def detect_rope_layout_from_state(state: dict) -> Optional[str]:
    """Best-effort guess of the RoPE convention used at training time.

    The compact custom-trainer format (`tok_emb.weight` etc.) historically uses
    GPT-NeoX-style split-halves rotation. sph's default and HF transformers'
    LlamaModel use interleaved. If the caller's config doesn't specify
    `rope_layout`, default per format.
    """
    if "tok_emb.weight" in state:
        return "split_halves"
    return None

def build_config(arch: str, cfg_dict: dict, ctx_override: Optional[int]):
    _, cfg_cls = ARCH_REGISTRY[arch]
    if ctx_override:
        cfg_dict = dict(cfg_dict)
        cfg_dict["block_size"] = ctx_override
    if arch == "gpt2":
        return GPTConfig(
            block_size=cfg_dict.get("block_size", 1024),
            vocab_size=cfg_dict.get("vocab_size", 50304),
            n_layer=cfg_dict["n_layer"],
            n_head=cfg_dict["n_head"],
            n_embd=cfg_dict["n_embd"],
            bias=cfg_dict.get("bias", True),
            dropout=0.0,
        )
    if arch == "llama":
        return LlamaConfig(
            block_size=cfg_dict.get("block_size", 2048),
            vocab_size=cfg_dict.get("vocab_size", 32000),
            n_layer=cfg_dict["n_layer"],
            n_head=cfg_dict["n_head"],
            n_embd=cfg_dict["n_embd"],
            num_key_value_heads=cfg_dict.get("num_key_value_heads"),
            intermediate_size=cfg_dict.get("intermediate_size"),
            rope_theta=cfg_dict.get("rope_theta", 10000.0),
            norm_eps=cfg_dict.get("norm_eps", 1e-5),
            rope_layout=cfg_dict.get("rope_layout", "interleaved"),
            tie_word_embeddings=bool(cfg_dict.get("tie_word_embeddings", False)),
        )
    raise RuntimeError(f"unknown arch: {arch}")

def log_err(msg):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def vram_total_mb(device):
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024))
    except Exception:
        return None

def vram_used_mb(device):
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.memory_allocated(device) / (1024 * 1024))
    except Exception:
        return None

def load_checkpoint(path, fp16, force_cpu, ctx_override, arch_override=None):
    device = pick_device(force_cpu)
    log_err(f"loading checkpoint from {path} (device={device}, fp16={fp16})...")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        log_err("  (loaded with mmap)")
    except (TypeError, RuntimeError) as e:
        log_err(f"  (mmap unavailable: {e}; loading normally)")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Checkpoint must contain a 'model' state_dict")
    cfg_src = ckpt.get("config", ckpt.get("model_config"))
    if cfg_src is None:
        raise RuntimeError("Checkpoint must contain a 'config' or 'model_config' dict")
    cfg_dict = dict(cfg_src)

    arch = arch_override or detect_arch(cfg_dict)
    if arch not in ARCH_REGISTRY:
        raise RuntimeError(f"unknown architecture: {arch}. Known: {list(ARCH_REGISTRY)}")
    log_err(f"  detected arch: {arch}")
    model_cls, _ = ARCH_REGISTRY[arch]

    if arch == "llama":
        peek_state = ckpt.get("model", {})
        if "rope_layout" not in cfg_dict:
            guessed = detect_rope_layout_from_state(peek_state)
            if guessed:
                cfg_dict = dict(cfg_dict)
                cfg_dict["rope_layout"] = guessed
                log_err(f"  rope_layout: {guessed} (auto-detected from state-dict layout)")
        else:
            log_err(f"  rope_layout: {cfg_dict['rope_layout']} (from config)")
        if "tie_word_embeddings" not in cfg_dict:
            if "tok_emb.weight" in peek_state:
                cfg_dict = dict(cfg_dict)
                cfg_dict["tie_word_embeddings"] = True
                log_err("  tie_word_embeddings: True (auto, from compact key naming)")

    cfg = build_config(arch, cfg_dict, ctx_override)

    target_dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32

    log_err(f"building model on meta device ({target_dtype}, {arch}): {cfg}")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with torch.device("meta"):
            model = model_cls(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)

    log_err(f"materializing on {device}...")
    try:
        model = model.to_empty(device=device)
    except torch.cuda.OutOfMemoryError as e:
        log_err(f"  CUDA OOM during materialization: {e}")
        log_err("  falling back to CPU (re-run with --cpu to skip this attempt next time)")
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        target_dtype = torch.float32
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            with torch.device("meta"):
                model = model_cls(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        model = model.to_empty(device=device)

    model.retie_weights()

    state = ckpt["model"]
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("_orig_mod.", "module."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    if arch == "llama":
        cleaned, remapped = remap_llama_state_dict(cleaned)
        if remapped:
            log_err("  remapped compact LLaMA state-dict keys → sph layout")
        if cfg.rope_layout == "split_halves":
            head_dim = cfg.n_embd // cfg.n_head
            n_kv = cfg.num_key_value_heads or cfg.n_head

            def _permute(w, n_h, hd):
                return w.view(n_h, 2, hd // 2, -1).transpose(1, 2).contiguous().view(n_h * hd, -1)

            permuted = 0
            for i in range(cfg.n_layer):
                qk = f"layers.{i}.self_attn.q_proj.weight"
                kk = f"layers.{i}.self_attn.k_proj.weight"
                if qk in cleaned:
                    cleaned[qk] = _permute(cleaned[qk], cfg.n_head, head_dim)
                    permuted += 1
                if kk in cleaned:
                    cleaned[kk] = _permute(cleaned[kk], n_kv, head_dim)
                    permuted += 1
            for layer in model.layers:
                layer.self_attn.rope_layout = "interleaved"
            log_err(f"  permuted {permuted} Q/K projection weights (split_halves → interleaved)")

    log_err("loading state dict (streaming copy → device)...")
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    del state, cleaned, ckpt

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    log_err(f"ready. (missing={len(missing)} unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, device, n_params, missing, unexpected, arch, cfg_dict

@torch.no_grad()
def generate_stream(model, idx, max_new_tokens, *, temperature, top_k, top_p, min_p, repetition_penalty):
    """Yield token ids one at a time, applying the full sampling pipeline."""
    block_size = model.config.block_size
    generated_so_far = idx[0].tolist()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]
        next_id = sample_next(
            logits, generated_so_far,
            temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, repetition_penalty=repetition_penalty,
        )
        idx = torch.cat((idx, next_id), dim=1)
        nid = int(next_id.item())
        generated_so_far.append(nid)
        yield nid

def run_server(args):
    try:
        model, cfg, device, n_params, missing, unexpected, arch, raw_cfg = load_checkpoint(
            args.checkpoint, args.fp16, args.cpu, args.ctx, arch_override=args.arch,
        )
    except Exception as e:
        emit({"type": "error", "message": f"load failed: {e}"})
        return

    enc = get_tokenizer(args.checkpoint, args.tokenizer, raw_cfg)
    log_err(f"  tokenizer: {enc.kind} (vocab={enc.vocab_size})")

    emit({
        "type": "ready",
        "config": {
            "arch": arch,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "n_params": n_params,
            "fp16": args.fp16 and device.type == "cuda",
            "checkpoint": os.path.abspath(args.checkpoint),
            "tokenizer": enc.kind,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "device": str(device),
        "vram_total_mb": vram_total_mb(device),
    })

    stop_flag = {"v": False}
    last_ctx = {"used": 0}

    def handle_generate(payload):
        stop_flag["v"] = False
        prompt = payload.get("prompt", "")
        max_new = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k") or 0)
        top_p = float(payload.get("top_p") or 1.0)
        min_p = float(payload.get("min_p") or 0.0)
        rep_penalty = float(payload.get("repetition_penalty") or 1.0)

        ids = enc.encode(prompt)
        if len(ids) >= cfg.block_size:
            ids = ids[-(cfg.block_size - 1):]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        start = time.time()
        last_tick = start
        last_tick_tokens = 0
        emitted = 0
        buf_ids: list = []

        try:
            for tok_id in generate_stream(
                model, idx, max_new,
                temperature=temperature, top_k=top_k, top_p=top_p,
                min_p=min_p, repetition_penalty=rep_penalty,
            ):
                if stop_flag["v"]:
                    break
                buf_ids.append(tok_id)
                emitted += 1
                last_ctx["used"] = len(ids) + emitted

                try:
                    text = enc.decode(buf_ids)
                except Exception:
                    text = None
                if text is not None:
                    emit({"type": "token", "text": text, "id": tok_id})
                    buf_ids = []

                now = time.time()
                if now - last_tick >= 0.2:
                    dt = now - last_tick
                    tps = (emitted - last_tick_tokens) / dt if dt > 0 else 0.0
                    last_tick = now
                    last_tick_tokens = emitted
                    emit({
                        "type": "stats",
                        "tokens_per_sec": tps,
                        "vram_used_mb": vram_used_mb(device),
                        "ctx_used": len(ids) + emitted,
                        "ctx_max": cfg.block_size,
                    })
        except Exception as e:
            emit({"type": "error", "message": f"generate failed: {e}"})
            return

        if buf_ids:
            try:
                emit({"type": "token", "text": enc.decode(buf_ids), "id": buf_ids[-1]})
            except Exception:
                pass

        elapsed_ms = int((time.time() - start) * 1000)
        emit({"type": "done", "total_tokens": emitted, "elapsed_ms": elapsed_ms})

    cmd_queue: "queue.Queue" = queue.Queue()

    def stdin_reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as e:
                emit({"type": "error", "message": f"bad command: {e}"})
                continue
            cmd = msg.get("cmd")
            if cmd == "stop":
                stop_flag["v"] = True
                continue
            if cmd == "shutdown":
                stop_flag["v"] = True
            cmd_queue.put(msg)
        cmd_queue.put(None)

    threading.Thread(target=stdin_reader, daemon=True).start()

    while True:
        msg = cmd_queue.get()
        if msg is None:
            return
        cmd = msg.get("cmd")
        if cmd == "generate":
            handle_generate(msg)
        elif cmd == "stats":
            emit({
                "type": "stats",
                "tokens_per_sec": None,
                "vram_used_mb": vram_used_mb(device),
                "ctx_used": last_ctx["used"],
                "ctx_max": cfg.block_size,
            })
        elif cmd == "shutdown":
            return
        else:
            emit({"type": "error", "message": f"unknown cmd: {cmd}"})

def run_oneshot(args):
    model, cfg, device, n_params, _, _, arch, raw_cfg = load_checkpoint(
        args.checkpoint, args.fp16, args.cpu, args.ctx, arch_override=args.arch,
    )
    enc = get_tokenizer(args.checkpoint, args.tokenizer, raw_cfg)
    ids = enc.encode(args.prompt or "")
    if len(ids) >= cfg.block_size:
        ids = ids[-(cfg.block_size - 1):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    sys.stdout.write(args.prompt or "")
    sys.stdout.flush()
    buf_ids: list = []
    for tok_id in generate_stream(
        model, idx, args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    ):
        buf_ids.append(tok_id)
        try:
            text = enc.decode(buf_ids)
            sys.stdout.write(text)
            sys.stdout.flush()
            buf_ids = []
        except Exception:
            continue
    sys.stdout.write("\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=1.0,
                   help="nucleus sampling: keep tokens with cumulative prob ≤ top_p (1.0 disables)")
    p.add_argument("--min_p", type=float, default=0.0,
                   help="min-p sampling: drop tokens below min_p × p_max (0.0 disables; 0.05 recommended)")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="penalize repeated tokens (1.0 disables; 1.1 typical)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--arch", default=None, choices=list(ARCH_REGISTRY.keys()) + [None],
                   help="override architecture detection (default: auto)")
    p.add_argument("--tokenizer", default=None,
                   help="path to tokenizer.json (Hugging Face tokenizers format)")
    p.add_argument("--server", action="store_true", help="line-delimited JSON server mode")
    args = p.parse_args()
    if args.server:
        run_server(args)
    else:
        run_oneshot(args)

if __name__ == "__main__":
    main()
