from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    dropout: float = 0.0


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
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
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

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield int(next_id.item())




def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")




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


def vram_total_mb(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        return int(props.total_memory / (1024 * 1024))
    except Exception:
        return None


def vram_used_mb(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.memory_allocated(device) / (1024 * 1024))
    except Exception:
        return None


def load_checkpoint(path: str, fp16: bool, force_cpu: bool, ctx_override: Optional[int]):
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
    if ctx_override:
        cfg_dict["block_size"] = ctx_override
    cfg = GPTConfig(
        block_size=cfg_dict.get("block_size", 1024),
        vocab_size=cfg_dict.get("vocab_size", 50304),
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"],
        bias=cfg_dict.get("bias", True),
        dropout=0.0,
    )

    target_dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32

    log_err(f"building model on meta device ({target_dtype}): {cfg}")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with torch.device("meta"):
            model = GPT(cfg)
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
                model = GPT(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        model = model.to_empty(device=device)
    model.transformer.wte.weight = model.lm_head.weight

    state = ckpt["model"]
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("_orig_mod.", "module."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    log_err("loading state dict (streaming copy → device)...")
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    del state, cleaned, ckpt

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    log_err(f"ready. (missing={len(missing)} unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, device, n_params, missing, unexpected




def run_server(args):
    try:
        model, cfg, device, n_params, missing, unexpected = load_checkpoint(
            args.checkpoint, args.fp16, args.cpu, args.ctx
        )
    except Exception as e:
        emit({"type": "error", "message": f"load failed: {e}"})
        return

    enc = get_tokenizer()

    emit(
        {
            "type": "ready",
            "config": {
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "bias": cfg.bias,
                "n_params": n_params,
                "fp16": args.fp16 and device.type == "cuda",
                "checkpoint": os.path.abspath(args.checkpoint),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            "device": str(device),
            "vram_total_mb": vram_total_mb(device),
        }
    )

    stop_flag = {"v": False}

    def handle_generate(payload):
        prompt = payload.get("prompt", "")
        max_new = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.8))
        top_k = payload.get("top_k", 40)
        top_k = int(top_k) if top_k else None

        ids = enc.encode(prompt, allowed_special=set())
        if len(ids) >= cfg.block_size:
            ids = ids[-(cfg.block_size - 1):]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        start = time.time()
        last_tick = start
        last_tick_tokens = 0
        emitted = 0
        buf_ids: list[int] = []

        try:
            for tok_id in model.generate_stream(idx, max_new, temperature=temperature, top_k=top_k):
                if stop_flag["v"]:
                    stop_flag["v"] = False
                    break
                buf_ids.append(tok_id)
                emitted += 1

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
                    emit(
                        {
                            "type": "stats",
                            "tokens_per_sec": tps,
                            "vram_used_mb": vram_used_mb(device),
                            "ctx_used": len(ids) + emitted,
                            "ctx_max": cfg.block_size,
                        }
                    )
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
        if cmd == "generate":
            handle_generate(msg)
        elif cmd == "stop":
            stop_flag["v"] = True
        elif cmd == "shutdown":
            return
        else:
            emit({"type": "error", "message": f"unknown cmd: {cmd}"})




def run_oneshot(args):
    model, cfg, device, n_params, _, _ = load_checkpoint(
        args.checkpoint, args.fp16, args.cpu, args.ctx
    )
    enc = get_tokenizer()
    ids = enc.encode(args.prompt or "", allowed_special=set())
    if len(ids) >= cfg.block_size:
        ids = ids[-(cfg.block_size - 1):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    sys.stdout.write(args.prompt or "")
    sys.stdout.flush()
    buf_ids: list[int] = []
    for tok_id in model.generate_stream(
        idx,
        args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
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
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--server", action="store_true", help="Run line-delimited JSON server")
    args = p.parse_args()
    if args.server:
        run_server(args)
    else:
        run_oneshot(args)


if __name__ == "__main__":
    main()
