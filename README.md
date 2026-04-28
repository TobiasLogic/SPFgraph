# @tobiascantcode/SPFgraph — `zeroshot-run`

Run LLMs locally in your terminal — like Ollama, but with `.pt` checkpoint
support for custom GPT-2 decoder-only models, plus GGUF via `node-llama-cpp`.

```
┌─────────────────────────────────────────────┐
│  zeroshot-run  |  ZeroShot-500M  |  530M params  |  fp16  │
│  VRAM: 2.1GB / 4GB   |  Speed: 34 tok/s   |  Ctx: 128/2048 │
├──────────────────┬──────────────────────────┤
│  tok/s over time │  VRAM usage over time    │
├──────────────────┴──────────────────────────┤
│  > you: hello                               │
│    model: Hi! How can I help?               │
├─────────────────────────────────────────────┤
│  You: _                                     │
└─────────────────────────────────────────────┘
```

## Install

```bash
npm i -g @tobiascantcode/SPFgraph
pip install -r python/requirements.txt   # for .pt support: torch, tiktoken
```

The CLI exposes two equivalent binaries: `zeroshot-run` and `SPH`.

## Quickstart

```bash
# Drop a model into ~/.zeroshot/models, or register it explicitly:
zeroshot-run register zeroshot-500m ./checkpoints/zeroshot-500m.pt

zeroshot-run list
zeroshot-run load zeroshot-500m
zeroshot-run bench zeroshot-500m --tokens 256
```

You can also load a path directly:

```bash
zeroshot-run load ./checkpoints/zeroshot-500m.pt
zeroshot-run load ./models/llama3-8b.Q4_K_M.gguf
```

## REPL controls

- **Enter** — send the prompt
- **Esc** — cancel the current generation
- **Ctrl+C** / `q` — quit
- `/clear` — clear the chat history
- `/quit` — exit

## Checkpoint format (`.pt`)

The bundled `python/inference.py` expects checkpoints saved as:

```python
torch.save({'model': model.state_dict(), 'config': asdict(cfg)}, path)
```

with `config` keys: `n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`,
`bias`. Architecture is GPT-2 decoder-only with weight tying and Flash
Attention through `F.scaled_dot_product_attention`. Tokenizer is `tiktoken`'s
GPT-2 encoding (vocab padded to 50304).

## CLI

```
zeroshot-run load <model>      Load a model and start the REPL
zeroshot-run list              List local models
zeroshot-run bench <model>     Throughput benchmark
zeroshot-run register <n> <p>  Add a model to the local registry
```

`load` flags: `--temperature`, `--top-k`, `--max-tokens`, `--ctx`, `--cpu`,
`--fp16`, `--python <path>`.

## Layout

```
src/
  ui/          dashboard.js (blessed-contrib screen)
  inference/   python_bridge.js, gguf_loader.js
  bench/       benchmark.js
  models/      registry.js (~/.zeroshot/config.json)
  commands/    load.js, list.js, bench.js
python/
  inference.py        # GPT-2 decoder, server protocol, one-shot CLI
  requirements.txt
bin/
  zeroshot-run.js
```

## Bridge protocol

The Node side spawns `python/inference.py --server ...` and exchanges
line-delimited JSON over stdin/stdout. Events:

```jsonc
{ "type": "ready",   "config": {...}, "device": "cuda", "vram_total_mb": 4096 }
{ "type": "token",   "text": "...", "id": 123 }
{ "type": "stats",   "tokens_per_sec": 34.2, "vram_used_mb": 2100, "ctx_used": 128, "ctx_max": 2048 }
{ "type": "done",    "total_tokens": 84, "elapsed_ms": 2450 }
{ "type": "error",   "message": "..." }
```

Commands accepted on stdin:

```jsonc
{ "cmd": "generate", "prompt": "...", "max_tokens": 256, "temperature": 0.8, "top_k": 40 }
{ "cmd": "stop" }
{ "cmd": "shutdown" }
```

The GGUF backend (`src/inference/gguf_loader.js`) emits the same event surface
so the dashboard is backend-agnostic.
