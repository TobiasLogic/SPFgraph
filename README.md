# @tobiascantcode/spfgraph — `zeroshot-run` / `sph`

[![npm](https://img.shields.io/npm/v/@tobiascantcode/spfgraph.svg)](https://www.npmjs.com/package/@tobiascantcode/spfgraph)
[![license](https://img.shields.io/npm/l/@tobiascantcode/spfgraph.svg)](https://github.com/TobiasLogic/SPFgraph/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-github-black.svg)](https://github.com/TobiasLogic/SPFgraph)

Run LLMs locally in your terminal. Like Ollama, but with first-class support for
custom `.pt` GPT-2 decoder checkpoints. GGUF models work too via
`node-llama-cpp` (optional). The dashboard shows tokens/sec and VRAM in
real time while you chat.

> **Status:** early (`0.1.x`). The `.pt` path is the most exercised; GGUF
> support depends on `node-llama-cpp` building on your platform. Bug reports
> and PRs welcome — see [Issues](https://github.com/TobiasLogic/SPFgraph/issues).
>
> **Tested on:** Windows 11 (PyTorch 2.10 + CUDA 12.6, Python 3.10).
> macOS and Linux are expected to work but currently unverified — please
> open an issue if something's off.

```
┌─────────────────────────────────────────────────────────┐
│ zeroshot-run | ckpt_mid_final | 539M params | fp16      │
│ VRAM: 1.0GB / 4.0GB | Speed: 79.0 tok/s | Ctx: 21/2048  │
├──────────────────────────┬──────────────────────────────┤
│ tok/s over time          │ VRAM (MB) over time          │
├──────────────────────────┴──────────────────────────────┤
│ chat ↓                                                  │
│ you   The quick brown fox                               │
│ model  jumps over the lazy dog" is more concise...      │
├─────────────────────────────────────────────────────────┤
│ you _                                                   │
└─────────────────────────────────────────────────────────┘
```

## Install

```bash
npm i -g @tobiascantcode/spfgraph
```

That gives you two equivalent commands: `zeroshot-run` and `sph`.

### Prerequisites

For **`.pt` checkpoints** (the main path) you need Python with `torch` and
`tiktoken`:

```bash
pip install tiktoken
# For torch, use the install command from https://pytorch.org/get-started/locally/
# that matches your CUDA version (or CPU-only). A bare `pip install torch` may
# pull a build that doesn't match your driver.
```

For **`.gguf` models** the optional `node-llama-cpp` dependency must build
successfully. On Windows that needs Visual Studio Build Tools + cmake +
Python; on macOS/Linux it's usually automatic. If the build fails the npm
install still succeeds — only the GGUF backend is unavailable, the `.pt`
path keeps working.

## Quickstart

```bash
# Drop a checkpoint into ~/.zeroshot/models/, or register one explicitly:
sph register zeroshot-500m ./checkpoints/zeroshot-500m.pt

# See what's known
sph list

# Open the dashboard
sph load zeroshot-500m

# Or pass a path directly
sph load ./model.pt
sph load ./llama3-8b.Q4_K_M.gguf

# Throughput benchmark (no TUI)
sph bench zeroshot-500m --tokens 256 --prompt "Hello"
```

## Commands

```
sph load <model>            Load a model and start the interactive REPL
sph list                    List local models (registered + scanned)
sph bench <model>           Run a quick throughput benchmark
sph register <name> <path>  Add a model to the local registry
```

`load` flags:

| Flag | Default | Effect |
|---|---|---|
| `--temperature <n>` | `0.8` | sampling temperature |
| `--top-k <n>` | `40` | top-k sampling |
| `--max-tokens <n>` | `512` | max tokens per response |
| `--ctx <n>` | model's `block_size` | override context window |
| `--cpu` | off | force CPU inference (otherwise CUDA fp16 is used when available) |
| `--python <path>` | `python` | Python executable (also via `ZEROSHOT_PYTHON` env var) |

`bench` flags: `--tokens` `--prompt` `--python <path>`

## Dashboard controls

Once `load` is running:

| Key | Action |
|---|---|
| **Enter** | send the prompt |
| **Ctrl+C** | quit |
| **Esc** | cancel current generation |
| **Ctrl+T** | scroll up **one line** (slow / fine control) |
| **Ctrl+U** | scroll down **one line** |
| **PgUp** / **PgDn** | half-page scroll |
| **Home** / **End** | jump to top / bottom |
| **Ctrl+B** / **Ctrl+F** | full-page back / forward |
| **Ctrl+G** | jump to bottom and re-enable auto-follow |
| **Mouse wheel** | scroll 3 lines per tick |
| `/clear` | clear chat history |
| `/quit` or `/exit` | quit |

The chat label shows your scroll status: `chat ↓` when pinned to the bottom
(auto-following new tokens), `chat 12/30 (scrolled — End to follow)` when
you've scrolled up to read history. New tokens won't yank you back to the
bottom while you're scrolled up.

## Checkpoint format (`.pt`)

The bundled `python/inference.py` expects checkpoints saved as:

```python
torch.save({'model': model.state_dict(), 'config': asdict(cfg)}, path)
# either key works; `config` and `model_config` are both accepted:
torch.save({'model': model.state_dict(), 'model_config': asdict(cfg)}, path)
```

Required `config` keys: `n_layer`, `n_head`, `n_embd`. Optional keys:
`block_size` (default `1024`), `vocab_size` (default `50304`), `bias`
(default `True`).

Architecture is GPT-2 decoder-only with weight tying and Flash Attention
through `F.scaled_dot_product_attention`. Tokenizer is `tiktoken`'s GPT-2
encoding (vocab padded to 50304).

## How loading works

The loader is designed to fit large models on small GPUs (e.g., a 600M model
on a 4 GB card):

1. `torch.load(..., mmap=True)` — the checkpoint is memory-mapped, not
   read entirely into RAM (PyTorch ≥ 2.1).
2. The model is built on the **meta device** with `set_default_dtype(fp16)`
   — every parameter has a shape but **zero storage**, so there's no CPU
   RAM spike from default initialization.
3. `model.to_empty(device='cuda')` allocates fresh fp16 storage directly on
   the GPU.
4. `load_state_dict(...)` copies tensors **one at a time**, casting fp32 →
   fp16 in flight. Peak temporary memory ≈ one tensor.

For a 600M-parameter model the peak footprint works out to about 2.4 GB CPU
RAM (mmap'd state dict) plus 1.2 GB VRAM (fp16 model). YMMV depending on
model size and `n_embd`.

## CUDA OOM auto-fallback

If `to_empty(device='cuda')` raises `OutOfMemoryError` (your GPU is
contended — browser, other ML processes, etc.), the loader automatically
retries on CPU instead of bailing:

```
materializing on cuda...
  CUDA OOM during materialization: ...
  falling back to CPU (re-run with --cpu to skip this attempt next time)
```

CPU is much slower than GPU but always works.

## Bridge protocol

`zeroshot-run` spawns `python/inference.py --server ...` and exchanges
line-delimited JSON over stdin/stdout. Events from Python:

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

The GGUF backend (`src/inference/gguf_loader.js`) emits the same event
surface so the dashboard is backend-agnostic.

## Config

`~/.zeroshot/config.json` — model registry and defaults. Edit it directly to
change defaults or scan paths:

```json
{
  "defaults": { "temperature": 0.8, "top_k": 40, "max_tokens": 512, "fp16": true },
  "models": {
    "zeroshot-500m": { "path": "C:\\path\\to\\model.pt", "added": "..." }
  },
  "scan_paths": ["C:\\Users\\you\\.zeroshot\\models"]
}
```

Drop any `.pt` / `.gguf` file into `~/.zeroshot/models/` and `sph list`
will pick it up automatically.

## Environment variables

| Variable | Effect |
|---|---|
| `ZEROSHOT_PYTHON` | Python executable to spawn (default `python`) |
| `ZEROSHOT_DEBUG=1` | Log every keypress and mouse event the dashboard receives to `~/.zeroshot/debug.log`. Use this if a key or wheel feels unresponsive — the log tells you whether the terminal is sending the event at all. |
| `PYTORCH_CUDA_ALLOC_CONF` | Inherited by the Python process. Defaults to `expandable_segments:True` (no-op on Windows, helps on Linux when free VRAM is fragmented). |

## Layout

```
src/
  ui/dashboard.js          blessed-contrib screen, scroll handling, render loop
  inference/
    python_bridge.js       spawn + line-delimited JSON protocol for .pt models
    gguf_loader.js         optional node-llama-cpp backend, same event surface
  bench/benchmark.js       throughput benchmark
  models/registry.js       ~/.zeroshot/config.json
  commands/                load.js, list.js, bench.js
python/
  inference.py             GPT-2 decoder + meta-device loader + JSON server
  requirements.txt
bin/
  zeroshot-run.js
```

## Troubleshooting

**`node-llama-cpp` install warnings on Windows** — benign. The package is in
`optionalDependencies`, so a build failure is just a warning. Only the GGUF
backend is affected; `.pt` loading goes through Python and works regardless.

**`python exited before ready (code=3221225477)`** — that's `0xC0000005`,
a Windows access violation, historically caused by torch running out of CPU
RAM during default model initialization for large models on systems with
≤8 GB RAM. The current loader uses meta-device construction to avoid that
spike. If you still hit it, run with `--cpu` and report the stderr in an
[issue](https://github.com/TobiasLogic/SPFgraph/issues).

**`CUDA out of memory`** — the loader auto-retries on CPU. To skip the CUDA
attempt entirely, pass `--cpu`. To free GPU memory, close other apps that
use the GPU (browsers with hardware acceleration, Discord, OBS, other
training jobs).

**Scrolling feels dead** — try `Ctrl+T` / `Ctrl+U` (one-line scroll) before
PgUp/PgDn; some Windows console configurations don't forward PgUp/PgDn to
applications. If even Ctrl+T does nothing, run with `ZEROSHOT_DEBUG=1` and
check `~/.zeroshot/debug.log` to see whether the terminal is sending the
event at all.

**`sph: command not found` after install** — make sure you installed with
`-g` and that npm's global bin directory is on your PATH. `npm config get prefix`
shows where global bins go.

**`E404` on `npm publish`** (developers only) — npm's confusing way of
saying "you don't have permission to publish to this scope." Run
`npm whoami`; if it shows the wrong user, run `npm logout && npm login`
and retry. Add `--otp=123456` if 2FA is on.

## Contributing

Source: [github.com/TobiasLogic/SPFgraph](https://github.com/TobiasLogic/SPFgraph).
Issues and PRs welcome — especially:
- Reports from macOS or Linux (currently unverified)
- GGUF / `node-llama-cpp` interactions on different platforms
- Other GPT-style checkpoint formats worth supporting

## License

[MIT](https://github.com/TobiasLogic/SPFgraph/blob/main/LICENSE)
