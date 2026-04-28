'use strict';

const PythonBridge = require('../inference/python_bridge');
const GgufBackend = require('../inference/gguf_loader');
const registry = require('../models/registry');

async function benchmark({ modelPath, kind, prompt, tokens, python }) {
  const backend = kind === 'pt'
    ? new PythonBridge({ checkpoint: modelPath, python, fp16: true })
    : new GgufBackend({ modelPath, ctx: 2048 });

  await backend.start();

  let peakVram = 0;
  backend.on('stats', (msg) => {
    if (msg.vram_used_mb && msg.vram_used_mb > peakVram) peakVram = msg.vram_used_mb;
  });

  const result = await new Promise((resolve, reject) => {
    backend.on('done', (msg) => resolve(msg));
    backend.on('error', (err) => reject(err instanceof Error ? err : new Error(err.message || String(err))));
    backend.generate({ prompt, maxTokens: tokens, temperature: 0.8, topK: 40 });
  });

  backend.shutdown();

  const elapsedMs = result.elapsed_ms;
  const tps = elapsedMs > 0 ? (result.total_tokens * 1000) / elapsedMs : 0;
  return {
    tokens: result.total_tokens,
    elapsedMs,
    tokensPerSec: tps,
    peakVramMb: peakVram || null,
    config: backend.config,
  };
}

module.exports = benchmark;
module.exports.resolveModel = (arg) => {
  const resolved = registry.resolve(arg);
  if (!resolved) throw new Error(`Model not found: ${arg}`);
  return { path: resolved, kind: registry.detectKind(resolved) };
};
