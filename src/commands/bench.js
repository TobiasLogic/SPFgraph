'use strict';

const path = require('path');
const benchmark = require('../bench/benchmark');
const registry = require('../models/registry');

async function benchCmd(modelArg, opts) {
  const resolved = registry.resolve(modelArg);
  if (!resolved) {
    console.error(`Model not found: ${modelArg}`);
    process.exit(1);
  }
  const kind = registry.detectKind(resolved);
  if (kind === 'unknown') {
    console.error(`Unknown model kind for ${resolved}`);
    process.exit(1);
  }

  const label = path.basename(resolved);
  process.stdout.write(`Benchmarking ${label} (${kind})\n`);
  process.stdout.write(`  prompt: ${JSON.stringify(opts.prompt)}\n`);
  process.stdout.write(`  tokens: ${opts.tokens}\n\n`);

  try {
    const r = await benchmark({
      modelPath: resolved,
      kind,
      prompt: opts.prompt,
      tokens: opts.tokens,
      python: opts.python,
    });
    process.stdout.write(`tokens generated : ${r.tokens}\n`);
    process.stdout.write(`elapsed          : ${r.elapsedMs} ms\n`);
    process.stdout.write(`throughput       : ${r.tokensPerSec.toFixed(2)} tok/s\n`);
    if (r.peakVramMb) process.stdout.write(`peak VRAM        : ${r.peakVramMb} MB\n`);
    if (r.config?.n_params) process.stdout.write(`params           : ${r.config.n_params}\n`);
  } catch (err) {
    console.error(`Benchmark failed: ${err.message || err}`);
    process.exit(1);
  }
}

module.exports = benchCmd;
