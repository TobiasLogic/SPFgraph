'use strict';

const path = require('path');
const fs = require('fs');

const registry = require('../models/registry');
const PythonBridge = require('../inference/python_bridge');
const GgufBackend = require('../inference/gguf_loader');
const Dashboard = require('../ui/dashboard');

async function loadCmd(modelArg, opts) {
  const resolved = registry.resolve(modelArg);
  if (!resolved) {
    console.error(`Model not found: ${modelArg}`);
    console.error('Tried: filesystem, then ~/.zeroshot/config.json registry.');
    process.exit(1);
  }

  const kind = registry.detectKind(resolved);
  if (kind === 'unknown') {
    console.error(`Unknown model kind for ${resolved} (expected .pt / .gguf).`);
    process.exit(1);
  }

  const modelLabel = path.basename(resolved, path.extname(resolved));
  let backend;

  const chat = !opts.raw;
  if (kind === 'pt') {
    backend = new PythonBridge({
      checkpoint: resolved,
      python: opts.python,
      fp16: opts.fp16 !== false && !opts.cpu,
      cpu: !!opts.cpu,
      ctx: opts.ctx,
      tokenizer: opts.tokenizer,
      arch: opts.arch,
      chat,
    });
  } else {
    backend = new GgufBackend({
      modelPath: resolved,
      ctx: opts.ctx,
      cpu: !!opts.cpu,
      chat,
    });
  }

  const onLoadStderr = (line) => {
    if (line && line.trim()) process.stdout.write(`  · ${line}\n`);
  };
  backend.on('stderr', onLoadStderr);

  const t0 = Date.now();
  const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  let spinnerIdx = 0;
  const spinner = setInterval(() => {
    if (!process.stdout.isTTY) return;
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const frame = spinnerFrames[spinnerIdx++ % spinnerFrames.length];
    process.stdout.write(`\r${frame} loading… ${elapsed}s `);
  }, 100);
  const stopSpinner = () => {
    clearInterval(spinner);
    if (process.stdout.isTTY) process.stdout.write('\r\x1b[K');
  };

  process.stdout.write(`Loading ${modelLabel} (${kind})...\n`);
  try {
    await backend.start();
  } catch (err) {
    stopSpinner();
    backend.removeListener('stderr', onLoadStderr);
    console.error(`\nFailed to load model: ${err.message || err}`);
    if (err && err.stderr && !String(err.message).includes(err.stderr)) {
      console.error('--- python stderr ---\n' + err.stderr);
    }
    process.exit(1);
  }
  stopSpinner();
  backend.removeListener('stderr', onLoadStderr);
  const cfg = backend.config || {};
  const paramsPart = cfg.n_params ? `, ${cfg.n_params} params` : '';
  process.stdout.write(`Ready (${backend.device || kind}${paramsPart}, ${((Date.now() - t0) / 1000).toFixed(1)}s)\n`);

  const dash = new Dashboard({
    backend,
    modelLabel,
    modelPath: resolved,
    opts: {
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      topK: opts.topK,
      topP: opts.topP,
      minP: opts.minP,
      repetitionPenalty: opts.repetitionPenalty,
    },
  });
  await dash.start();
}

module.exports = loadCmd;
