#!/usr/bin/env node
'use strict';

const { Command } = require('commander');
const path = require('path');
const fs = require('fs');

const pkg = require('../package.json');
const loadCmd = require('../src/commands/load');
const listCmd = require('../src/commands/list');
const benchCmd = require('../src/commands/bench');
const serveCmd = require('../src/commands/serve');
const registry = require('../src/models/registry');

const cfgDefaults = registry.read().defaults || {};
const defaultPython = process.env.ZEROSHOT_PYTHON || cfgDefaults.python || 'python';

const toInt = (v) => {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) throw new Error(`expected integer, got "${v}"`);
  return n;
};
const toFloat = (v) => {
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`expected number, got "${v}"`);
  return n;
};

const program = new Command();

program
  .name('zeroshot-run')
  .description('Run LLMs locally in the terminal — .pt GPT-2 checkpoints and GGUF models')
  .version(pkg.version);

program
  .command('load')
  .description('Load a model and start the interactive REPL dashboard')
  .argument('<model>', 'Path to a .pt checkpoint or .gguf file, or a name from the registry')
  .option('-t, --temperature <n>', 'Sampling temperature', toFloat, cfgDefaults.temperature ?? 0.8)
  .option('-k, --top-k <n>', 'Top-k sampling', toInt, cfgDefaults.top_k ?? 40)
  .option('-m, --max-tokens <n>', 'Max tokens per response', toInt, cfgDefaults.max_tokens ?? 512)
  .option('-p, --top-p <n>', 'Top-p (nucleus) sampling', toFloat, cfgDefaults.top_p ?? 1.0)
  .option('--min-p <n>', 'Min-p sampling', toFloat, cfgDefaults.min_p ?? 0.0)
  .option('--repetition-penalty <n>', 'Repetition penalty', toFloat, cfgDefaults.repetition_penalty ?? 1.0)
  .option('--ctx <n>', 'Context window override', toInt)
  .option('--fp16', 'Use fp16 weights (default for CUDA)', true)
  .option('--cpu', 'Force CPU inference')
  .option('--raw', 'Raw completion mode: no chat template, no conversation history')
  .option('--python <path>', 'Python executable', defaultPython)
  .option('--tokenizer <path>', 'Path to tokenizer.json')
  .option('--arch <name>', 'Architecture override (gpt2, llama)')
  .action((model, opts) => loadCmd(model, opts));

program
  .command('list')
  .description('List models discovered in the local registry')
  .option('--json', 'Output as JSON')
  .action((opts) => listCmd(opts));

program
  .command('bench')
  .description('Run a quick throughput / perplexity benchmark on a model')
  .argument('<model>', 'Model path or name')
  .option('-n, --tokens <n>', 'Tokens to generate for throughput test', toInt, 256)
  .option('--prompt <text>', 'Prompt to use', 'The quick brown fox')
  .option('--python <path>', 'Python executable', defaultPython)
  .action((model, opts) => benchCmd(model, opts));

program
  .command('serve')
  .description('Start an OpenAI-compatible HTTP server')
  .argument('<model>', 'Path to a .pt checkpoint or .gguf file, or a name from the registry')
  .option('--port <n>', 'Port to listen on', toInt, 11434)
  .option('--host <host>', 'Host to listen on', '127.0.0.1')
  .option('--ctx <n>', 'Context window override', toInt)
  .option('--fp16', 'Use fp16 weights (default for CUDA)', true)
  .option('--cpu', 'Force CPU inference')
  .option('--python <path>', 'Python executable', defaultPython)
  .option('--tokenizer <path>', 'Path to tokenizer.json')
  .option('--arch <name>', 'Architecture override (gpt2, llama)')
  .action((model, opts) => serveCmd(model, opts));

program
  .command('register')
  .description('Add a model to the local registry')
  .argument('<name>', 'Short name for the model')
  .argument('<path>', 'Path to the .pt or .gguf file')
  .action((name, modelPath) => {
    const abs = path.resolve(modelPath);
    if (!fs.existsSync(abs)) {
      console.error(`File not found: ${abs}`);
      process.exit(1);
    }
    registry.add(name, abs);
    console.log(`Registered "${name}" -> ${abs}`);
  });

program.parseAsync(process.argv).catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
