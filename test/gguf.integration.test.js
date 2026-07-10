'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const MODEL =
  process.env.ZEROSHOT_TEST_GGUF ||
  path.join(os.homedir(), '.unsloth', '.cache', 'stories260K.gguf');

let hasDep = true;
try {
  require.resolve('node-llama-cpp');
} catch {
  hasDep = false;
}

const skip = !hasDep
  ? 'node-llama-cpp not installed'
  : !fs.existsSync(MODEL)
    ? `no test model at ${MODEL} (set ZEROSHOT_TEST_GGUF)`
    : false;

test('gguf backend loads, generates, aborts, and generates again', { skip }, async () => {
  const GgufBackend = require('../src/inference/gguf_loader');
  const backend = new GgufBackend({ modelPath: MODEL, ctx: 512 });
  const ready = await backend.start();
  assert.equal(ready.type, 'ready');
  assert.equal(backend.config.backend, 'gguf');
  assert.ok(backend.config.ctx > 0);

  let tokens = 0;
  backend.on('token', () => tokens++);

  const done = await new Promise((resolve, reject) => {
    backend.once('done', resolve);
    backend.once('error', reject);
    backend.generate({ prompt: 'Once upon a time', maxTokens: 8 });
  });
  assert.ok(done.total_tokens >= 1, 'produced tokens');
  assert.ok(tokens >= 1, 'emitted token events');

  const aborted = await new Promise((resolve, reject) => {
    backend.once('done', resolve);
    backend.once('error', reject);
    backend.generate({ prompt: 'Once upon a time there was a', maxTokens: 2000 });
    setTimeout(() => backend.stop(), 150);
  });
  assert.ok(aborted.total_tokens < 2000, 'abort cut the generation short');

  const after = await new Promise((resolve, reject) => {
    backend.once('done', resolve);
    backend.once('error', reject);
    backend.generate({ prompt: 'The little dog', maxTokens: 8 });
  });
  assert.ok(after.total_tokens >= 1, 'generation works after an abort');

  backend.shutdown();
});
