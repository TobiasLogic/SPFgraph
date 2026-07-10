'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'zeroshot-test-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;

const registry = require('../src/models/registry');

test('detectKind maps extensions', () => {
  assert.equal(registry.detectKind('/a/model.pt'), 'pt');
  assert.equal(registry.detectKind('/a/model.pth'), 'pt');
  assert.equal(registry.detectKind('/a/model.ckpt'), 'pt');
  assert.equal(registry.detectKind('/a/model.gguf'), 'gguf');
  assert.equal(registry.detectKind('/a/MODEL.GGUF'), 'gguf');
  assert.equal(registry.detectKind('/a/model.bin'), 'gguf');
  assert.equal(registry.detectKind('/a/model.safetensors'), 'unknown');
});

test('read creates a default config', () => {
  const cfg = registry.read();
  assert.ok(cfg.defaults);
  assert.ok(cfg.models);
  assert.ok(Array.isArray(cfg.scan_paths));
  assert.ok(fs.existsSync(registry.CONFIG_PATH));
});

test('add + resolve + remove round-trip', () => {
  const modelPath = path.join(tmpHome, 'fake.pt');
  fs.writeFileSync(modelPath, 'x');
  registry.add('fake', modelPath);
  assert.equal(registry.resolve('fake'), modelPath);
  registry.remove('fake');
  assert.equal(registry.resolve('fake'), null);
});

test('resolve prefers an existing filesystem path over the registry', () => {
  const modelPath = path.join(tmpHome, 'direct.gguf');
  fs.writeFileSync(modelPath, 'x');
  assert.equal(registry.resolve(modelPath), modelPath);
  assert.equal(registry.resolve('no-such-model'), null);
});

test('scan includes registry entries with a size', () => {
  const modelPath = path.join(tmpHome, 'sized.pt');
  fs.writeFileSync(modelPath, 'abc');
  registry.add('sized', modelPath);
  const entry = registry.scan().find((e) => e.name === 'sized');
  assert.ok(entry, 'registered model appears in scan');
  assert.equal(entry.source, 'registry');
  assert.equal(entry.size, 3);
});

test('scan picks up loose files from scan_paths', () => {
  const loose = path.join(registry.MODELS_DIR, 'loose.gguf');
  fs.writeFileSync(loose, 'xyz');
  const entry = registry.scan().find((e) => e.path === loose);
  assert.ok(entry, 'loose model file appears in scan');
  assert.equal(entry.source, 'scan');
  assert.equal(entry.name, 'loose');
});

test('registry entries missing on disk are skipped by scan', () => {
  registry.add('ghost', path.join(tmpHome, 'ghost.pt'));
  fs.rmSync(path.join(tmpHome, 'ghost.pt'), { force: true });
  assert.equal(registry.scan().find((e) => e.name === 'ghost'), undefined);
});
