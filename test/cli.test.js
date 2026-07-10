'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const BIN = path.resolve(__dirname, '..', 'bin', 'zeroshot-run.js');
const pkg = require('../package.json');

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'zeroshot-cli-test-'));
const env = { ...process.env, HOME: tmpHome, USERPROFILE: tmpHome };

function run(args) {
  return spawnSync(process.execPath, [BIN, ...args], { env, encoding: 'utf8' });
}

test('--help lists all commands', () => {
  const r = run(['--help']);
  assert.equal(r.status, 0);
  for (const cmd of ['load', 'serve', 'list', 'bench', 'register']) {
    assert.ok(r.stdout.includes(cmd), `help mentions ${cmd}`);
  }
});

test('--version matches package.json', () => {
  const r = run(['--version']);
  assert.equal(r.status, 0);
  assert.equal(r.stdout.trim(), pkg.version);
});

test('load --help includes --raw and sampling flags', () => {
  const r = run(['load', '--help']);
  assert.equal(r.status, 0);
  for (const flag of ['--raw', '--temperature', '--top-k', '--min-p', '--repetition-penalty', '--ctx']) {
    assert.ok(r.stdout.includes(flag), `load help mentions ${flag}`);
  }
});

test('unknown command exits nonzero', () => {
  const r = run(['frobnicate']);
  assert.notEqual(r.status, 0);
});

test('list runs cleanly on an empty registry', () => {
  const r = run(['list']);
  assert.equal(r.status, 0);
});

test('register rejects a missing file', () => {
  const r = run(['register', 'nope', path.join(tmpHome, 'missing.pt')]);
  assert.notEqual(r.status, 0);
  assert.ok((r.stderr + r.stdout).includes('not found'));
});

test('register + list round-trip', () => {
  const modelPath = path.join(tmpHome, 'm.gguf');
  fs.writeFileSync(modelPath, 'x');
  assert.equal(run(['register', 'm', modelPath]).status, 0);
  const r = run(['list']);
  assert.equal(r.status, 0);
  assert.ok(r.stdout.includes('m'), 'registered model listed');
});
