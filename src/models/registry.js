'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

const HOME = os.homedir();
const ROOT = path.join(HOME, '.zeroshot');
const CONFIG_PATH = path.join(ROOT, 'config.json');
const MODELS_DIR = path.join(ROOT, 'models');

function ensureRoot() {
  if (!fs.existsSync(ROOT)) fs.mkdirSync(ROOT, { recursive: true });
  if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true });
}

function defaultConfig() {
  return {
    defaults: {
      temperature: 0.8,
      top_k: 40,
      max_tokens: 512,
      fp16: true,
    },
    models: {},
    scan_paths: [MODELS_DIR],
  };
}

function read() {
  ensureRoot();
  if (!fs.existsSync(CONFIG_PATH)) {
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(defaultConfig(), null, 2));
  }
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
    return { ...defaultConfig(), ...JSON.parse(raw) };
  } catch (err) {
    return defaultConfig();
  }
}

function write(cfg) {
  ensureRoot();
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2));
}

function add(name, modelPath) {
  const cfg = read();
  cfg.models[name] = { path: path.resolve(modelPath), added: new Date().toISOString() };
  write(cfg);
}

function remove(name) {
  const cfg = read();
  delete cfg.models[name];
  write(cfg);
}

function resolve(nameOrPath) {
  if (fs.existsSync(nameOrPath)) return path.resolve(nameOrPath);
  const cfg = read();
  if (cfg.models[nameOrPath]) return cfg.models[nameOrPath].path;
  return null;
}

function scan() {
  const cfg = read();
  const seen = new Map();
  for (const [name, meta] of Object.entries(cfg.models)) {
    if (fs.existsSync(meta.path)) {
      seen.set(meta.path, { name, ...meta, source: 'registry' });
    }
  }
  for (const dir of cfg.scan_paths) {
    if (!fs.existsSync(dir)) continue;
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (ext !== '.pt' && ext !== '.gguf' && ext !== '.bin') continue;
      const full = path.join(dir, entry.name);
      if (seen.has(full)) continue;
      const stat = fs.statSync(full);
      seen.set(full, {
        name: path.basename(entry.name, ext),
        path: full,
        size: stat.size,
        added: stat.mtime.toISOString(),
        source: 'scan',
      });
    }
  }
  return Array.from(seen.values());
}

function detectKind(modelPath) {
  const ext = path.extname(modelPath).toLowerCase();
  if (ext === '.pt' || ext === '.pth' || ext === '.ckpt') return 'pt';
  if (ext === '.gguf' || ext === '.ggml' || ext === '.bin') return 'gguf';
  return 'unknown';
}

module.exports = {
  ROOT,
  CONFIG_PATH,
  MODELS_DIR,
  read,
  write,
  add,
  remove,
  resolve,
  scan,
  detectKind,
};
