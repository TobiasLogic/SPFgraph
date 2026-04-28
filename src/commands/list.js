'use strict';

const path = require('path');
const registry = require('../models/registry');

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return '?';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(1)} ${units[i]}`;
}

function listCmd(opts) {
  const entries = registry.scan();
  if (opts.json) {
    process.stdout.write(JSON.stringify(entries, null, 2) + '\n');
    return;
  }
  if (entries.length === 0) {
    console.log('No models found.');
    console.log(`Drop .pt or .gguf files into ${registry.MODELS_DIR}`);
    console.log('or register one with: zeroshot-run register <name> <path>');
    return;
  }
  const rows = entries.map((e) => ({
    name: e.name,
    kind: registry.detectKind(e.path),
    size: e.size != null ? fmtSize(e.size) : '-',
    source: e.source,
    path: e.path,
  }));
  const widths = {
    name: Math.max(4, ...rows.map((r) => r.name.length)),
    kind: 4,
    size: Math.max(4, ...rows.map((r) => r.size.length)),
    source: 8,
  };
  const pad = (s, w) => String(s).padEnd(w, ' ');
  console.log(
    `${pad('NAME', widths.name)}  ${pad('KIND', widths.kind)}  ${pad('SIZE', widths.size)}  ${pad('SOURCE', widths.source)}  PATH`
  );
  for (const r of rows) {
    console.log(
      `${pad(r.name, widths.name)}  ${pad(r.kind, widths.kind)}  ${pad(r.size, widths.size)}  ${pad(r.source, widths.source)}  ${r.path}`
    );
  }
}

module.exports = listCmd;
