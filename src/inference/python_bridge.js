'use strict';

const { spawn } = require('child_process');
const path = require('path');
const { EventEmitter } = require('events');

const SCRIPT = path.resolve(__dirname, '..', '..', 'python', 'inference.py');

class PythonBridge extends EventEmitter {
  constructor(opts) {
    super();
    this.checkpoint = opts.checkpoint;
    this.python = opts.python || 'python';
    this.fp16 = opts.fp16 !== false;
    this.cpu = !!opts.cpu;
    this.ctx = opts.ctx;
    this.proc = null;
    this.ready = false;
    this.config = null;
    this.device = null;
    this.vramTotalMb = null;
    this._stdoutBuf = '';
    this._stderrBuf = '';
  }

  start() {
    const args = [SCRIPT, '--checkpoint', this.checkpoint, '--server'];
    if (this.fp16) args.push('--fp16');
    if (this.cpu) args.push('--cpu');
    if (this.ctx) args.push('--ctx', String(this.ctx));

    this.proc = spawn(this.python, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: {
        ...process.env,
        PYTORCH_CUDA_ALLOC_CONF:
          process.env.PYTORCH_CUDA_ALLOC_CONF || 'expandable_segments:True',
        PYTHONIOENCODING: 'utf-8',
      },
    });

    this.proc.stdout.setEncoding('utf8');
    this.proc.stderr.setEncoding('utf8');

    this.proc.stdout.on('data', (chunk) => this._onStdout(chunk));
    this.proc.stderr.on('data', (chunk) => this._onStderr(chunk));

    this.proc.on('error', (err) => this.emit('error', err));
    this.proc.on('exit', (code, signal) => {
      this.ready = false;
      this.emit('exit', { code, signal });
    });

    return new Promise((resolve, reject) => {
      const stderrBuf = [];
      const onStderr = (line) => {
        stderrBuf.push(line);
        if (stderrBuf.length > 200) stderrBuf.shift();
      };
      this.on('stderr', onStderr);

      const cleanup = () => {
        this.removeListener('ready', onReady);
        this.removeListener('error', onError);
        this.removeListener('exit', onExit);
        this.removeListener('stderr', onStderr);
      };

      const onReady = (msg) => {
        this.ready = true;
        this.config = msg.config;
        this.device = msg.device;
        this.vramTotalMb = msg.vram_total_mb;
        cleanup();
        resolve(msg);
      };
      const onError = (err) => {
        cleanup();
        const e = err instanceof Error ? err : new Error(String(err));
        if (stderrBuf.length) e.stderr = stderrBuf.join('\n');
        reject(e);
      };
      const onExit = ({ code, signal }) => {
        cleanup();
        const tail = stderrBuf.length ? '\n--- python stderr ---\n' + stderrBuf.join('\n') : '';
        reject(new Error(`python exited before ready (code=${code}, signal=${signal})${tail}`));
      };
      this.once('ready', onReady);
      this.once('error', onError);
      this.once('exit', onExit);
    });
  }

  _onStdout(chunk) {
    this._stdoutBuf += chunk;
    let idx;
    while ((idx = this._stdoutBuf.indexOf('\n')) !== -1) {
      const line = this._stdoutBuf.slice(0, idx).trim();
      this._stdoutBuf = this._stdoutBuf.slice(idx + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        this.emit('log', line);
        continue;
      }
      if (!msg || typeof msg.type !== 'string') continue;
      if (msg.type === 'error') {
        const e = new Error(msg.message || 'unknown python error');
        e.payload = msg;
        this.emit('error', e);
      } else {
        this.emit(msg.type, msg);
      }
      this.emit('message', msg);
    }
  }

  _onStderr(chunk) {
    this._stderrBuf += chunk;
    let idx;
    while ((idx = this._stderrBuf.indexOf('\n')) !== -1) {
      const line = this._stderrBuf.slice(0, idx);
      this._stderrBuf = this._stderrBuf.slice(idx + 1);
      this.emit('stderr', line);
    }
  }

  send(obj) {
    if (!this.proc || this.proc.killed) throw new Error('Python process is not running');
    this.proc.stdin.write(JSON.stringify(obj) + '\n');
  }

  generate({ prompt, maxTokens = 256, temperature = 0.8, topK = 40 }) {
    this.send({
      cmd: 'generate',
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
    });
  }

  stop() {
    try {
      this.send({ cmd: 'stop' });
    } catch {}
  }

  shutdown() {
    try {
      this.send({ cmd: 'shutdown' });
    } catch {}
    if (this.proc) {
      const proc = this.proc;
      setTimeout(() => {
        if (!proc.killed) proc.kill();
      }, 500);
    }
  }
}

module.exports = PythonBridge;
module.exports.SCRIPT = SCRIPT;
