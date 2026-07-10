'use strict';

const { spawn } = require('child_process');
const path = require('path');
const { EventEmitter } = require('events');

const SCRIPT = path.resolve(__dirname, '..', '..', 'python', 'inference.py');

const CHAT_STOP_MARKERS = ['\nUser:', '\nAssistant:'];

class PythonBridge extends EventEmitter {
  constructor(opts) {
    super();
    this.checkpoint = opts.checkpoint;
    this.python = opts.python || 'python';
    this.fp16 = opts.fp16 !== false;
    this.cpu = !!opts.cpu;
    this.ctx = opts.ctx;
    this.tokenizer = opts.tokenizer;
    this.arch = opts.arch;
    this.chat = !!opts.chat;
    this.proc = null;
    this.ready = false;
    this.config = null;
    this.device = null;
    this.vramTotalMb = null;
    this._stdoutBuf = '';
    this._stderrBuf = '';
    this._history = [];
    this._pending = null;
    this._softStopped = false;

    if (this.chat) {
      this.on('token', (msg) => {
        if (this._pending == null) return;
        this._pending += msg.text || '';
        if (!this._softStopped && CHAT_STOP_MARKERS.some((m) => this._pending.includes(m))) {
          this._softStopped = true;
          this.stop();
        }
      });
      this.on('done', () => {
        if (this._pending == null) return;
        let text = this._pending;
        for (const m of CHAT_STOP_MARKERS) {
          const i = text.indexOf(m);
          if (i !== -1) text = text.slice(0, i);
        }
        this._history.push({ role: 'assistant', text: text.trim() });
        this._pending = null;
      });
      this.on('error', () => {
        this._pending = null;
      });
    }
  }

  start() {
    const args = [SCRIPT, '--checkpoint', this.checkpoint, '--server'];
    if (this.fp16) args.push('--fp16');
    if (this.cpu) args.push('--cpu');
    if (this.ctx) args.push('--ctx', String(this.ctx));
    if (this.tokenizer) args.push('--tokenizer', this.tokenizer);
    if (this.arch) args.push('--arch', this.arch);

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

    const killChild = () => {
      try {
        if (this.proc && !this.proc.killed) this.proc.kill('SIGKILL');
      } catch {}
    };
    process.once('exit', killChild);

    this.proc.on('exit', (code, signal) => {
      this.ready = false;
      process.removeListener('exit', killChild);
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

  generate({
    prompt,
    maxTokens = 256,
    temperature = 0.8,
    topK = 40,
    topP = 1.0,
    minP = 0.0,
    repetitionPenalty = 1.0,
  }) {
    let fullPrompt = prompt;
    if (this.chat) {
      const parts = [];
      for (const m of this._history) {
        parts.push(`${m.role === 'user' ? 'User' : 'Assistant'}: ${m.text}`);
      }
      parts.push(`User: ${prompt}`);
      parts.push('Assistant:');
      fullPrompt = parts.join('\n');
      this._history.push({ role: 'user', text: prompt });
      this._pending = '';
      this._softStopped = false;
    }
    this.send({
      cmd: 'generate',
      prompt: fullPrompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      min_p: minP,
      repetition_penalty: repetitionPenalty,
    });
  }

  resetChat() {
    this._history = [];
    this._pending = null;
    this._softStopped = false;
  }

  stop() {
    try {
      this.send({ cmd: 'stop' });
    } catch {}
  }

  requestStats() {
    if (!this.ready) return;
    try {
      this.send({ cmd: 'stats' });
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
