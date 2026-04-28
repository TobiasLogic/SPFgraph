'use strict';

const { EventEmitter } = require('events');

class GgufBackend extends EventEmitter {
  constructor(opts) {
    super();
    this.modelPath = opts.modelPath;
    this.ctx = opts.ctx || 2048;
    this.temperature = opts.temperature ?? 0.8;
    this.topK = opts.topK ?? 40;
    this.ready = false;
    this._stop = false;
    this._llama = null;
    this._model = null;
    this._context = null;
    this._session = null;
    this.config = null;
  }

  async start() {
    let mod;
    try {
      mod = require('node-llama-cpp');
    } catch (err) {
      const e = new Error(
        'GGUF backend requires the optional dependency `node-llama-cpp`. ' +
          'Install with: npm i node-llama-cpp'
      );
      this.emit('error', e);
      throw e;
    }
    this._llama = mod;

    const { LlamaModel, LlamaContext, LlamaChatSession } = mod;
    this._model = new LlamaModel({ modelPath: this.modelPath });
    this._context = new LlamaContext({ model: this._model, contextSize: this.ctx });
    this._session = new LlamaChatSession({ context: this._context });

    this.config = {
      backend: 'gguf',
      ctx: this.ctx,
      model_path: this.modelPath,
    };
    this.ready = true;

    const ready = {
      type: 'ready',
      config: this.config,
      device: 'cpu/gpu (llama.cpp)',
      vram_total_mb: null,
    };
    setImmediate(() => this.emit('ready', ready));
    return ready;
  }

  async generate({ prompt, maxTokens = 256, temperature = 0.8, topK = 40 }) {
    if (!this.ready) throw new Error('Backend not ready');
    this._stop = false;
    const start = Date.now();
    let total = 0;
    let lastTickTokens = 0;
    let lastTickTime = start;

    try {
      await this._session.prompt(prompt, {
        temperature,
        topK,
        maxTokens,
        onToken: (chunk) => {
          if (this._stop) return false;
          const text = this._model.detokenize(chunk);
          total += chunk.length;
          this.emit('token', { type: 'token', text });

          const now = Date.now();
          const dt = now - lastTickTime;
          if (dt >= 200) {
            const tps = ((total - lastTickTokens) * 1000) / dt;
            lastTickTokens = total;
            lastTickTime = now;
            this.emit('stats', {
              type: 'stats',
              tokens_per_sec: tps,
              vram_used_mb: null,
              ctx_used: total,
              ctx_max: this.ctx,
            });
          }
          return true;
        },
      });
    } catch (err) {
      this.emit('error', err);
    }
    const elapsed = Date.now() - start;
    this.emit('done', { type: 'done', total_tokens: total, elapsed_ms: elapsed });
  }

  stop() {
    this._stop = true;
  }

  shutdown() {
    this._stop = true;
    this._session = null;
    this._context = null;
    this._model = null;
  }
}

module.exports = GgufBackend;
