'use strict';

const { EventEmitter } = require('events');

let llamaModulePromise = null;
function loadLlamaModule() {
  if (!llamaModulePromise) llamaModulePromise = import('node-llama-cpp');
  return llamaModulePromise;
}

const STATS_INTERVAL_MS = 500;
const DEFAULT_CTX = 4096;

class GgufBackend extends EventEmitter {
  constructor(opts) {
    super();
    this.modelPath = opts.modelPath;
    this.ctx = opts.ctx || 0;
    this.cpu = !!opts.cpu;
    this.chat = !!opts.chat;
    this.ready = false;
    this.device = null;
    this.vramTotalMb = null;
    this.config = null;
    this._llama = null;
    this._model = null;
    this._context = null;
    this._sequence = null;
    this._completion = null;
    this._session = null;
    this._initialChatHistory = null;
    this._abort = null;
  }

  async start() {
    let mod;
    try {
      mod = await loadLlamaModule();
    } catch (err) {
      const e = new Error(
        'GGUF backend requires the optional dependency `node-llama-cpp` v3. ' +
          `Install with: npm i node-llama-cpp (load error: ${err.message || err})`
      );
      this.emit('error', e);
      throw e;
    }

    try {
      this._llama = await mod.getLlama({
        gpu: this.cpu ? false : 'auto',
        logLevel: mod.LlamaLogLevel.error,
        logger: (level, message) => this.emit('stderr', `[llama.cpp] ${message}`),
      });
      this._model = await this._llama.loadModel({ modelPath: this.modelPath });
      const ctxSize =
        this.ctx || Math.min(this._model.trainContextSize || DEFAULT_CTX, DEFAULT_CTX);
      this._context = await this._model.createContext({ contextSize: ctxSize });
      this._sequence = this._context.getSequence();
      if (this.chat) {
        this._session = new mod.LlamaChatSession({ contextSequence: this._sequence });
        try {
          this._initialChatHistory = JSON.parse(JSON.stringify(this._session.getChatHistory()));
        } catch {}
      } else {
        this._completion = new mod.LlamaCompletion({ contextSequence: this._sequence });
      }
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      this.emit('error', e);
      throw e;
    }

    this.device = this._llama.gpu ? String(this._llama.gpu) : 'cpu';
    try {
      const vram = await this._llama.getVramState();
      if (vram && vram.total) this.vramTotalMb = Math.round(vram.total / 1048576);
    } catch {}

    this.config = {
      backend: 'gguf',
      ctx: this._context.contextSize,
      block_size: this._context.contextSize,
      train_ctx: this._model.trainContextSize,
      model_path: this.modelPath,
    };
    this.ready = true;

    const ready = {
      type: 'ready',
      config: this.config,
      device: this.device,
      vram_total_mb: this.vramTotalMb,
    };
    setImmediate(() => this.emit('ready', ready));
    return ready;
  }

  async generate({
    prompt,
    maxTokens = 256,
    temperature = 0.8,
    topK = 40,
    topP = 1.0,
    minP = 0.0,
    repetitionPenalty = 1.0,
  }) {
    if (!this.ready) throw new Error('Backend not ready');
    const start = Date.now();
    let totalTokens = 0;
    let lastTickTokens = 0;
    let lastTickTime = start;
    let vramPending = false;
    this._abort = new AbortController();

    const statsTimer = setInterval(() => {
      const now = Date.now();
      const dt = now - lastTickTime;
      const tps = dt > 0 ? ((totalTokens - lastTickTokens) * 1000) / dt : 0;
      lastTickTokens = totalTokens;
      lastTickTime = now;
      const emitStats = (vramUsedMb) =>
        this.emit('stats', {
          type: 'stats',
          tokens_per_sec: tps,
          vram_used_mb: vramUsedMb,
          ctx_used: this._sequence ? this._sequence.nextTokenIndex : totalTokens,
          ctx_max: this.config.ctx,
        });
      if (this.vramTotalMb != null && !vramPending && this._llama) {
        vramPending = true;
        this._llama
          .getVramState()
          .then((v) => emitStats(v && v.total ? Math.round(v.used / 1048576) : null))
          .catch(() => emitStats(null))
          .finally(() => (vramPending = false));
      } else {
        emitStats(null);
      }
    }, STATS_INTERVAL_MS);

    const genOptions = {
      maxTokens,
      temperature,
      topK,
      topP,
      minP,
      repeatPenalty: repetitionPenalty && repetitionPenalty !== 1.0
        ? { penalty: repetitionPenalty }
        : false,
      signal: this._abort.signal,
      stopOnAbortSignal: true,
      onTextChunk: (text) => {
        totalTokens += 1;
        this.emit('token', { type: 'token', text });
      },
    };
    try {
      if (this.chat) {
        await this._session.prompt(prompt, genOptions);
      } else {
        await this._completion.generateCompletion(prompt, genOptions);
      }
    } catch (err) {
      clearInterval(statsTimer);
      if (err && (err.name === 'AbortError' || this._abort.signal.aborted)) {
      } else {
        this.emit('error', err instanceof Error ? err : new Error(String(err)));
        return;
      }
    }
    clearInterval(statsTimer);
    const elapsed = Date.now() - start;
    this.emit('done', { type: 'done', total_tokens: totalTokens, elapsed_ms: elapsed });
  }

  stop() {
    if (this._abort) this._abort.abort();
  }

  resetChat() {
    if (!this._session) return;
    try {
      this._session.setChatHistory(
        this._initialChatHistory ? JSON.parse(JSON.stringify(this._initialChatHistory)) : []
      );
    } catch {}
  }

  requestStats() {
    if (!this.ready || !this._llama || this._statsRequestPending) return;
    this._statsRequestPending = true;
    const emitStats = (vramUsedMb) => {
      this._statsRequestPending = false;
      if (!this.ready) return;
      this.emit('stats', {
        type: 'stats',
        tokens_per_sec: null,
        vram_used_mb: vramUsedMb,
        ctx_used: this._sequence ? this._sequence.nextTokenIndex : 0,
        ctx_max: this.config.ctx,
      });
    };
    if (this.vramTotalMb == null) return emitStats(null);
    this._llama
      .getVramState()
      .then((v) => emitStats(v && v.total ? Math.round(v.used / 1048576) : null))
      .catch(() => emitStats(null));
  }

  shutdown() {
    this.stop();
    this.ready = false;
    try {
      if (this._context) this._context.dispose();
      if (this._model) this._model.dispose();
    } catch {}
    this._completion = null;
    this._session = null;
    this._sequence = null;
    this._context = null;
    this._model = null;
  }
}

module.exports = GgufBackend;
