'use strict';

const blessed = require('blessed');
const contrib = require('blessed-contrib');
const chalk = require('chalk');

const HISTORY_POINTS = 60;

function fmtParams(n) {
  if (!n) return '?';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
  return String(n);
}

function sanitize(s) {
  return String(s).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
}

class Dashboard {
  constructor({ backend, modelLabel, modelPath, opts }) {
    this.backend = backend;
    this.modelLabel = modelLabel;
    this.modelPath = modelPath;
    this.opts = opts || {};

    this.tpsHistory = [];
    this.vramHistory = [];
    this.lastStats = {
      tokens_per_sec: 0,
      vram_used_mb: 0,
      ctx_used: 0,
      ctx_max: backend.config?.block_size || backend.config?.ctx || 2048,
    };
    this.generating = false;

    this._messages = [];
    this._streaming = null;

    this._followBottom = true;
  }

  _scrollChatToBottom() {
    const total = this.chat.getScrollHeight();
    const viewport = Math.max(0, this.chat.height - 2);
    const target = Math.max(0, total - viewport);
    this.chat.setScroll(target);
  }

  _isAtBottom() {
    const total = this.chat.getScrollHeight();
    const viewport = Math.max(0, this.chat.height - 2);
    const max = Math.max(0, total - viewport);
    return this.chat.getScroll() >= max - 1;
  }

  _refreshChatLabel() {
    if (!this.chat) return;
    const total = this.chat.getScrollHeight();
    const viewport = Math.max(0, this.chat.height - 2);
    const max = Math.max(0, total - viewport);
    const cur = Math.min(this.chat.getScroll(), max);
    const label = this._followBottom
      ? ' chat ↓ '
      : ` chat ${cur}/${max} (scrolled — End to follow) `;
    this.chat.setLabel(label);
  }

  start() {
    this.screen = blessed.screen({
      smartCSR: true,
      title: 'zeroshot-run',
      fullUnicode: true,
      autoPadding: true,
    });
    if (typeof this.screen.enableMouse === 'function') {
      this.screen.enableMouse();
    }

    this.grid = new contrib.grid({ rows: 12, cols: 12, screen: this.screen });

    this.header = this.grid.set(0, 0, 2, 12, blessed.box, {
      tags: true,
      border: { type: 'line' },
      style: { border: { fg: 'cyan' } },
      label: ' zeroshot-run ',
      content: '',
    });

    this.tpsGraph = this.grid.set(2, 0, 5, 6, contrib.line, {
      label: ' tok/s over time ',
      showLegend: false,
      wholeNumbersOnly: false,
      xLabelPadding: 3,
      xPadding: 5,
      minY: 0,
      style: { text: 'white', baseline: 'white' },
    });
    this.vramGraph = this.grid.set(2, 6, 5, 6, contrib.line, {
      label: ' VRAM (MB) over time ',
      showLegend: false,
      wholeNumbersOnly: true,
      xLabelPadding: 3,
      xPadding: 5,
      minY: 0,
      style: { text: 'white', baseline: 'white' },
    });

    this.chat = this.grid.set(7, 0, 4, 12, blessed.box, {
      label: ' chat ',
      tags: false,
      border: { type: 'line' },
      style: { border: { fg: 'cyan' } },
      scrollable: true,
      alwaysScroll: true,
      scrollbar: { ch: ' ', style: { bg: 'cyan' } },
      content: '',
      wrap: true,
    });

    this.input = this.grid.set(11, 0, 1, 12, blessed.textbox, {
      label: ' you ',
      border: { type: 'line' },
      style: { border: { fg: 'yellow' }, focus: { border: { fg: 'green' } } },
      inputOnFocus: true,
      keys: true,
      mouse: true,
    });

    this._bindKeys();
    this._bindBackend();

    this._messages.push({
      role: 'system',
      text: 'Type a message and press Enter. Ctrl+C to quit. Esc to cancel a generation.',
    });

    this._renderHeader();
    this._renderGraphs();
    this._renderChat();
    this.input.focus();
    this.screen.render();

    return new Promise((resolve) => {
      this._exitResolver = resolve;
    });
  }

  _bindKeys() {

    const halfPage = () => {
      const h = typeof this.chat.height === 'number' ? this.chat.height : 10;
      return Math.max(1, Math.floor((h - 2) / 2));
    };
    const fullPage = () => {
      const h = typeof this.chat.height === 'number' ? this.chat.height : 10;
      return Math.max(1, h - 2);
    };
    const scrollUp = (n) => {
      this.chat.scroll(-n);
      this._followBottom = false;
      this._refreshChatLabel();
      this.screen.render();
    };
    const scrollDown = (n) => {
      this.chat.scroll(n);
      this._followBottom = this._isAtBottom();
      this._refreshChatLabel();
      this.screen.render();
    };
    const scrollToBottom = () => {
      this._followBottom = true;
      this._scrollChatToBottom();
      this._refreshChatLabel();
      this.screen.render();
    };
    const scrollToTop = () => {
      this._followBottom = false;
      this.chat.setScroll(0);
      this._refreshChatLabel();
      this.screen.render();
    };

    const handleEscape = () => {
      if (this.generating) {
        this.backend.stop();
        this._messages.push({ role: 'system', text: '[stop requested]' });
        this._renderChat();
        this.screen.render();
      }
    };

    this.input.on('cancel', () => {
      handleEscape();
      this.input.readInput();
    });

    const dbg = (() => {
      if (!process.env.ZEROSHOT_DEBUG) return null;
      const os = require('os');
      const path = require('path');
      const fs = require('fs');
      const dir = path.join(os.homedir(), '.zeroshot');
      try { fs.mkdirSync(dir, { recursive: true }); } catch {}
      const fd = fs.openSync(path.join(dir, 'debug.log'), 'a');
      fs.writeSync(fd, `\n--- session ${new Date().toISOString()} ---\n`);
      return (line) => { try { fs.writeSync(fd, line + '\n'); } catch {} };
    })();

    this.input.on('keypress', (ch, key) => {
      if (!key) return;
      if (dbg) dbg(`keypress name=${key.name} full=${key.full} ctrl=${key.ctrl} ch=${JSON.stringify(ch)}`);

      if (key.ctrl && key.name === 'c') return this.exit();

      if (key.name === 'escape') return handleEscape();

      if ((key.ctrl && key.name === 't') || (key.meta && key.name === 'up') || (key.shift && key.name === 'up') || key.name === 'f3') return scrollUp(1);
      if ((key.ctrl && key.name === 'u') || (key.meta && key.name === 'down') || (key.shift && key.name === 'down') || key.name === 'f4') return scrollDown(1);



      if (key.name === 'pageup') return scrollUp(halfPage());
      if (key.name === 'pagedown') return scrollDown(halfPage());

      if (key.name === 'home') return scrollToTop();
      if (key.name === 'end') return scrollToBottom();

      if (key.ctrl && key.name === 'b') return scrollUp(fullPage());
      if (key.ctrl && key.name === 'f') return scrollDown(fullPage());

      if (key.ctrl && key.name === 'g') return scrollToBottom();
    });

    this.screen.on('mouse', (data) => {
      if (dbg) dbg(`mouse action=${data && data.action} button=${data && data.button}`);
      if (!data) return;
      if (data.action === 'wheelup') return scrollUp(3);
      if (data.action === 'wheeldown') return scrollDown(3);
    });
    this.input.on('submit', (value) => {
      const text = (value || '').trim();
      this.input.clearValue();
      this.input.focus();
      this.screen.render();
      if (!text) return;
      if (text === '/quit' || text === '/exit') return this.exit();
      if (text === '/clear') {
        this._messages = [];
        this._streaming = null;
        this._followBottom = true;
        this.chat.setScroll(0);
        this._renderChat();
        this.screen.render();
        return;
      }
      this._sendPrompt(text);
    });
  }

  _bindBackend() {
    this.backend.on('token', (msg) => this._onToken(msg.text));
    this.backend.on('stats', (msg) => this._onStats(msg));
    this.backend.on('done', (msg) => this._onDone(msg));
    this.backend.on('error', (err) => {
      this._messages.push({ role: 'error', text: `error: ${err.message || err}` });
      this.generating = false;
      this._streaming = null;
      this._renderChat();
      this.screen.render();
    });
    this.backend.on('stderr', (line) => {
      if (line && line.trim()) {
        this._messages.push({ role: 'system', text: `[py] ${line}` });
        this._renderChat();
        this.screen.render();
      }
    });
    this.backend.on('exit', ({ code }) => {
      this._messages.push({ role: 'system', text: `[backend exited code=${code}]` });
      this._renderChat();
      this.screen.render();
    });
  }

  _sendPrompt(text) {
    if (this.generating) {
      this._messages.push({ role: 'system', text: '[busy — wait or press Esc]' });
      this._renderChat();
      this.screen.render();
      return;
    }
    this._messages.push({ role: 'you', text });
    this._streaming = { text: '' };
    this.generating = true;
    this._followBottom = true;
    this._renderChat();
    this.backend.generate({
      prompt: text,
      maxTokens: this.opts.maxTokens ?? 512,
      temperature: this.opts.temperature ?? 0.8,
      topK: this.opts.topK ?? 40,
    });
    this.screen.render();
  }

  _onToken(text) {
    if (!text) return;
    if (!this._streaming) this._streaming = { text: '' };
    this._streaming.text += sanitize(text);
    this._renderChat();
    this.screen.render();
  }

  _onStats(msg) {
    this.lastStats = {
      tokens_per_sec: msg.tokens_per_sec ?? 0,
      vram_used_mb: msg.vram_used_mb ?? this.lastStats.vram_used_mb,
      ctx_used: msg.ctx_used ?? this.lastStats.ctx_used,
      ctx_max: msg.ctx_max ?? this.lastStats.ctx_max,
    };
    this.tpsHistory.push(this.lastStats.tokens_per_sec);
    this.vramHistory.push(this.lastStats.vram_used_mb || 0);
    if (this.tpsHistory.length > HISTORY_POINTS) this.tpsHistory.shift();
    if (this.vramHistory.length > HISTORY_POINTS) this.vramHistory.shift();
    this._renderHeader();
    this._renderGraphs();
    this.screen.render();
  }

  _onDone(msg) {
    this.generating = false;
    if (this._streaming) {
      this._messages.push({ role: 'model', text: this._streaming.text });
      this._streaming = null;
    }
    const tps = msg.elapsed_ms > 0 ? (msg.total_tokens * 1000) / msg.elapsed_ms : 0;
    this._messages.push({
      role: 'system',
      text: `[done — ${msg.total_tokens} tokens in ${msg.elapsed_ms} ms · avg ${tps.toFixed(1)} tok/s]`,
    });
    this._renderChat();
    this.screen.render();
  }

  _renderHeader() {
    const cfg = this.backend.config || {};
    const params = fmtParams(cfg.n_params);
    const precision = cfg.fp16 ? 'fp16' : 'fp32';
    const device = this.backend.device || 'cpu';
    const vramTotal = this.backend.vramTotalMb;
    const vramUsed = this.lastStats.vram_used_mb;
    const vramStr =
      vramTotal != null
        ? `${(vramUsed / 1024).toFixed(1)}GB / ${(vramTotal / 1024).toFixed(1)}GB`
        : vramUsed
        ? `${vramUsed} MB`
        : 'n/a';
    const tps = this.lastStats.tokens_per_sec.toFixed(1);
    const ctxUsed = this.lastStats.ctx_used;
    const ctxMax = this.lastStats.ctx_max;

    const line1 = ` {bold}zeroshot-run{/bold}  |  {cyan-fg}${this.modelLabel}{/cyan-fg}  |  ${params} params  |  ${precision}  |  {gray-fg}${device}{/gray-fg}`;
    const line2 = ` VRAM: ${vramStr}   |  Speed: ${tps} tok/s   |  Ctx: ${ctxUsed}/${ctxMax}`;
    this.header.setContent(`${line1}\n${line2}`);
  }

  _renderGraphs() {
    if (this.tpsHistory.length >= 2) {
      const xs = this.tpsHistory.map((_, i) => String(i));
      this.tpsGraph.setData([
        { title: 'tok/s', style: { line: 'green' }, x: xs, y: this.tpsHistory.slice() },
      ]);
    }
    if (this.vramHistory.length >= 2) {
      const xs = this.vramHistory.map((_, i) => String(i));
      this.vramGraph.setData([
        { title: 'VRAM', style: { line: 'magenta' }, x: xs, y: this.vramHistory.slice() },
      ]);
    }
  }

  _renderChat() {
    const lines = [];
    for (const m of this._messages) {
      lines.push(this._formatMessage(m));
    }
    if (this._streaming) {
      lines.push(this._formatMessage({ role: 'model', text: this._streaming.text }));
    }
    const wasFollowing = this._followBottom;
    const oldScroll = this.chat.getScroll();
    this.chat.setContent(lines.join('\n'));
    if (wasFollowing) {
      this._scrollChatToBottom();
    } else {
      this.chat.setScroll(oldScroll);
    }
    this._refreshChatLabel();
  }

  _formatMessage(m) {
    const text = sanitize(m.text);
    switch (m.role) {
      case 'you':
        return `${chalk.yellow.bold('you')}   ${text}`;
      case 'model':
        return `${chalk.green.bold('model')} ${text}`;
      case 'error':
        return chalk.red(text);
      case 'system':
      default:
        return chalk.cyan(text);
    }
  }

  exit() {
    try {
      this.backend.shutdown();
    } catch {}
    if (this.screen) this.screen.destroy();
    if (this._exitResolver) this._exitResolver();
    process.exit(0);
  }
}

module.exports = Dashboard;
