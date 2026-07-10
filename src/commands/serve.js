'use strict';

const http = require('http');
const path = require('path');
const crypto = require('crypto');
const registry = require('../models/registry');
const PythonBridge = require('../inference/python_bridge');
const GgufBackend = require('../inference/gguf_loader');

async function serveCmd(modelArg, opts) {
  const resolved = registry.resolve(modelArg);
  if (!resolved) {
    console.error(`Model not found: ${modelArg}`);
    process.exit(1);
  }

  const kind = registry.detectKind(resolved);
  const modelLabel = path.basename(resolved, path.extname(resolved));

  let bridge;
  if (kind === 'pt') {
    bridge = new PythonBridge({
      checkpoint: resolved,
      python: opts.python,
      fp16: opts.fp16 !== false && !opts.cpu,
      cpu: !!opts.cpu,
      ctx: opts.ctx,
      tokenizer: opts.tokenizer,
      arch: opts.arch,
    });
  } else {
    bridge = new GgufBackend({
      modelPath: resolved,
      ctx: opts.ctx,
      cpu: !!opts.cpu,
    });
  }

  bridge.on('stderr', (line) => {
    if (line && line.trim()) process.stdout.write(`[py] ${line}\n`);
  });

  process.stdout.write(`Loading ${modelLabel}...\n`);
  try {
    await bridge.start();
  } catch (err) {
    console.error(`Failed to load model: ${err.message}`);
    process.exit(1);
  }
  process.stdout.write(`Ready. Model loaded on ${bridge.device || 'gguf'}.\n`);

  let isGenerating = false;

  const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    if (req.method === 'GET' && req.url === '/v1/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        object: 'list',
        data: [{ id: modelLabel, object: 'model', owned_by: 'local' }]
      }));
      return;
    }

    if (req.method === 'POST' && req.url === '/v1/chat/completions') {
      let bodyStr = '';
      req.on('data', chunk => { bodyStr += chunk; });
      req.on('end', () => {
        try {
          const body = JSON.parse(bodyStr);
          handleChatCompletion(req, res, body);
        } catch (err) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: { message: 'Invalid JSON' } }));
        }
      });
      return;
    }

    res.writeHead(404);
    res.end();
  });

  function formatMessages(messages) {
    if (!messages || !Array.isArray(messages)) return '';
    return messages.map(m => {
      if (m.role === 'system') return `${m.content}\n\n`;
      if (m.role === 'user') return `User: ${m.content}\n`;
      if (m.role === 'assistant') return `Assistant: ${m.content}\n`;
      return `${m.content}\n`;
    }).join('') + 'Assistant:';
  }

  function handleChatCompletion(req, res, body) {
    if (isGenerating) {
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'Server is busy processing another request.' } }));
      return;
    }
    isGenerating = true;

    const messages = body.messages || [];
    const stream = !!body.stream;
    const prompt = formatMessages(messages);
    
    const maxTokens = body.max_tokens || 256;
    const temperature = body.temperature ?? 0.8;
    const topP = body.top_p ?? 1.0;
    const topK = body.top_k ?? 40;
    const minP = body.min_p ?? 0.0;
    const repetitionPenalty = body.repetition_penalty ?? 1.0;

    const id = `chatcmpl-${crypto.randomUUID ? crypto.randomUUID() : crypto.randomBytes(8).toString('hex')}`;
    const created = Math.floor(Date.now() / 1000);

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      const onToken = (msg) => {
        res.write('data: ' + JSON.stringify({
          id,
          object: 'chat.completion.chunk',
          created,
          model: modelLabel,
          choices: [{ index: 0, delta: { content: msg.text }, finish_reason: null }]
        }) + '\n\n');
      };

      const onDone = (msg) => {
        res.write('data: ' + JSON.stringify({
          id,
          object: 'chat.completion.chunk',
          created,
          model: modelLabel,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
        }) + '\n\n');
        res.write('data: [DONE]\n\n');
        res.end();
        cleanup();
      };

      const onError = (err) => {
        res.write('data: ' + JSON.stringify({ error: String(err) }) + '\n\n');
        res.end();
        cleanup();
      };

      const cleanup = () => {
        bridge.removeListener('token', onToken);
        bridge.removeListener('done', onDone);
        bridge.removeListener('error', onError);
        isGenerating = false;
      };

      bridge.on('token', onToken);
      bridge.on('done', onDone);
      bridge.on('error', onError);

      bridge.generate({
        prompt,
        maxTokens,
        temperature,
        topK,
        topP,
        minP,
        repetitionPenalty
      });

    } else {
      let textOut = '';
      
      const onToken = (msg) => {
        textOut += msg.text;
      };

      const onDone = (msg) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          id,
          object: 'chat.completion',
          created,
          model: modelLabel,
          choices: [{
            index: 0,
            message: { role: 'assistant', content: textOut },
            finish_reason: 'stop'
          }],
          usage: {
            prompt_tokens: 0,
            completion_tokens: msg.total_tokens,
            total_tokens: msg.total_tokens
          }
        }));
        cleanup();
      };

      const onError = (err) => {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: String(err) } }));
        cleanup();
      };

      const cleanup = () => {
        bridge.removeListener('token', onToken);
        bridge.removeListener('done', onDone);
        bridge.removeListener('error', onError);
        isGenerating = false;
      };

      bridge.on('token', onToken);
      bridge.on('done', onDone);
      bridge.on('error', onError);

      bridge.generate({
        prompt,
        maxTokens,
        temperature,
        topK,
        topP,
        minP,
        repetitionPenalty
      });
    }
  }

  const port = opts.port || 11434;
  const host = opts.host || '127.0.0.1';
  server.listen(port, host, () => {
    console.log(`OpenAI compatible server listening on http://${host}:${port}/v1/chat/completions`);
  });
}

module.exports = serveCmd;
