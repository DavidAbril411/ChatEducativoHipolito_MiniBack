import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3000;
// Groq configuration (default behaviour retained for backwards compatibility)
const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.GROQ_KEY;
const GROQ_API_BASE = process.env.GROQ_API_BASE || 'https://api.groq.com/openai/v1';

// Vertex AI (Generative Language / Gemini) configuration
const VERTEX_API_KEY = process.env.VERTEX_API_KEY || process.env.GOOGLE_API_KEY || process.env.VERTEX_KEY;
const VERTEX_API_BASE = process.env.VERTEX_API_BASE || 'https://generativelanguage.googleapis.com/v1beta';
const VERTEX_DEFAULT_MODEL = process.env.VERTEX_MODEL || 'gemini-1.5-flash';

const CHAT_PROVIDER = (process.env.CHAT_PROVIDER || (VERTEX_API_KEY ? 'vertex' : 'groq')).toLowerCase();

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_req, res) => {
  const hasKey = CHAT_PROVIDER === 'vertex'
    ? Boolean(VERTEX_API_KEY)
    : Boolean(GROQ_API_KEY);

  res.json({ ok: true, service: 'hipolito-chat-backend', provider: CHAT_PROVIDER, hasKey });
});

app.post('/api/chat', async (req, res) => {
  try {
    const { model, messages, temperature = 0.7, max_tokens = 180, top_p = 0.9, stream = false } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array is required' });
    }
    if (CHAT_PROVIDER === 'vertex') {
      if (!VERTEX_API_KEY) {
        return res.status(500).json({ error: 'Server missing VERTEX_API_KEY' });
      }
      if (stream) {
        return res.status(400).json({ error: 'stream mode is not supported with Vertex AI' });
      }

      const vertexModel = model || VERTEX_DEFAULT_MODEL;
      const { contents, systemInstruction } = mapMessagesToVertex(messages);
      if (contents.length === 0) {
        return res.status(400).json({ error: 'Vertex requires at least one non-system message' });
      }

      const payload = {
        contents,
        generationConfig: {
          temperature,
          topP: top_p,
          maxOutputTokens: max_tokens
        }
      };
      if (systemInstruction) {
        payload.systemInstruction = systemInstruction;
      }

      const vertexUrl = `${VERTEX_API_BASE}/models/${encodeURIComponent(vertexModel)}:generateContent?key=${VERTEX_API_KEY}`;
      const resp = await fetch(vertexUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const json = await resp.json();
      if (!resp.ok) {
        return res.status(resp.status).json(json);
      }

      const reply = extractVertexReply(json);
      return res.json(reply);
    }

    if (!GROQ_API_KEY) {
      return res.status(500).json({ error: 'Server missing GROQ_API_KEY' });
    }

    const resp = await fetch(`${GROQ_API_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model, messages, temperature, max_tokens, top_p, stream })
    });
    const text = await resp.text();
    if (!resp.ok) {
      return res.status(resp.status).type('application/json').send(text);
    }
    res.type('application/json').send(text);
  } catch (err) {
    console.error('Error in /api/chat:', err);
    res.status(500).json({ error: 'internal_error', detail: String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Chat backend listening on :${PORT}`);
});

function mapMessagesToVertex(messages = []) {
  const contents = [];
  let systemInstruction;

  for (const msg of messages) {
    if (!msg || !msg.role) {
      continue;
    }
    const parts = normaliseMessageParts(msg.content);

    if (msg.role === 'system') {
      if (!systemInstruction) {
        systemInstruction = { role: 'system', parts: [] };
      }
      systemInstruction.parts.push(...parts);
      continue;
    }

    contents.push({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: parts.length > 0 ? parts : [{ text: '' }]
    });
  }

  return { contents, systemInstruction };
}

function normaliseMessageParts(content) {
  if (content == null) {
    return [{ text: '' }];
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') {
          return { text: item };
        }
        if (item && typeof item === 'object') {
          if (item.type === 'text' && typeof item.text === 'string') {
            return { text: item.text };
          }
          if (typeof item.content === 'string') {
            return { text: item.content };
          }
        }
        return null;
      })
      .filter(Boolean);
    return parts.length > 0 ? parts : [{ text: '' }];
  }
  if (typeof content === 'object') {
    if (typeof content.text === 'string') {
      return [{ text: content.text }];
    }
    if (typeof content.content === 'string') {
      return [{ text: content.content }];
    }
  }
  return [{ text: String(content) }];
}

function extractVertexReply(responseJson) {
  const { candidates = [], usageMetadata } = responseJson || {};
  const first = candidates.find((candidate) => candidate && candidate.content && candidate.content.parts && candidate.content.parts.length > 0);

  const text = first
    ? first.content.parts
        .map((part) => part.text || '')
        .join('')
        .trim()
    : '';

  return {
    id: responseJson?.name || `vertex-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: responseJson?.model || VERTEX_DEFAULT_MODEL,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: first?.finishReason || 'stop'
      }
    ],
    usage: usageMetadata ? {
      prompt_tokens: usageMetadata.promptTokenCount,
      completion_tokens: usageMetadata.candidatesTokenCount,
      total_tokens: usageMetadata.totalTokenCount
    } : undefined
  };
}
