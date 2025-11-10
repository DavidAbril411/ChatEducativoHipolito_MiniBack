import express from 'express';
import cors from 'cors';
import fs from 'node:fs';
import { GoogleAuth } from 'google-auth-library';

const app = express();
const PORT = process.env.PORT || 3000;

// Vertex AI (Gemini) configuration
const VERTEX_API_KEY = process.env.VERTEX_API_KEY || process.env.GOOGLE_API_KEY || process.env.VERTEX_KEY;
const VERTEX_API_BASE = process.env.VERTEX_API_BASE || 'https://generativelanguage.googleapis.com/v1beta';
const VERTEX_DEFAULT_MODEL = process.env.VERTEX_MODEL || 'gemini-2.5-flash';

const vertexAuth = initialiseVertexAuth();

if (vertexAuth.mode === 'none') {
  console.warn('Vertex credentials not detected. Set VERTEX_API_KEY or VERTEX_SERVICE_ACCOUNT to enable responses.');
}

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'hipolito-chat-backend',
    provider: 'vertex',
    hasCredentials: vertexAuth.mode !== 'none',
    authMode: vertexAuth.mode,
    model: VERTEX_DEFAULT_MODEL
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    const { model, messages, temperature = 0.7, max_tokens = 180, top_p = 0.9, stream = false } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array is required' });
    }
    if (vertexAuth.mode === 'none') {
      return res.status(500).json({ error: 'Server missing Vertex credentials' });
    }
    if (stream) {
      return res.status(400).json({ error: 'stream mode is not supported with Vertex AI' });
    }

    const vertexModel = model || VERTEX_DEFAULT_MODEL;
    const { contents, systemInstructionParts } = mapMessagesToVertex(messages);
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
    if (systemInstructionParts.length > 0) {
      payload.systemInstruction = {
        role: 'system',
        parts: systemInstructionParts
      };
    }

    let vertexUrl = `${VERTEX_API_BASE}/models/${encodeURIComponent(vertexModel)}:generateContent`;
    const headers = { 'Content-Type': 'application/json' };

    if (vertexAuth.mode === 'apiKey') {
      vertexUrl += `?key=${vertexAuth.apiKey}`;
    } else if (vertexAuth.mode === 'serviceAccount') {
      const accessToken = await vertexAuth.getAccessToken();
      if (!accessToken) {
        return res.status(500).json({ error: 'Vertex authentication failed: missing access token' });
      }
      headers.Authorization = `Bearer ${accessToken}`;
    }

    const resp = await fetch(vertexUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    const json = await resp.json();
    if (!resp.ok) {
      console.error('Vertex API error:', {
        status: resp.status,
        statusText: resp.statusText,
        body: json
      });

      if (resp.status === 403) {
        json.hint = 'Vertex returned 403. Verify the service account has “Vertex AI User” (or Generative AI User) role and the Generative Language API is enabled.';
      }

      return res.status(resp.status).json(json);
    }
    
    console.log('Vertex API raw response', JSON.stringify(json, null, 2))
    const reply = extractVertexReply(json);
    return res.json(reply);
  } catch (err) {
    console.error('Error in /api/chat:', err);
    res.status(500).json({ error: 'internal_error', detail: String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Chat backend listening on :${PORT}`);
});

function initialiseVertexAuth() {
  if (VERTEX_API_KEY) {
    return {
      mode: 'apiKey',
      apiKey: VERTEX_API_KEY
    };
  }

  const credentials = loadServiceAccountCredentials();
  if (credentials) {
    const auth = new GoogleAuth({
      credentials,
      scopes: [
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/generative-language'
      ]
    });

    return {
      mode: 'serviceAccount',
      getAccessToken: async () => {
        try {
          const token = await auth.getAccessToken();
          if (typeof token === 'string' && token.length > 0) {
            return token;
          }
          if (token && typeof token.token === 'string') {
            return token.token;
          }
          return null;
        } catch (error) {
          console.error('Error obtaining Vertex access token:', error);
          return null;
        }
      }
    };
  }

  return { mode: 'none' };
}

function loadServiceAccountCredentials() {
  const inlineCredentials = process.env.VERTEX_SERVICE_ACCOUNT;
  if (inlineCredentials) {
    try {
      const parsed = JSON.parse(inlineCredentials);
      if (parsed && parsed.client_email && parsed.private_key) {
        return parsed;
      }
    } catch (error) {
      console.error('Failed to parse VERTEX_SERVICE_ACCOUNT JSON:', error);
    }
  }

  const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (credentialsPath) {
    try {
      if (fs.existsSync(credentialsPath)) {
        const fileContent = fs.readFileSync(credentialsPath, 'utf8');
        const parsed = JSON.parse(fileContent);
        if (parsed && parsed.client_email && parsed.private_key) {
          return parsed;
        }
      } else {
        console.warn(`GOOGLE_APPLICATION_CREDENTIALS points to missing file: ${credentialsPath}`);
      }
    } catch (error) {
      console.error('Failed to load service account credentials from file:', error);
    }
  }

  return null;
}

function mapMessagesToVertex(messages = []) {
  const contents = [];
  const systemInstructionParts = [];

  for (const msg of messages) {
    if (!msg || !msg.role) {
      continue;
    }
    const parts = normaliseMessageParts(msg.content);

    if (msg.role === 'system') {
      systemInstructionParts.push(...parts);
      continue;
    }

    contents.push({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: parts.length > 0 ? parts : [{ text: '' }]
    });
  }

  return { contents, systemInstructionParts };
}

function normaliseMessageParts(content) {
  if (content == null) {
    return [{ text: '' }];
  }
  if (Array.isArray(content)) {
    const parts = content
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
