import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.GROQ_KEY;
const GROQ_API_BASE = process.env.GROQ_API_BASE || 'https://api.groq.com/openai/v1';

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'hipolito-chat-backend', hasKey: !!GROQ_API_KEY });
});

app.post('/api/chat', async (req, res) => {
  try {
    if (!GROQ_API_KEY) {
      return res.status(500).json({ error: 'Server missing GROQ_API_KEY' });
    }
    const { model, messages, temperature = 0.7, max_tokens = 180, top_p = 0.9, stream = false } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array is required' });
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
