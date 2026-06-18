# Voice Agent Browser Demo

Small browser-based demo for the medical appointment reminder voice agent.

The current project state is the version that runs locally in a browser:

1. The user clicks **Start Demo**.
2. Browser microphone audio streams over WebSocket.
3. Deepgram streams speech-to-text results.
4. The LangGraph supervisor picks the next conversation node.
5. Fast local rules and hospital RAG answer grounded questions when possible.
6. Gemini or Groq generates the agent response only when needed.
7. Cartesia or Deepgram generates speech audio.
8. The browser plays the AI reply and keeps listening until the user ends the call.

## Setup

Use Python 3.9 or newer.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your real provider keys.

Required:

- `DEEPGRAM_API_KEY`
- `GOOGLE_API_KEY` if `LLM_BACKEND=gemini`
- `GROQ_API_KEY` if `LLM_BACKEND=groq`

Optional:

- `CARTESIA_API_KEY`
- `CARTESIA_VOICE_ID`

If Cartesia is not configured, the demo uses Deepgram TTS.

## Run

```powershell
.\start_web_demo.ps1
```

Open:

```text
http://localhost:8002
```

Allow microphone access, click **Start Demo**, and talk to the agent.

## Files

| File | Purpose |
| --- | --- |
| `web_voice_demo.py` | FastAPI demo backend, WebSocket STT bridge, LLM turn handling, TTS audio serving |
| `web_voice_demo.html` | Browser frontend |
| `web_voice_worklet.js` | Low-latency PCM microphone streaming |
| `start_web_demo.ps1` | Windows launcher |
| `utils/supervisor.py` | LangGraph conversation routing and node responses |
| `utils/rag_service.py` | Lightweight hospital RAG, cache lookup, and background prefetch |
| `utils/config.py` | Environment config |
| `utils/helpers.py` | TTS text cleanup |
| `data/hospital_kb.json` | Local hospital, doctor, department, and policy knowledge base |
| `docs/hospital_knowledge_details.md` | Human-readable hospital KB and RAG design notes |

## Demo Flow Snapshot

The preserved demo flow currently handles greeting, availability, appointment
review, general questions, reschedule requests, and wrap-up.

## Hospital RAG

The demo includes a lightweight hospital RAG layer inspired by the VoiceAgentRAG
paper. It is intentionally local and simple because the demo knowledge base is
small.

Flow:

```text
user question
  -> local fast path
  -> hospital RAG cache lookup
  -> local KB search
  -> grounded answer
  -> background prefetch for likely follow-up facts
```

Example questions:

- "What is Dr. Smith's full name?"
- "What is Dr. Smith specialized in?"
- "Which doctors are available?"
- "What are the clinic hours?"
- "Where is the clinic?"
- "Can I come early and wait in the lobby?"

Debug endpoint:

```text
http://localhost:8002/api/rag-debug?q=doctor%20specialty
```
