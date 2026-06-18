"""
Sarah Voice Agent - Main FastAPI Application

Appointment reminder voice AI system using:
- FastAPI WebSocket for Telnyx media streaming
- Deepgram Nova-3 for speech-to-text
- vLLM for LLM responses
- Cartesia/Deepgram/XTTS for text-to-speech
- Telnyx for call management
"""

import os
import asyncio
import json
import base64
import traceback
import logging
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import re
import struct
import websockets as websockets_client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from starlette.websockets import WebSocketState
import httpx

# ============ Filler phrases (fired immediately while LLM processes) ============
import random as _random

_FILLER_PHRASES: Dict[str, list] = {
    "greeting":           ["Sure, one moment...",    "Just a moment..."],
    "verify_patient":     ["One moment please...",   "Let me check..."],
    "availability_check": ["Of course...",           "Sure..."],
    "appointment_review": ["Let me check on that...", "One moment..."],
    "confirm":            ["Of course...",           "Sure thing..."],
    "cancel":             ["I understand...",        "Of course..."],
    "reschedule":         ["Let me look at that...", "Sure, one moment..."],
    "general_question":   ["Good question...",       "Let me think..."],
    "wrap_up":            ["Of course...",           "Sure..."],
}

async def _play_filler(call_control_id: str, phrase: str) -> None:
    """Play a pre-generated filler phrase. Phrase is produced by the caller
    (already awaited from generate_smart_filler) so this is pure TTS+playback."""
    try:
        await play_speech(call_control_id, phrase)
    except Exception as e:
        logger.debug(f"[FILLER] play failed: {e}")


# ============ Backchannel detection ============
# Short acknowledgements ("yeah", "uh-huh") during Sarah's playback should NOT
# trigger barge-in — the caller is just listening, not trying to interrupt.
_RE_BACKCHANNEL = re.compile(
    r'^(yeah|yes|yep|ok|okay|uh.huh|mm.hmm|mhm|right|i see|got it|sure|alright|fine)[.!?,\s]*$',
    re.I,
)

# ============ Valid State Transitions (enforced in code — LLM cannot regress) ============
# Any LLM-suggested next_state not in this set is rejected and the current state is kept.
STATE_TRANSITIONS: Dict[str, set] = {
    "greeting":           {"greeting", "verify_patient", "availability_check", "wrong_person", "ended"},
    "verify_patient":     {"verify_patient", "appointment_review", "callback_request", "wrong_person", "ended"},
    "availability_check": {"availability_check", "verify_patient", "appointment_review", "callback_request", "ended"},
    "wrong_person":       {"wrong_person", "greeting", "ended"},
    "callback_request":   {"callback_request", "ended"},
    "appointment_review": {"appointment_review", "confirm", "reschedule", "cancel", "wrap_up", "ended"},
    "confirm":            {"confirm", "wrap_up", "ended"},
    "reschedule":         {"reschedule", "wrap_up", "ended"},
    "cancel":             {"cancel", "wrap_up", "ended"},
    "wrap_up":            {"wrap_up", "general_question", "cancel", "reschedule", "ended"},
    "general_question":   {"general_question", "appointment_review", "wrap_up", "ended"},
    "ended":              {"ended"},
}

# Import from utils package
from utils.supervisor import generate_smart_filler, decide_node, build_agent_state, run_node_for_state
from utils.tts import prewarm_cartesia_websocket, close_cartesia_websocket
from utils.turn_detector import TurnDetector
from utils import (
    # Config
    DEEPGRAM_API_KEY,
    DEEPGRAM_URL,
    TELNYX_API_KEY,
    WEBHOOK_BASE_URL,
    AUDIO_CACHE_TTL,
    AGENT_NAME,
    STT_MIN_CONFIDENCE,
    STT_MIN_WORDS,
    STT_INTRO_GATE_SECS,
    MISSING_KEYS,
    TELNYX_CONNECTION_ID,
    TELNYX_ATC_NUMBER,
    # Models
    CallState,
    CallLogger,
    # TTS
    text_to_speech_wav,
    TTS_ENGINE,
    # Tools
    execute_tool_action,
    speak_via_telnyx,
    # Database
    CallDatabase,
    PostgresDatabase,
    # Helpers
    clean_text_for_tts,
)

from fastapi.responses import Response as FastAPIResponse, StreamingResponse

# First line of outbound greeting — static except agent name; TTS is prefetched at startup / on dial.
GREETING_OPENING_TEXT = f"Hi! This is {AGENT_NAME}. Thanks for picking up!"

# ============ Logging ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _iso_utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _log_turn_timing(call_state_dict: dict, event: str, **fields: Any) -> None:
    """Structured turn timing logs with wall timestamp + per-call context."""
    cid = str(call_state_dict.get("_call_id", "unknown"))
    turn_id = call_state_dict.get("_active_turn_id", "-")
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(
        f"[TURN-TIMING] ts={_iso_utc_ts()} call={cid[:12]} turn={turn_id} event={event} {payload}".strip()
    )

def _log_stream_timing(event: str, token: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(
        f"[STREAM-TIMING] ts={_iso_utc_ts()} token={token[:8]} event={event} {payload}".strip()
    )


# ============ Global State ============
db = CallDatabase()
pg_db = PostgresDatabase()
audio_cache: Dict[str, tuple] = {}

# Prefetched WAV for GREETING_OPENING_TEXT (avoids ~1s TTS before first audio on answer).
_greeting_opening_wav: Optional[bytes] = None
_greeting_opening_wav_lock = asyncio.Lock()

# Turn detection — shared across all calls (model loaded once at startup)
_turn_detector = TurnDetector()

# Persistent HTTP clients — created in lifespan, reused across all requests.
# Saves TCP+TLS handshake overhead (~75ms) on every Telnyx and LLM API call.
_telnyx_client: Optional[httpx.AsyncClient] = None
_llm_client: Optional[httpx.AsyncClient] = None

# ============ Startup/Shutdown ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown"""
    global _telnyx_client, _llm_client
    logger.info("🚀 Startup: Initializing services...")

    if MISSING_KEYS:
        logger.warning(f"⚠️  Missing config: {', '.join(MISSING_KEYS)}")

    # Persistent HTTP clients with connection pooling
    _telnyx_client = httpx.AsyncClient(
        base_url="https://api.telnyx.com",
        headers={"Authorization": f"Bearer {TELNYX_API_KEY}", "Content-Type": "application/json"},
        timeout=10.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    _llm_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    logger.info("  ✓ Persistent HTTP clients initialized")
    if TTS_ENGINE == "cartesia":
        await prewarm_cartesia_websocket()
    asyncio.create_task(ensure_greeting_opening_wav_cached())

    # Connect to Redis
    connected = await db.connect()
    if not connected:
        logger.warning("⚠️  Redis connection failed - using in-memory state only")

    # Connect to PostgreSQL
    pg_connected = await pg_db.connect()
    if not pg_connected:
        logger.warning("⚠️  PostgreSQL connection failed - tool saves will be Redis-only")

    yield

    # Shutdown
    logger.info("🛑 Shutdown: Cleaning up...")
    await db.disconnect()
    await pg_db.disconnect()
    await close_cartesia_websocket()
    await _telnyx_client.aclose()
    await _llm_client.aclose()


app = FastAPI(title="Sarah Voice Agent", lifespan=lifespan)


# ============ Health Check ============
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# ============ Audio Serving ============
@app.get("/audio/{token}")
async def serve_audio(token: str):
    """Serve cached audio by token - supports both buffered and chunked streaming.
    For chunked playback, streams chunks from queue as they're generated.
    For buffered playback, serves pre-cached bytes.
    """
    entry = audio_cache.get(token)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audio not found")

    # Chunked streaming case (dict with queue)
    if isinstance(entry, dict) and "queue" in entry:
        async def chunk_generator():
            queue = entry["queue"]
            first_chunk_sent = False
            playback_start_t = entry.get("playback_start_t")
            first_tts_chunk_t = entry.get("first_tts_chunk_t")
            while True:
                chunk = await queue.get()
                if chunk is None:
                    _log_stream_timing(
                        "audio_stream_eof",
                        token,
                        total_chunks=entry.get("chunks_sent", 0),
                        total_bytes=entry.get("bytes_sent", 0),
                    )
                    break
                if not first_chunk_sent:
                    first_chunk_sent = True
                    now = time.perf_counter()
                    entry["first_chunk_served_t"] = now
                    fields: Dict[str, Any] = {"chunk_bytes": len(chunk)}
                    if playback_start_t:
                        fields["ms_from_playback_start"] = f"{(now - playback_start_t) * 1000:.0f}"
                    if first_tts_chunk_t:
                        fields["ms_from_first_tts_chunk"] = f"{(now - first_tts_chunk_t) * 1000:.0f}"
                    _log_stream_timing("audio_stream_first_chunk_out", token, **fields)
                entry["chunks_sent"] = int(entry.get("chunks_sent", 0)) + 1
                entry["bytes_sent"] = int(entry.get("bytes_sent", 0)) + len(chunk)
                yield chunk
        return StreamingResponse(
            chunk_generator(),
            media_type="audio/wav",
            headers={"Transfer-Encoding": "chunked"},
        )
    
    # Buffered case (tuple with bytes and expiry)
    wav_bytes, expiry = entry
    if time.time() > expiry:
        audio_cache.pop(token, None)
        raise HTTPException(status_code=404, detail="Audio expired")

    return FastAPIResponse(content=wav_bytes, media_type="audio/wav")


# ============ Audio Playback (Telnyx) ============
def _wav_duration_secs(wav_bytes: bytes) -> float:
    """Return approximate playback duration from a PCM WAV buffer."""
    try:
        num_channels    = struct.unpack_from('<H', wav_bytes, 22)[0]
        sample_rate     = struct.unpack_from('<I', wav_bytes, 24)[0]
        bits_per_sample = struct.unpack_from('<H', wav_bytes, 34)[0]
        pcm_data = max(0, len(wav_bytes) - 44)
        return pcm_data / (sample_rate * num_channels * (bits_per_sample // 8))
    except Exception:
        return max(0.0, len(wav_bytes) - 44) / 16000.0


async def play_speech(
    call_control_id: str,
    text: str,
    *,
    precomputed_wav: Optional[bytes] = None,
) -> tuple[bool, float]:
    """Play speech via Telnyx. Returns (success, estimated_duration_secs).

    Hot path (cache hit):  0ms TTS — serve from memory, call playback_start ~250ms.
    Cold path cartesia:    WebSocket stream — Telnyx gets first audio in ~150ms.
    Cold path deepgram/xtts: REST bytes — get full WAV then call playback_start.
    After cold-path generation, result saved to cache for future calls.

    If precomputed_wav is set, skips TTS and uses buffered playback (any TTS_ENGINE).
    """
    tts_start = time.perf_counter()
    token = str(uuid.uuid4())
    audio_url = f"{WEBHOOK_BASE_URL}/audio/{token}"
    clean_text = clean_text_for_tts(text)

    # ── Pre-baked WAV (e.g. prefetched opening line) ──────────────────────────
    if precomputed_wav is not None:
        wav = precomputed_wav
        if not wav:
            logger.error("[TTS] precomputed_wav empty")
            return False, 0.0
        audio_cache[token] = (wav, time.time() + AUDIO_CACHE_TTL)
        try:
            client = _telnyx_client or httpx.AsyncClient()
            r = await client.post(
                f"/v2/calls/{call_control_id}/actions/playback_start",
                json={"audio_url": audio_url, "loop": 1},
            )
            elapsed = time.perf_counter() - tts_start
            logger.info(
                f"[LATENCY] Telnyx playback_start (precached): {elapsed*1000:.0f}ms ({len(wav)} bytes)"
            )
            if r.status_code in (200, 202):
                logger.info(f"[PLAY] playback_start OK (precached) — {text[:60]}")
                return True, _wav_duration_secs(wav)
            logger.warning(f"[PLAY] playback_start failed ({r.status_code})")
        except Exception as e:
            logger.warning(f"[PLAY] playback_start error: {e}")
        return False, 0.0

    # ── Bytes path (deepgram / xtts) ─────────────────────────────────────────
    if TTS_ENGINE != "cartesia":
        wav = await text_to_speech_wav(clean_text)
        if not wav:
            logger.error(f"[TTS] All engines failed for: {clean_text[:60]}")
            return False, 0.0
        audio_cache[token] = (wav, time.time() + AUDIO_CACHE_TTL)
        try:
            client = _telnyx_client or httpx.AsyncClient()
            r = await client.post(
                f"/v2/calls/{call_control_id}/actions/playback_start",
                json={"audio_url": audio_url, "loop": 1},
            )
            elapsed = time.perf_counter() - tts_start
            logger.info(f"[LATENCY] Telnyx playback_start: {elapsed*1000:.0f}ms ({len(wav)} bytes, {TTS_ENGINE})")
            if r.status_code in (200, 202):
                logger.info(f"[PLAY] playback_start OK — {text[:60]}")
                return True, _wav_duration_secs(wav)
            logger.warning(f"[PLAY] playback_start failed ({r.status_code})")
        except Exception as e:
            logger.warning(f"[PLAY] playback_start error: {e}")
        return False, 0.0

    # ── Cache miss: Cartesia WebSocket stream ─────────────────────────────────
    from utils.tts import text_to_speech_cartesia_websocket
    chunk_queue = asyncio.Queue()
    collected_chunks: list[bytes] = []
    total_bytes = [0]
    first_chunk_meta = {"t": None}

    async def collect_chunks():
        try:
            async for chunk in text_to_speech_cartesia_websocket(clean_text):
                if first_chunk_meta["t"] is None:
                    first_chunk_meta["t"] = time.perf_counter()
                    entry_now = audio_cache.get(token)
                    if isinstance(entry_now, dict):
                        entry_now["first_tts_chunk_t"] = first_chunk_meta["t"]
                    _log_stream_timing(
                        "tts_first_chunk_generated",
                        token,
                        chunk_bytes=len(chunk),
                    )
                await chunk_queue.put(chunk)
                collected_chunks.append(chunk)
                total_bytes[0] += len(chunk)
        except Exception as e:
            logger.warning(f"[CHUNKED] Collection failed: {e}")
        finally:
            await chunk_queue.put(None)  # EOF signal

    collection_task = asyncio.create_task(collect_chunks())
    audio_cache[token] = {
        "queue": chunk_queue,
        "collected": total_bytes,
        "playback_start_t": None,
        "first_tts_chunk_t": None,
        "first_chunk_served_t": None,
        "chunks_sent": 0,
        "bytes_sent": 0,
    }

    playback_start_t = time.perf_counter()
    try:
        client = _telnyx_client or httpx.AsyncClient()
        _log_stream_timing("playback_start_request", token, text_chars=len(clean_text))
        r = await client.post(
            f"/v2/calls/{call_control_id}/actions/playback_start",
            json={"audio_url": audio_url, "loop": 1},
        )
        pb_elapsed = time.perf_counter() - playback_start_t
        logger.info(f"[LATENCY] Telnyx playback_start: {pb_elapsed*1000:.0f}ms (ws-stream)")
        _log_stream_timing(
            "playback_start_response",
            token,
            status_code=r.status_code,
            duration_ms=f"{pb_elapsed * 1000:.0f}",
        )
        entry = audio_cache.get(token)
        if isinstance(entry, dict):
            entry["playback_start_t"] = playback_start_t

        if r.status_code not in (200, 202):
            logger.warning(f"[PLAY] playback_start failed ({r.status_code})")
            audio_cache.pop(token, None)
            collection_task.cancel()
            return False, 0.0

        logger.info(f"[PLAY] playback_start OK — {text[:60]}")

        # Wait for generation to complete in background while Telnyx streams
        await asyncio.wait_for(collection_task, timeout=30.0)
        entry = audio_cache.get(token)
        if isinstance(entry, dict):
            entry["first_tts_chunk_t"] = first_chunk_meta["t"]
            first_out = entry.get("first_chunk_served_t")
            if first_out and first_chunk_meta["t"]:
                _log_stream_timing(
                    "tts_to_first_http_chunk_gap",
                    token,
                    ms=f"{(first_out - first_chunk_meta['t']) * 1000:.0f}",
                )
        tts_elapsed = time.perf_counter() - tts_start
        logger.info(f"[LATENCY] TTS ws-stream total: {tts_elapsed*1000:.0f}ms ({total_bytes[0]} bytes)")
        _log_stream_timing(
            "tts_stream_generation_complete",
            token,
            total_duration_ms=f"{tts_elapsed * 1000:.0f}",
            bytes=total_bytes[0],
        )

        # Replace queue entry with static bytes so any secondary Telnyx edge
        # server fetch (common: 2+ edge IPs per call) gets served from bytes
        # instead of blocking forever on an exhausted queue.
        wav_bytes_full = b"".join(collected_chunks)
        audio_cache[token] = (wav_bytes_full, time.time() + AUDIO_CACHE_TTL)

        return True, _wav_duration_secs(wav_bytes_full)

    except asyncio.TimeoutError:
        logger.warning("[PLAY] TTS generation timeout")
        audio_cache.pop(token, None)
        collection_task.cancel()
    except Exception as e:
        logger.warning(f"[PLAY] playback_start error: {e}")
        audio_cache.pop(token, None)
        collection_task.cancel()

    # Fallback: Telnyx built-in TTS
    fallback_start = time.perf_counter()
    ok = await speak_via_telnyx(call_control_id, text)
    logger.info(f"[LATENCY] Telnyx speak fallback: {(time.perf_counter()-fallback_start)*1000:.0f}ms")
    return ok, 0.0


async def ensure_greeting_opening_wav_cached() -> None:
    """Generate and retain WAV for GREETING_OPENING_TEXT so the first audio on answer is instant."""
    global _greeting_opening_wav
    if _greeting_opening_wav:
        return
    async with _greeting_opening_wav_lock:
        if _greeting_opening_wav:
            return
        logger.info("[GREETING] Prefetch opening-line TTS (runs at startup / on outbound dial)...")
        wav = await text_to_speech_wav(GREETING_OPENING_TEXT)
        if wav:
            _greeting_opening_wav = wav
            logger.info(f"[GREETING] Opening line cached ({len(wav)} bytes)")
        else:
            logger.warning("[GREETING] Opening prefetch failed — will use live TTS on the call")


async def _play_greeting_opening_line(call_control_id: str) -> tuple[bool, float]:
    await ensure_greeting_opening_wav_cached()
    wav = _greeting_opening_wav
    if wav:
        return await play_speech(
            call_control_id, GREETING_OPENING_TEXT, precomputed_wav=wav
        )
    return await play_speech(call_control_id, GREETING_OPENING_TEXT)


async def _play_split_greeting_sequence(
    call_control_id: str,
    redis_task: asyncio.Task,
) -> None:
    """Opening audio first (prefetched), then personalized suffix after Redis resolves."""
    try:
        await _play_greeting_opening_line(call_control_id)
        patient_ctx = await redis_task
        patient_name = patient_ctx.get("name", "there")
        clinic_name = patient_ctx.get("clinic_name", "the clinic")
        suffix = (
            f"I'm calling from {clinic_name}. Am I speaking with {patient_name}?"
        )
        await play_speech(call_control_id, suffix)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(f"[GREETING] Split greeting sequence failed: {e}")


async def load_patient_context_for_stream(
    called_number: Optional[str],
    calling_number: Optional[str],
) -> dict:
    """Redis patient:{phone} lookup with the same fallbacks as the media stream start handler."""
    default_ctx: dict = {
        "name": "Patient",
        "phone": calling_number or "",
        "appointment": "your appointment",
        "clinic_name": "our clinic",
        "provider_name": "your doctor",
    }
    if not db.client:
        return default_ctx
    try:
        patient_context = await db.client.hgetall(f"patient:{called_number}")
        if not patient_context:
            patient_context = await db.client.hgetall(f"patient:{calling_number}")
        if patient_context:
            logger.info(f"  ✓ Found patient: {patient_context.get('name', 'Unknown')}")
            return patient_context
        return default_ctx
    except Exception as e:
        logger.error(f"  ⚠️  Redis lookup failed: {e}")
        return {
            "name": "Patient",
            "phone": calling_number or "",
            "appointment": "your appointment",
            "clinic_name": "our clinic",
            "provider_name": "your doctor",
        }


# ============ Silence Watchdog (Vapi-style silenceTimeoutSeconds) ============
async def _silence_watchdog(
    call_state_dict: dict,
    call_control_id: str,
) -> None:
    """
    Hang up if the user is silent for too long — mirrors Vapi.ai's silenceTimeoutSeconds.
    - 20 s initial wait after greeting (covers TTS playback + response lag)
    - If still in greeting state → likely voicemail / unanswered → just hangup silently
    - 15 s mid-call silence → play "Are you still there?", then 10 s final grace → hangup
    """
    INITIAL_WAIT   = 20.0   # seconds to wait for FIRST user speech after greeting
    MID_CALL_LIMIT = 20.0   # seconds of silence allowed mid-conversation
    FINAL_GRACE    =  10.0  # seconds after re-prompt before force-hangup

    await asyncio.sleep(INITIAL_WAIT)

    while True:
        state = call_state_dict.get("state", CallState.GREETING)
        if state == CallState.ENDED:
            return

        now          = time.time()
        last_speech  = call_state_dict.get("_last_speech_time", now)
        silence_secs = now - last_speech

        if state == CallState.GREETING and silence_secs >= INITIAL_WAIT:
            # No response to greeting → likely voicemail or the person hung up immediately
            logger.info(f"[SILENCE] {silence_secs:.0f}s no speech in greeting — "
                        f"likely unanswered/voicemail, hanging up")
            call_state_dict["state"] = CallState.ENDED
            try:
                _hc = _telnyx_client or httpx.AsyncClient()
                await _hc.post(f"/v2/calls/{call_control_id}/actions/hangup")
            except Exception as _e:
                logger.debug(f"[SILENCE] hangup error: {_e}")
            return

        if state != CallState.GREETING and silence_secs >= MID_CALL_LIMIT:
            # Mid-call silence → re-prompt once, then hangup if still no response
            logger.info(f"[SILENCE] {silence_secs:.0f}s mid-call silence — re-prompting")
            call_state_dict["_last_speech_time"] = time.time()  # reset clock
            try:
                await play_speech(call_control_id, "Hello, are you still there?")
                await asyncio.sleep(FINAL_GRACE)
                if time.time() - call_state_dict.get("_last_speech_time", 0) >= FINAL_GRACE - 1:
                    logger.info("[SILENCE] No response after re-prompt — hanging up")
                    call_state_dict["state"] = CallState.ENDED
                    _hc = _telnyx_client or httpx.AsyncClient()
                    await _hc.post(f"/v2/calls/{call_control_id}/actions/hangup")
                    return
            except Exception as _e:
                logger.warning(f"[SILENCE] watchdog error: {_e}")
                return

        await asyncio.sleep(5.0)


def _cancel_speculative_fast(call_state_dict: dict) -> None:
    """Cancel in-flight speculative LLM and clear cached result (non-awaiting)."""
    t = call_state_dict.get("_speculative_llm_task")
    if t and not t.done():
        t.cancel()
    if call_state_dict.get("_speculative_active"):
        _log_turn_timing(call_state_dict, "speculative_cancelled")
    call_state_dict["_speculative_llm_task"] = None
    call_state_dict["_speculative_result"] = None
    call_state_dict["_speculative_active"] = False
    call_state_dict["_speculative_source_text"] = None
    call_state_dict["_speculative_started_at"] = None
    call_state_dict["_speculative_ready_at"] = None


def _clear_speculative_after_use(call_state_dict: dict) -> None:
    call_state_dict["_speculative_llm_task"] = None
    call_state_dict["_speculative_result"] = None
    call_state_dict["_speculative_active"] = False
    call_state_dict["_speculative_source_text"] = None
    call_state_dict["_speculative_started_at"] = None
    call_state_dict["_speculative_ready_at"] = None


async def run_full_llm_pipeline(
    transcript: str,
    call_control_id: str,
    patient_context: dict,
    call_state_dict: dict,
    history_for_llm: list,
    *,
    play_filler_audio: bool,
    pipeline_mode: str = "normal",
) -> Tuple[str, Optional[str], dict, str]:
    """
    Shared supervisor → (filler) → node path. Returns (speech, action, action_data, next_state_str).
    When play_filler_audio is False (speculative), filler LLMs still run for routing parity but no filler TTS.
    """
    from utils.config import (
        GOOGLE_API_KEY,
        GEMINI_MODEL,
        GROQ_API_KEY,
        GROQ_MODEL,
        LLM_BACKEND,
    )

    current_state = call_state_dict.get("state", CallState.GREETING)
    if LLM_BACKEND.lower() == "groq":
        api_key, model, backend = GROQ_API_KEY, GROQ_MODEL, "groq"
    else:
        api_key, model, backend = GOOGLE_API_KEY, GEMINI_MODEL, "gemini"

    logger.info(f"[LLM] Using {backend.upper()} ({model})")

    llm_start = time.perf_counter()
    _log_turn_timing(
        call_state_dict,
        "llm_pipeline_start",
        mode=pipeline_mode,
        transcript_chars=len(transcript),
    )

    _bypass_output = None
    if current_state == CallState.GREETING:
        _t = transcript.lower().strip().rstrip(".,!?")
        _CONFIRM = {
            "yes", "yeah", "yep", "yup", "sure", "i am", "this is",
            "speaking", "correct", "right", "that's me", "thats me",
            "it is", "yes i am", "yes this is", "hi yes", "hi yeah",
        }
        _DENY = {"no", "nope", "wrong number", "wrong person", "not me"}
        _is_confirm = _t in _CONFIRM or any(
            _t.startswith(w + " ") for w in _CONFIRM if len(w) > 3
        )
        _is_deny = _t in _DENY or any(
            _t.startswith(w) for w in _DENY if len(w) > 2
        )
        if _is_confirm:
            logger.info(
                f"[GREETING-BYPASS] Identity confirmed ('{transcript}') "
                f"→ availability_check (no supervisor LLM)"
            )
            _ag = build_agent_state(
                transcript,
                {**patient_context, "agent_name": AGENT_NAME},
                "availability_check",
                history_for_llm,
                api_key,
                model,
                backend,
                previous_node="greeting",
            )
            bypass_start = time.perf_counter()
            _bypass_output = await run_node_for_state("availability_check", _ag)
            _log_turn_timing(
                call_state_dict,
                "llm_stage_done",
                mode=pipeline_mode,
                stage="greeting_bypass_node",
                duration_ms=f"{(time.perf_counter() - bypass_start) * 1000:.0f}",
            )
        elif _is_deny:
            logger.info(f"[GREETING-BYPASS] Wrong person ('{transcript}') → ended")
            _bypass_output = (
                "I'm sorry, I must have the wrong number. Have a great day!",
                None,
                {},
                "ended",
            )

    if _bypass_output is not None:
        speech, action, action_data, next_state_str = _bypass_output
    else:
        stage1_start = time.perf_counter()
        node_result, filler_phrase = await asyncio.gather(
            decide_node(
                transcript,
                current_state.value,
                history_for_llm,
                api_key,
                model,
                backend,
            ),
            generate_smart_filler(
                transcript,
                current_state.value,
                current_state.value,
                api_key,
                model,
                backend,
            ),
        )
        _log_turn_timing(
            call_state_dict,
            "llm_stage_done",
            mode=pipeline_mode,
            stage="supervisor_plus_filler",
            duration_ms=f"{(time.perf_counter() - stage1_start) * 1000:.0f}",
            node=node_result,
        )
        logger.info(
            f"[PIPELINE] supervisor→{node_result}  filler='{filler_phrase}'"
        )

        agent_state = build_agent_state(
            transcript,
            {**patient_context, "agent_name": AGENT_NAME},
            current_state.value,
            history_for_llm,
            api_key,
            model,
            backend,
            previous_node=call_state_dict.get("previous_node", ""),
        )

        if play_filler_audio:
            filler_task = asyncio.create_task(
                _play_filler(call_control_id, filler_phrase)
            )
            node_task = asyncio.create_task(
                run_node_for_state(node_result, agent_state)
            )
            node_start = time.perf_counter()
            speech, action, action_data, next_state_str = await node_task
            _log_turn_timing(
                call_state_dict,
                "llm_stage_done",
                mode=pipeline_mode,
                stage="node_execution",
                duration_ms=f"{(time.perf_counter() - node_start) * 1000:.0f}",
                node=node_result,
            )
            llm_elapsed = time.perf_counter() - llm_start
            logger.info(f"[LATENCY] LLM full response: {llm_elapsed*1000:.0f}ms")
            try:
                await asyncio.wait_for(filler_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                filler_task.cancel()
        else:
            node_start = time.perf_counter()
            speech, action, action_data, next_state_str = await run_node_for_state(
                node_result, agent_state
            )
            _log_turn_timing(
                call_state_dict,
                "llm_stage_done",
                mode=pipeline_mode,
                stage="node_execution",
                duration_ms=f"{(time.perf_counter() - node_start) * 1000:.0f}",
                node=node_result,
            )
            llm_elapsed = time.perf_counter() - llm_start
            logger.info(
                f"[LATENCY] LLM full response (no filler audio): {llm_elapsed*1000:.0f}ms"
            )

    total_ms = (time.perf_counter() - llm_start) * 1000
    call_state_dict["_last_llm_done_at"] = time.perf_counter()
    call_state_dict["_last_llm_mode"] = pipeline_mode
    _log_turn_timing(
        call_state_dict,
        "llm_pipeline_done",
        mode=pipeline_mode,
        duration_ms=f"{total_ms:.0f}",
    )
    return (speech, action, action_data, next_state_str)

#-----------------------------------------------------------------
async def trigger_speculative_llm(
    call_state_dict: dict,
    call_control_id: str,
    patient_context: dict,
) -> None:
    if call_state_dict.get("_speculative_active"):
        return
    if call_state_dict.get("_speculative_result") is not None:
        return
    if call_state_dict.get("_processing"):
        return
    intro = call_state_dict.get("_intro_started_at", 0)
    if intro and (time.time() - intro) < STT_INTRO_GATE_SECS:
        return

    buf = call_state_dict.get("_transcript_buffer") or []
    combined = " ".join(buf).strip()
    if not combined:
        return

    call_state_dict["_speculative_source_text"] = combined
    call_state_dict["_speculative_active"] = True
    call_state_dict["_speculative_started_at"] = time.perf_counter()
    _log_turn_timing(
        call_state_dict,
        "speculative_start",
        transcript_chars=len(combined),
    )

    eff_history = list(call_state_dict.get("history", [])) + [
        {"role": "user", "content": combined}
    ]

    async def _spec_runner():
        try:
            r = await run_full_llm_pipeline(
                combined,
                call_control_id,
                patient_context,
                call_state_dict,
                eff_history,
                play_filler_audio=False,
                pipeline_mode="speculative",
            )
            call_state_dict["_speculative_result"] = r
            call_state_dict["_speculative_ready_at"] = time.perf_counter()
            _log_turn_timing(
                call_state_dict,
                "speculative_ready",
                duration_ms=f"{(call_state_dict['_speculative_ready_at'] - call_state_dict['_speculative_started_at']) * 1000:.0f}",
            )
            return r
        except asyncio.CancelledError:
            call_state_dict["_speculative_result"] = None
            raise
        finally:
            call_state_dict["_speculative_active"] = False

    call_state_dict["_speculative_llm_task"] = asyncio.create_task(_spec_runner())
    logger.debug(f"[SPEC] Started speculative LLM for '{combined[:80]}'")


async def _try_use_speculative_llm(
    transcript: str,
    call_state_dict: dict,
) -> Optional[Tuple[str, Optional[str], dict, str]]:
    src = (call_state_dict.get("_speculative_source_text") or "").strip()
    if src != transcript.strip():
        if (
            call_state_dict.get("_speculative_llm_task")
            or call_state_dict.get("_speculative_result") is not None
        ):
            _cancel_speculative_fast(call_state_dict)
        return None

    res = call_state_dict.get("_speculative_result")
    if res is not None:
        logger.info("[SPEC] Using completed speculative LLM result")
        turn_at = call_state_dict.get("_turn_signal_at", time.perf_counter())
        ready_at = call_state_dict.get("_speculative_ready_at", turn_at)
        _log_turn_timing(
            call_state_dict,
            "speculative_result_reuse",
            waited_after_turn_signal_ms=f"{max(0.0, turn_at - ready_at) * 1000:.0f}",
        )
        _clear_speculative_after_use(call_state_dict)
        return res

    t = call_state_dict.get("_speculative_llm_task")
    if t is not None:
        try:
            if not t.done():
                logger.info("[SPEC] Awaiting in-flight speculative LLM")
                wait_start = time.perf_counter()
                await t
                _log_turn_timing(
                    call_state_dict,
                    "speculative_wait_after_turn_signal",
                    wait_ms=f"{(time.perf_counter() - wait_start) * 1000:.0f}",
                )
            out = t.result()
        except asyncio.CancelledError:
            _cancel_speculative_fast(call_state_dict)
            return None
        except Exception as e:
            logger.warning(f"[SPEC] Speculative task failed: {e}")
            _cancel_speculative_fast(call_state_dict)
            return None
        logger.info("[SPEC] Using awaited speculative LLM result")
        _clear_speculative_after_use(call_state_dict)
        return out

    return None


# ============ Deepgram STT Handler ============
async def receive_from_deepgram(
    dg_ws, telnyx_ws: WebSocket, call_control_id: str, patient_context: dict, call_state_dict: dict
):
    """Listen to Deepgram for transcripts, drive LLM + TTS + tool actions"""
    try:
        _last_interim = ""        # track latest interim for UtteranceEnd fallback
        _last_confidence = 0.0
        _transcript_buffer: list = call_state_dict.setdefault("_transcript_buffer", [])
        _utterance_end_flush = False    # True when UtteranceEnd forces a hard flush

        async def _silence_monitor() -> None:
            try:
                while True:
                    await asyncio.sleep(0.1)
                    if call_state_dict.get("state") == CallState.ENDED:
                        return
                    if call_state_dict.get("_processing"):
                        continue
                    last_w = call_state_dict.get("_last_word_time", 0.0)
                    if time.time() - last_w < 0.5:
                        continue
                    await trigger_speculative_llm(
                        call_state_dict, call_control_id, patient_context
                    )
            except asyncio.CancelledError:
                raise

        _st = call_state_dict.get("_silence_timer_task")
        if _st is None or _st.done():
            call_state_dict["_silence_timer_task"] = asyncio.create_task(
                _silence_monitor()
            )

        async for message in dg_ws:
            response = json.loads(message)

            # UtteranceEnd fires when no speech detected for utterance_end_ms ms.
            # Hard-flush: skip EOU check and process whatever we have accumulated.
            if response.get("type") == "UtteranceEnd":
                if _transcript_buffer:
                    # Flush all buffered finals as one combined turn
                    combined = " ".join(_transcript_buffer).strip()
                    _transcript_buffer.clear()
                    logger.info(f"[STT] UtteranceEnd flushing buffer: '{combined}'")
                    response = {
                        "type": "Results",
                        "is_final": True,
                        "channel": {"alternatives": [{"transcript": combined, "confidence": _last_confidence}]},
                    }
                    _utterance_end_flush = True
                    # fall through to process
                elif _last_interim.strip():
                    # No buffered finals but we have a dangling interim — flush it
                    logger.info(f"[STT] UtteranceEnd flushing interim: '{_last_interim}'")
                    response = {
                        "type": "Results",
                        "is_final": True,
                        "channel": {"alternatives": [{"transcript": _last_interim, "confidence": _last_confidence}]},
                    }
                    _last_interim = ""
                    _utterance_end_flush = True
                    # fall through to process
                else:
                    continue

            if response.get("type") != "Results":
                continue

            is_final = response.get("is_final", False)
            alternatives = response.get("channel", {}).get("alternatives", [])
            if not alternatives:
                continue

            transcript = alternatives[0].get("transcript", "")
            confidence = alternatives[0].get("confidence", 0)

            if not transcript.strip():
                continue

            call_state_dict["_last_word_time"] = time.time()

            if not is_final:
                # Barge-in: if the person starts speaking while Sarah is playing audio,
                # stop the playback so the call feels like a real two-way conversation.
                if transcript.strip() and call_state_dict.get("_playing"):
                    if _RE_BACKCHANNEL.match(transcript.strip()):
                        logger.debug(f"[SMART-BARGE-IN] Backchannel during playback — ignoring: '{transcript}'")
                    else:
                        call_state_dict["_playing"] = False
                        call_state_dict["_barge_in_override"] = True
                        _last_interim = ""       # prevent UtteranceEnd flushing stale partial
                        _transcript_buffer.clear()  # discard any mid-turn buffer from old turn
                        _cancel_speculative_fast(call_state_dict)
                        logger.info(f"[BARGE-IN] Person speaking — stopping playback")
                        try:
                            _tc = _telnyx_client or httpx.AsyncClient()
                            await _tc.post(
                                f"/v2/calls/{call_control_id}/actions/playback_stop",
                            )
                        except Exception as _bge:
                            logger.debug(f"[BARGE-IN] playback_stop error: {_bge}")
                logger.debug(f"[Interim] {transcript} time: {time.time()}")#print with time stamp
                _last_interim = transcript          # track for UtteranceEnd fallback
                _last_confidence = confidence
                if call_state_dict.get("_speculative_active"):
                    src = (
                        call_state_dict.get("_speculative_source_text") or ""
                    ).strip()
                    it = transcript.strip()
                    if (
                        it
                        and src
                        and it != src
                        and not src.startswith(it)
                    ):
                        _cancel_speculative_fast(call_state_dict)
                # Reset silence timer — patient is actively speaking
                if transcript.strip():
                    call_state_dict["_last_speech_time"] = time.time()
                continue

            _last_interim = ""  # clear — we got a proper final
            logger.info(f"[FINAL] {transcript} (confidence: {confidence:.2f})")

            # Discard noise: single-word fragments and low-confidence short transcripts.
            # Exception: well-known single-word responses ("okay", "yes", "no", etc.)
            # are semantically complete and must be allowed through.
            word_count = len(transcript.strip().split())
            if word_count < 2:
                _bare = transcript.strip().lower().rstrip(".,!?")
                _SINGLE_WORD_OK = {
                    "yes", "yeah", "yep", "yup", "no", "nope",
                    "okay", "ok", "sure", "right", "alright",
                    "correct", "fine", "good", "great", "perfect",
                    "bye", "goodbye", "thanks", "nevermind",
                    "speaking",
                }
                if _bare not in _SINGLE_WORD_OK:
                    logger.warning(f"[STT] Discarding single-word fragment: '{transcript}'")
                    continue
                logger.info(f"[STT] Allowing known single-word response: '{transcript}'")
            if confidence < STT_MIN_CONFIDENCE and word_count <= STT_MIN_WORDS:
                logger.warning(f"[STT] Discarding low-confidence transcript ({confidence:.2f}, {word_count}w): '{transcript}'")
                continue

            # ── EOU (End-of-Utterance) turn detection ──────────────────────────
            # UtteranceEnd flush bypasses EOU — the silence timeout is itself
            # the hard-flush signal, so we process whatever is in the buffer.
            # Normal is_final fragments are buffered and only dispatched when
            # the EOU model judges the patient has finished their turn.
            if not _utterance_end_flush:
                if call_state_dict.get("_speculative_active"):
                    _cancel_speculative_fast(call_state_dict)
                _transcript_buffer.append(transcript)
                eou_complete, eou_prob = await _turn_detector.predict(_transcript_buffer)
                logger.debug(
                    f"[EOU] prob={eou_prob:.2f} complete={eou_complete} "
                    f"segs={len(_transcript_buffer)} '{' '.join(_transcript_buffer)[:60]}'"
                )
                if not eou_complete:
                    logger.info(
                        f"[EOU] Holding — waiting for more speech "
                        f"(prob={eou_prob:.2f}): '{transcript}'"
                    )
                    continue
                # Turn complete — combine all buffered segments into one transcript
                transcript = " ".join(_transcript_buffer).strip()
                _transcript_buffer.clear()
                logger.info(f"[EOU] Turn complete (prob={eou_prob:.2f}): '{transcript}'")
                turn_source = "eou"
            else:
                _utterance_end_flush = False   # reset flag; transcript already combined above
                turn_source = "utterance_end"

            call_state_dict["_turn_seq"] = int(call_state_dict.get("_turn_seq", 0)) + 1
            call_state_dict["_active_turn_id"] = call_state_dict["_turn_seq"]
            call_state_dict["_turn_signal_at"] = time.perf_counter()
            call_state_dict["_turn_signal_source"] = turn_source
            _log_turn_timing(
                call_state_dict,
                "turn_signal",
                source=turn_source,
                transcript_chars=len(transcript),
            )
            if call_state_dict.get("_speculative_started_at"):
                head_start_ms = (
                    call_state_dict["_turn_signal_at"] - call_state_dict["_speculative_started_at"]
                ) * 1000
                _log_turn_timing(
                    call_state_dict,
                    "speculative_headstart",
                    ms=f"{head_start_ms:.0f}",
                )

            # ── Debounce: skip if pipeline is already processing ────────────────
            # Exception: after a barge-in the new turn is allowed through so the
            # patient isn't silently ignored after interrupting the agent.
            if call_state_dict.get("_processing"):
                if not call_state_dict.get("_barge_in_override"):
                    logger.warning(f"[DEDUP] Skipping transcript (already processing): {transcript[:40]}")
                    _transcript_buffer.clear()  # don't carry stale buffer forward
                    _cancel_speculative_fast(call_state_dict)
                    continue
                else:
                    logger.info(f"[DEDUP] Allowing post-barge-in transcript: {transcript[:40]}")
                    call_state_dict["_barge_in_override"] = False
            call_state_dict["_processing"] = True

            # Intro gate: ignore STT finals fired while the intro TTS is streaming.
            # The caller may say "Hello?" before the greeting is done playing; that
            # produces a low-quality turn that confuses the state machine.
            intro_started = call_state_dict.get("_intro_started_at", 0)
            if intro_started and (time.time() - intro_started) < STT_INTRO_GATE_SECS:
                logger.info(f"[INTRO-GATE] Dropping transcript during intro window: '{transcript[:40]}'")
                call_state_dict["_processing"] = False
                continue

            call_log = call_state_dict.get("_logger")
            if call_log:
                call_log.transcript(transcript, confidence)

            # Update silence watchdog clock — any real speech resets the timer
            call_state_dict["_last_speech_time"] = time.time()

            current_state = call_state_dict.get("state", CallState.GREETING)
            if current_state == CallState.ENDED:
                call_state_dict["_processing"] = False
                continue

            history = call_state_dict.setdefault("history", [])
            history.append({"role": "user", "content": transcript})

            spec_tuple = await _try_use_speculative_llm(transcript, call_state_dict)
            if spec_tuple is not None:
                speech, action, action_data, next_state_str = spec_tuple
                _log_turn_timing(call_state_dict, "llm_result_reused_speculative")
            else:
                speech, action, action_data, next_state_str = await run_full_llm_pipeline(
                    transcript,
                    call_control_id,
                    patient_context,
                    call_state_dict,
                    history,
                    play_filler_audio=True,
                    pipeline_mode="normal",
                )

            llm_ready_at = time.perf_counter()
            turn_signal_at = call_state_dict.get("_turn_signal_at", llm_ready_at)
            _log_turn_timing(
                call_state_dict,
                "llm_response_ready_for_turn",
                delay_from_turn_signal_ms=f"{(llm_ready_at - turn_signal_at) * 1000:.0f}",
                llm_mode=call_state_dict.get("_last_llm_mode", "unknown"),
            )

            # Update history with just the node's speech (the filler is a neutral
            # short ack like "Sure one moment" — not meaningful conversational content)
            if speech:
                history.append({"role": "assistant", "content": speech})
            call_state_dict["history"] = history[-10:]

            if call_log:
                call_log.llm_response(speech, action, action_data, next_state_str)

            # Speak the response
            audio_dur = 0.0
            if speech:
                tts_start = time.perf_counter()
                call_state_dict["_playing"] = True
                _log_turn_timing(
                    call_state_dict,
                    "speech_dispatch_start",
                    delay_from_turn_signal_ms=f"{(tts_start - turn_signal_at) * 1000:.0f}",
                )
                _, audio_dur = await play_speech(call_control_id, speech)
                tts_elapsed = time.perf_counter() - tts_start
                logger.info(f"[LATENCY] TTS+playback: {tts_elapsed*1000:.0f}ms")
                _log_turn_timing(
                    call_state_dict,
                    "speech_dispatch_done",
                    tts_and_playback_ms=f"{tts_elapsed * 1000:.0f}",
                    est_audio_secs=f"{audio_dur:.2f}",
                )

            if audio_dur > 0:
                # Auto-clear _playing flag after estimated audio finishes
                async def _clear_playing(dur: float, state: dict):
                    await asyncio.sleep(dur + 0.5)
                    state["_playing"] = False
                asyncio.create_task(_clear_playing(audio_dur, call_state_dict))
                # Record when we started speaking so the cooldown can reference it
                call_state_dict["_speech_started_at"] = time.time()
                call_state_dict["_speech_dur"] = audio_dur

            # Execute tool action — saves outcome to PostgreSQL (+ Redis side-write).
            # confirm and cancel are one-shot: skip if already executed this call.
            _ONE_SHOT = {"confirm_appointment", "cancel_appointment"}
            _done_tools = call_state_dict.setdefault("_done_tools", set())
            if action and action in _ONE_SHOT and action in _done_tools:
                logger.warning(f"[TOOL] Skipping duplicate one-shot action: {action}")
                action = None
            if action:
                try:
                    await execute_tool_action(
                        action, action_data or {}, call_control_id,
                        patient_context, call_state_dict,
                        redis_client=db.client,
                        pg_db=pg_db,
                    )
                    _done_tools.add(action)
                except Exception as tool_err:
                    logger.warning(f"[TOOL] Exception executing '{action}': {tool_err}")

            # Update state
            try:
                new_state = CallState(next_state_str.lower().strip())

                # Enforce one-way state transitions — reject any LLM regression
                allowed = STATE_TRANSITIONS.get(current_state.value, {new_state.value})
                if new_state.value not in allowed:
                    logger.warning(
                        f"[STATE] Blocked regression {current_state.value} → {new_state.value} (LLM tried to go backwards)"
                    )
                    new_state = current_state  # keep where we are

                # Check if a tool (e.g. end_call) already set the state to ENDED
                tool_set_ended = call_state_dict.get("state") == CallState.ENDED

                if not tool_set_ended:
                    call_state_dict["previous_node"] = current_state.value
                    call_state_dict["state"] = new_state
                    if call_log:
                        call_log.state_change(current_state.value, new_state.value)
                    logger.info(f"[STATE] {current_state.value} → {new_state.value}")

                # Auto-hangup after final audio finishes — triggered whether state
                # was set here by the LLM or by a tool action (end_call).
                if call_state_dict.get("state") == CallState.ENDED:
                    remaining_audio = max(0.0, audio_dur)
                    wait_secs = remaining_audio + 1.5  # 1.5s network/jitter buffer
                    logger.info(f"[STATE] Waiting {wait_secs:.1f}s for final audio before hangup")
                    await asyncio.sleep(wait_secs)

                    # Save call_history if not already written by end_call tool
                    if not call_state_dict.get("_call_history_saved"):
                        call_state_dict["_call_history_saved"] = True
                        call_id   = call_state_dict.get("_call_id", call_control_id or "unknown")
                        appt_id   = patient_context.get("appointment_id", f"appt_{patient_context.get('phone','')}")
                        call_log_obj = call_state_dict.get("_logger")
                        import datetime as _dt
                        duration  = round((_dt.datetime.utcnow() - call_log_obj.start_time).total_seconds(), 1) if call_log_obj else 0.0
                        # Derive outcome from which tool fired during the call
                        if call_state_dict.get("_confirmed"):
                            outcome = "confirmed"
                        elif call_state_dict.get("_cancel_reason") is not None:
                            outcome = "cancelled"
                        elif call_state_dict.get("_preferred_reschedule_time") is not None:
                            outcome = "rescheduled"
                        else:
                            outcome = "hung_up"
                        await pg_db.save_call_history(
                            call_id=call_id,
                            appointment_id=appt_id,
                            patient_context=patient_context,
                            outcome=outcome,
                            duration_seconds=duration,
                            cancel_reason=call_state_dict.get("_cancel_reason", ""),
                            preferred_reschedule_time=call_state_dict.get("_preferred_reschedule_time", ""),
                        )
                        logger.info(f"[PG] Auto-saved call_history: outcome={outcome}")

                    try:
                        _hc = _telnyx_client or httpx.AsyncClient()
                        await _hc.post(
                            f"/v2/calls/{call_control_id}/actions/hangup",
                        )
                        logger.info("[STATE] Auto-hangup fired after reaching ENDED")
                    except Exception as hup_err:
                        logger.warning(f"[STATE] Auto-hangup failed: {hup_err}")
            except ValueError:
                logger.error(f"[LLM] Invalid next_state '{next_state_str}' — keeping current state")
            finally:
                # Hold the processing lock until after Sarah's audio has had a chance
                # to actually start playing. This prevents rapid double-utterances
                # ("yes yes", "okay okay") from firing twice and skipping a state.
                # Minimum 0.8 s; extends to cover the audio duration + 0.3 s buffer.
                _a_dur = call_state_dict.get("_speech_dur", 0.0)
                _cooldown = max(0.8, _a_dur + 0.3)

                async def _release_lock(delay: float, state: dict):
                    await asyncio.sleep(delay)
                    state["_processing"] = False

                asyncio.create_task(_release_lock(_cooldown, call_state_dict))

    except Exception as e:
        # 1011 = Deepgram timeout (no audio data received within keepalive window).
        # Keepalive task should prevent this, but log it clearly if it slips through.
        err_str = str(e)
        if "1011" in err_str:
            logger.warning(f"[Deepgram] Keepalive timeout (1011) — connection dropped. "
                           f"Keepalive task should have prevented this.")
        else:
            logger.error(f"[Deepgram] Error: {e}")
            logger.error(traceback.format_exc())


# ============ Outbound Calls ============
@app.post("/calls/outbound")
async def make_outbound_call(request: dict):
    """Initiate outbound call via Telnyx"""
    to_number = request.get("to")
    if not to_number:
        return {"error": "Missing 'to' field"}, 400

    patient_name = request.get("patient_name", "Patient")
    appointment_date = request.get("appointment_date", "your upcoming appointment")
    appointment_time = request.get("appointment_time", "")
    department = request.get("department", "General Medicine")
    clinic_name = request.get("clinic_name", "Talbot Health Service")
    provider_name = request.get("provider_name", "your doctor")
    appointment_datetime = f"{appointment_date} at {appointment_time}".strip()

    # Generate a stable appointment_id for this patient+call
    appointment_id = str(uuid.uuid4())

    # Store patient context in Redis (short-lived, for the duration of the call)
    if db.client:
        try:
            _patient_fields = {
                "name": patient_name,
                "phone": to_number,
                "appointment": appointment_datetime,
                "appointment_id": appointment_id,
                "department": department,
                "clinic_name": clinic_name,
                "provider_name": provider_name,
            }
            for _f, _v in _patient_fields.items():
                await db.client.hset(f"patient:{to_number}", _f, str(_v))
            await db.client.expire(f"patient:{to_number}", 3600)
        except Exception as e:
            logger.warning(f"[OUTBOUND] Redis store failed: {e}")

    # Upsert appointment row in PostgreSQL so tool saves have a row to UPDATE
    await pg_db.upsert_appointment(appointment_id, {
        "phone": to_number,
        "name": patient_name,
        "appointment": appointment_datetime,
        "clinic_name": clinic_name,
        "provider_name": provider_name,
    })

    # Dial via Telnyx
    stream_url = f"{WEBHOOK_BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/media-stream"
    webhook_url = f"{WEBHOOK_BASE_URL}/webhooks/calls"
    try:
        client = _telnyx_client or httpx.AsyncClient()
        resp = await client.post(
            "/v2/calls",
            json={
                "connection_id": TELNYX_CONNECTION_ID,
                "to": to_number,
                "from": TELNYX_ATC_NUMBER,
                "webhook_url": webhook_url,
                "stream_url": stream_url,
                "stream_track": "inbound_track",
            },
        )
        result = resp.json()
        logger.info(f"[OUTBOUND] Telnyx dial {to_number}: {resp.status_code}")
        asyncio.create_task(ensure_greeting_opening_wav_cached())

        return {
            "status": "dialing",
            "to": to_number,
            "call_control_id": result.get("data", {}).get("call_control_id"),
        }
    except Exception as e:
        logger.error(f"[OUTBOUND] Telnyx dial failed: {e}")
        return {"error": str(e)}


# ============ Webhook (HTTP) ============
@app.post("/webhooks/calls")
async def handle_webhook(request: Request):
    """Handle Telnyx webhook events"""
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        logger.warning(f"[WEBHOOK] Could not parse JSON body: {raw[:200]}")
        return {"status": "received"}

    # Telnyx sends events in two possible shapes:
    #   New:  {"event_type": "...", "payload": {...}, ...}         (top-level)
    #   Old:  {"data": {"event_type": "...", "payload": {...}}}    (nested)
    if "event_type" in body:
        event_type = body.get("event_type")
        payload    = body.get("payload", {})
    else:
        event_type = body.get("data", {}).get("event_type")
        payload    = body.get("data", {}).get("payload", {})

    call_control_id = payload.get("call_control_id")
    logger.info(f"[WEBHOOK] Incoming event: {event_type} — cid={call_control_id}")

    if event_type in ("call.initiated", "call_initiated") and payload.get("direction") == "incoming":
        # Answer the incoming call and start media streaming
        try:
            client = _telnyx_client or httpx.AsyncClient()
            # Step 1: Answer the call
            ans_resp = await client.post(
                f"/v2/calls/{call_control_id}/actions/answer",
                json={},
            )
            logger.info(f"[WEBHOOK] Answered incoming call {call_control_id}: {ans_resp.status_code}")

            # Step 2: Start media streaming to our WebSocket endpoint
            stream_url = f"{WEBHOOK_BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/media-stream"
            stream_resp = await client.post(
                f"/v2/calls/{call_control_id}/actions/streaming_start",
                json={
                    "stream_url": stream_url,
                    "stream_track": "inbound_track",
                },
            )
            logger.info(f"[WEBHOOK] Streaming started {call_control_id}: {stream_resp.status_code} — {stream_url}")
        except Exception as e:
            logger.error(f"[WEBHOOK] Failed to answer/stream incoming call: {e}")

    elif event_type in ("call.answered", "call_answered") and payload.get("direction") == "outbound":
        # Outbound call was answered — start media streaming
        try:
            client = _telnyx_client or httpx.AsyncClient()
            stream_url = f"{WEBHOOK_BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/media-stream"
            stream_resp = await client.post(
                f"/v2/calls/{call_control_id}/actions/streaming_start",
                json={
                    "stream_url": stream_url,
                    "stream_track": "inbound_track",
                },
            )
            logger.info(f"[WEBHOOK] Outbound call answered, streaming started {call_control_id}: {stream_resp.status_code} — {stream_url}")
        except Exception as e:
            logger.error(f"[WEBHOOK] Failed to start outbound streaming: {e}")

    return {"status": "received"}


# ============ Media Stream (WebSocket) ============
@app.websocket("/ws/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle incoming call from Telnyx via WebSocket"""
    await websocket.accept()
    logger.info("📞 Telnyx media stream connected")

    dg_ws = None
    dg_receiver_task = None
    dg_keepalive_task = None
    silence_watchdog_task = None
    call_control_id = None
    patient_context = None
    call_state = {
        "state": CallState.GREETING,
        "history": [],
        "_last_speech_time": time.time(),
        "_transcript_buffer": [],
        "_speculative_llm_task": None,
        "_speculative_result": None,
        "_speculative_active": False,
        "_speculative_started_at": None,
        "_speculative_ready_at": None,
        "_last_word_time": time.time(),
        "_silence_timer_task": None,
        "_turn_seq": 0,
        "_active_turn_id": None,
        "_turn_signal_at": None,
    }

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception as recv_err:
                logger.error(f"[STREAM] receive failed: {recv_err}")
                break

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"[STREAM] Could not parse JSON: {str(raw)[:100]}")
                continue

            if not isinstance(data, dict):
                logger.warning(f"[STREAM] Non-dict: {str(data)[:100]}")
                continue

            event = data.get("event")

            if event == "start":
                # Extract call metadata
                start_data = data.get("start", {})
                call_control_id = start_data.get("call_control_id")
                calling_number = start_data.get("from")
                called_number = start_data.get("to")

                if not call_control_id:
                    logger.error("[STREAM] start event missing call_control_id")
                    continue

                logger.info(
                    f"[{call_control_id[:12]}...] Call started {calling_number} → {called_number}"
                )
                call_state["_logger"] = CallLogger(
                    call_control_id, calling_number or "unknown", called_number or "unknown"
                )
                call_state["_call_id"] = call_control_id

                # Redis lookup in parallel with first greeting audio: opening line plays immediately
                # (prefetched TTS), then personalized suffix once hgetall returns.
                redis_task = asyncio.create_task(
                    load_patient_context_for_stream(called_number, calling_number)
                )
                asyncio.create_task(
                    _play_split_greeting_sequence(call_control_id, redis_task)
                )

                patient_context = await redis_task

                patient_name = patient_context.get("name", "there")
                clinic_name = patient_context.get("clinic_name", "the clinic")
                suffix = (
                    f"I'm calling from {clinic_name}. Am I speaking with {patient_name}?"
                )
                greeting_text = f"{GREETING_OPENING_TEXT} {suffix}"
                logger.info(f"[GREETING] Split intro queued for {patient_name}")
                # Gate STT — patient may say "Hello?" while intro is still playing.
                call_state["_intro_started_at"] = time.time()
                call_state.setdefault("history", []).append(
                    {"role": "assistant", "content": greeting_text}
                )

                try:
                    auth_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                    dg_ws = await websockets_client.connect(DEEPGRAM_URL, additional_headers=auth_headers)
                    logger.info(f"   ✓ Connected to Deepgram (nova-3)")

                    # Start receiving transcripts
                    dg_receiver_task = asyncio.create_task(
                        receive_from_deepgram(dg_ws, websocket, call_control_id, patient_context, call_state)
                    )

                    # Keepalive — Deepgram closes with 1011 if no data for ~12s.
                    # Send KeepAlive every 8s during silences.
                    async def _dg_keepalive(ws):
                        try:
                            while True:
                                await asyncio.sleep(8)
                                await ws.send(json.dumps({"type": "KeepAlive"}))
                        except Exception:
                            pass
                    dg_keepalive_task = asyncio.create_task(_dg_keepalive(dg_ws))

                    # Start silence watchdog — like Vapi's silenceTimeoutSeconds
                    call_state["_last_speech_time"] = time.time()
                    silence_watchdog_task = asyncio.create_task(
                        _silence_watchdog(call_state, call_control_id)
                    )

                except Exception as e:
                    logger.error(f"  ✘ Deepgram connection failed: {e}")
                    await websocket.send_json({"error": "Deepgram failed"})
                    break

            elif event == "media":
                # Forward audio to Deepgram
                payload = data.get("media", {}).get("payload")
                if dg_ws and payload:
                    try:
                        audio_bytes = base64.b64decode(payload)
                        await dg_ws.send(audio_bytes)
                    except Exception as e:
                        logger.error(f"  ⚠️  Error forwarding audio: {e}")

            elif event == "stop":
                logger.info(f"  🛑 Telnyx stop event")
                break

    except WebSocketDisconnect:
        logger.info(f"  ❌ Telnyx disconnected")
    except Exception as e:
        logger.error(f"  ⚠️  Stream error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        if dg_ws:
            await dg_ws.close()
        if dg_receiver_task:
            dg_receiver_task.cancel()
        if dg_keepalive_task:
            dg_keepalive_task.cancel()
        if silence_watchdog_task:
            silence_watchdog_task.cancel()
        _st_cancel = call_state.get("_silence_timer_task")
        if _st_cancel:
            _st_cancel.cancel()
            try:
                await _st_cancel
            except asyncio.CancelledError:
                pass
        _cancel_speculative_fast(call_state)

        call_log = call_state.get("_logger")
        if call_log:
            state = call_state.get("state", "unknown")
            call_log.call_end(state.value if hasattr(state, "value") else str(state))

        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as _ws_err:
            logger.debug(f"[WS] close error (already closed): {_ws_err}")

        logger.info(f"  ✅ Stream finalized")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
