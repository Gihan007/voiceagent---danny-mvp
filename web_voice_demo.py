import asyncio
import base64
import json
import logging
import re
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response as FastAPIResponse, StreamingResponse
from pydantic import BaseModel, Field

from utils.config import (
    AGENT_NAME,
    CARTESIA_API_KEY,
    CARTESIA_VOICE_ID,
    DEEPGRAM_API_KEY,
    DEEPGRAM_TTS_VOICE,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_BACKEND,
)
from utils.helpers import clean_text_for_tts
from utils.rag_service import HospitalRAGService
from utils.supervisor import (
    build_agent_state,
    decide_node,
    run_node_for_state,
    stream_node_speech_for_state,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_voice_demo")

app = FastAPI(title="Voice Agent Web Demo")
HOSPITAL_RAG = HospitalRAGService()

DEFAULT_PATIENT_CONTEXT = {
    "name": "John",
    "phone": "+15550100001",
    "appointment": "Tuesday at 3 PM",
    "clinic_name": "City Clinic",
    "provider_name": "Dr. Smith",
}

DEMO_TTS_SAMPLE_RATE = 24000
DEMO_STT_SAMPLE_RATE = 16000
DEMO_CARTESIA_VOICE_ID = CARTESIA_VOICE_ID or "f786b574-daa5-4673-aa0c-cbe3e8534c02"
DEMO_DEEPGRAM_TTS_VOICE = (
    DEEPGRAM_TTS_VOICE
    if (DEEPGRAM_TTS_VOICE or "").startswith("aura-2-")
    else "aura-2-luna-en"
)
DEMO_CARTESIA_FIRST_CHUNK_TIMEOUT = 2.5
CARTESIA_FAILURES = 0
CARTESIA_DISABLED_UNTIL = 0.0
CARTESIA_FAILURE_LIMIT = 2
CARTESIA_COOLDOWN_SECONDS = 60

GREETING_TEXT = (
    f"Hi, this is Jamie from {DEFAULT_PATIENT_CONTEXT['clinic_name']}. "
    f"Am I speaking with {DEFAULT_PATIENT_CONTEXT['name']}?"
)
FAST_CONFIRM_TEXT = "Great! Do you have a quick minute to chat right now?"
FAST_APPOINTMENT_TEXT = (
    f"I'm calling about your appointment on {DEFAULT_PATIENT_CONTEXT['appointment']} "
    f"with {DEFAULT_PATIENT_CONTEXT['provider_name']}. Does that still work for you?"
)
FAST_CALLBACK_TEXT = (
    "No problem at all. Please stay safe, and we'll try you again at a better time. Take care!"
)
FAST_HEAR_YOU_TEXT = "Yes, I can hear you. Am I speaking with John?"
FAST_RECEPTIONIST_TEXT = "Yes, this is Jamie calling from City Clinic about your appointment."
FAST_VERIFYING_TEXT = "I'm just verifying the appointment details before we continue."
GROUNDED_PROVIDER_TEXT = f"Sure. Your appointment is with {DEFAULT_PATIENT_CONTEXT['provider_name']}."
GROUNDED_PROVIDER_BOUNDARY_TEXT = (
    f"Sure. Your appointment is with {DEFAULT_PATIENT_CONTEXT['provider_name']}. "
    "I only have the appointment details in this demo, so the clinic team can help with "
    "specialty, biography, or clinical background questions."
)
FAST_GOODBYE_TEXT = f"Thanks so much, {DEFAULT_PATIENT_CONTEXT['name']}. Have a wonderful day!"
COMMON_DEMO_SPEECH = [
    GREETING_TEXT,
    FAST_CONFIRM_TEXT,
    FAST_APPOINTMENT_TEXT,
    FAST_CALLBACK_TEXT,
    FAST_HEAR_YOU_TEXT,
    FAST_RECEPTIONIST_TEXT,
    FAST_VERIFYING_TEXT,
    GROUNDED_PROVIDER_TEXT,
    GROUNDED_PROVIDER_BOUNDARY_TEXT,
    FAST_GOODBYE_TEXT,
]

AUDIO_CACHE_TTL = 90
STREAM_AUDIO_CACHE: Dict[str, Any] = {}
BUFFER_AUDIO_CACHE: Dict[str, bytes] = {}


class SpeakRequest(BaseModel):
    text: str


class TranscribeRequest(BaseModel):
    audio_base64: str
    content_type: str = "audio/webm"


class VoiceTurnRequest(BaseModel):
    transcript: str
    current_state: str = "greeting"
    history: List[Dict[str, str]] = Field(default_factory=list)
    patient_context: Dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_PATIENT_CONTEXT)
    )


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
        for phrase in phrases
    )


def _looks_affirmative(text: str) -> bool:
    affirmative_terms = (
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "go ahead",
        "yes do", "yes i do", "yeah do", "yeah i do",
        "i do", "i can", "i have", "i have time", "i have a minute",
        "of course", "correct", "right", "that's me", "thats me",
        "this is", "speaking", "it is", "you do", "you have", "this you do",
    )
    if text in affirmative_terms:
        return True
    if re.match(r"^(yes|yeah|yep|yup|sure|okay|ok)\b", text):
        return True
    if re.match(r"^(yes|yeah|yep|yup)\s+(do|i do|you do|speaking|that's me|thats me)\b", text):
        return True
    if _contains_any(text, (" yes", " yeah", " yep", " yup", " sure", " of course")):
        return True
    if _contains_any(text, ("you do", "you have", "this you", "i do", "i have")):
        return True
    return False


def _looks_negative_or_unavailable(text: str) -> bool:
    terms = (
        "no", "nope", "nah", "not really", "i can't", "i cant", "i cannot",
        "driving", "drive", "busy", "bad time", "can't talk", "cannot talk",
        "cant talk", "not now", "not right now", "at work", "working",
        "in a meeting", "call me later", "later", "no time",
    )
    return text in terms or _contains_phrase(text, terms)


def _looks_like_goodbye(text: str) -> bool:
    if re.search(r"\b(bye|goodbye|see you|talk to you later|nothing else|that's all|thats all|all good|i'm good|im good)\b", text):
        return True
    return text in {"thanks", "thank you", "okay thanks", "ok thanks", "perfect thanks"}


def _asks_can_hear(text: str) -> bool:
    return _contains_any(text, ("can you hear me", "do you hear me", "hear me", "hello can you hear"))


def _asks_agent_identity(text: str) -> bool:
    return _contains_any(
        text,
        (
            "receptionist", "receptionist guy", "clinic assistant",
            "are you jamie", "who are you", "you are that",
        ),
    )


def _asks_why_verifying(text: str) -> bool:
    return _contains_any(
        text,
        (
            "why didn't you", "why didnt you", "why do you want",
            "why are you asking", "why you asking", "why verify",
            "why verifying", "want to know about that",
        ),
    )


def _looks_like_cancel_request(text: str) -> bool:
    return bool(re.search(r"\b(cancel|cancel it|cancel appointment|just cancel|cancel for now|can't do this|cant do this|need to cancel)\b", text))


def _looks_like_final_cancel(text: str) -> bool:
    return _looks_like_cancel_request(text) and _contains_any(text, ("just", "for now", "no need", "don't reschedule", "dont reschedule"))


def _looks_dissatisfied_or_profane(text: str) -> bool:
    return _contains_phrase(
        text,
        (
            "shitty", "shit", "bad service", "terrible", "annoying",
            "find another hospital", "go another hospital", "not happy",
        ),
    )


def _appointment_does_not_work(text: str) -> bool:
    terms = (
        "it won't", "it wont", "doesn't work", "doesnt work", "not work",
        "can't make", "cant make", "cannot make", "can't on that day",
        "cant on that day", "work to do", "busy at that time",
        "some work", "got work", "have work", "not at that time",
        "not that day", "difficult for me", "can't be there", "cant be there",
        "cannot be there",
    )
    return _looks_negative_or_unavailable(text) or _contains_any(text, terms)


def _appointment_confirmed(text: str) -> bool:
    if _looks_affirmative(text):
        return True
    return bool(
        re.search(
            r"^(yes|yeah|yep|yup|sure|okay|ok|perfect|great|correct|right)\b",
            text,
        )
    ) or _contains_any(text, ("that works", "it works", "still works", "looks good"))


def _looks_like_reschedule_detail(text: str) -> bool:
    return bool(
        re.search(
            r"\b(reschedule|change|move|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekdays|weekend|weekends|morning|afternoon|evening|after|before|\d+\s*(am|pm)|12pm|4pm)\b",
            text,
        )
    )


def _has_new_schedule_preference(text: str) -> bool:
    return bool(
        re.search(
            r"\b(reschedule|change|move|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekdays|weekend|weekends|morning|afternoon|evening|after|before|after\s+\d+\s*(am|pm)|before\s+\d+\s*(am|pm))\b",
            text,
        )
    )


def _reschedule_fast_response(
    text: str,
    patient_context: Dict[str, Any],
) -> Optional[tuple[str, Optional[str], dict, str, str]]:
    provider = patient_context.get("provider_name", DEFAULT_PATIENT_CONTEXT["provider_name"])

    if re.search(r"\b(hello|hello\?|hi|are you there)\b", text):
        return (
            "I'm here. We were looking at rescheduling; would morning or afternoon work better for you?",
            None,
            {},
            "reschedule",
            "reschedule_fast_resume",
        )

    if _contains_any(text, ("quick", "be quick", "hurry", "fast")):
        return (
            "I'll keep it brief. What day and time should I note as your preference?",
            None,
            {},
            "reschedule",
            "reschedule_fast_quick",
        )

    if _contains_any(text, ("what kind of day", "kind of day", "what day")):
        return (
            "We can note a weekday like Monday, or another day that works for you. Would morning or afternoon be better?",
            None,
            {},
            "reschedule",
            "reschedule_fast_day_options",
        )

    if "monday" in text:
        return (
            f"I can note Monday as your preference with {provider}. Would morning or afternoon work better?",
            "reschedule_request",
            {"preferred_day": "Monday"},
            "reschedule",
            "reschedule_fast_monday",
        )

    if "weekday" in text or "weekdays" in text:
        return (
            f"I can note a weekday appointment preference with {provider}. Would Monday, Wednesday, or Friday after 4 PM work as a backup preference?",
            "reschedule_request",
            {"preferred_day": "weekday"},
            "reschedule",
            "reschedule_fast_weekday",
        )

    if "weekend" in text or "weekends" in text:
        return (
            f"I can note a weekend appointment preference with {provider}. If that is not available, is there a weekday backup that works for you?",
            "reschedule_request",
            {"preferred_day": "weekend"},
            "reschedule",
            "reschedule_fast_weekend",
        )

    if _looks_like_reschedule_detail(text):
        return (
            f"I can note that preference for {provider}. Is there another backup time that would also work for you?",
            "reschedule_request",
            {},
            "reschedule",
            "reschedule_fast_preference",
        )

    return None


def _is_rag_grounded_node(node: str) -> bool:
    return node.startswith("rag_")


def _humanize_reply(transcript: str, node: str, speech: str, current_state: str = "") -> str:
    """Add small, controlled spoken backchannels without making the agent ramble."""
    clean_speech = speech.strip()
    if not clean_speech:
        return clean_speech

    text = _normalize_text(transcript)
    first_words = clean_speech.lower().split()[:4]
    first = " ".join(first_words)

    if re.match(r"^(okay|ok|sure|yes|no|right|got it|mhm|ah|oh|perfect|great|of course|i understand|no worries)\b", first):
        return clean_speech

    prefix = ""
    emotional_terms = (
        "personal issue", "personal issues", "emergency", "sick", "ill",
        "accident", "problem", "difficult", "hard for me",
        "can't make it", "cant make it",
    )
    confused_terms = ("what happened", "what happend", "what is this", "i don't understand", "i dont understand")

    has_emotional_term = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in emotional_terms
    )

    if _contains_any(text, confused_terms):
        prefix = "Oh, sorry about that."
    elif has_emotional_term:
        prefix = "Oh, I'm sorry to hear that."
    elif node == "reschedule":
        prefix = "Okay, let me check that."
    elif node == "cancel":
        prefix = "I understand."
    elif node.startswith("rag_") or node in {"general_question", "grounded_provider", "grounded_provider_boundary", "grounded_appointment", "grounded_clinic"}:
        prefix = "Sure."
    elif node in {"confirm", "wrap_up"} and _looks_affirmative(text):
        prefix = "Perfect."

    if not prefix:
        return clean_speech
    return f"{prefix} {clean_speech}"


def _grounded_demo_response(
    text: str,
    patient_context: Dict[str, Any],
    current_state: str,
) -> Optional[tuple[str, Optional[str], dict, str, str]]:
    """Answer only facts this local demo actually knows."""
    appointment = patient_context.get("appointment", DEFAULT_PATIENT_CONTEXT["appointment"])
    provider = patient_context.get("provider_name", DEFAULT_PATIENT_CONTEXT["provider_name"])
    clinic = patient_context.get("clinic_name", DEFAULT_PATIENT_CONTEXT["clinic_name"])

    rag_answer = HOSPITAL_RAG.answer(text, patient_context)
    if rag_answer:
        return (
            rag_answer.speech,
            None,
            rag_answer.metadata,
            current_state,
            rag_answer.node,
        )

    asks_provider = _contains_any(text, ("doctor", "provider", "physician", "dr smith", "dr. smith"))
    asks_provider_reference = asks_provider or _contains_any(text, ("him", "his", "that doctor"))
    asks_identity = _contains_any(text, ("who is", "who's", "which doctor", "what doctor", "doctor name"))
    asks_background = _contains_any(
        text,
        (
            "personal", "background", "experience", "qualification", "graduate",
            "school", "reviews", "specialty", "speciality", "specialization",
            "specializations", "specialist", "specialists", "specialized",
            "specialised", "details", "detail", "more about", "know more",
            "full name", "first name", "last name",
        ),
    )
    asks_appointment = _contains_any(text, ("when", "what time", "which day", "appointment", "schedule"))
    asks_clinic = _contains_any(text, ("clinic", "where", "location", "address"))

    if asks_provider_reference and asks_background:
        return (
            f"Your appointment is with {provider}. I only have the appointment details in this demo, so the clinic team can help with specialty, biography, or clinical background questions.",
            None,
            {},
            current_state,
            "grounded_provider_boundary",
        )
    if asks_provider or asks_identity:
        return (
            f"Your appointment is with {provider}.",
            None,
            {},
            current_state,
            "grounded_provider",
        )
    if asks_appointment and not _looks_like_reschedule_detail(text):
        return (
            f"Your appointment is scheduled for {appointment} with {provider}.",
            None,
            {},
            current_state,
            "grounded_appointment",
        )
    if asks_clinic:
        return (
            f"This call is from {clinic}. I do not have verified address or parking details in this local demo.",
            None,
            {},
            current_state,
            "grounded_clinic",
        )
    return None


def _local_turn_response(
    transcript: str,
    patient_context: Dict[str, Any],
    current_state: str,
) -> Optional[tuple[str, Optional[str], dict, str, str]]:
    """Zero-LLM fast paths for common call turns and grounded facts."""
    state = current_state or "greeting"
    text = _normalize_text(transcript)
    patient_name = patient_context.get("name", DEFAULT_PATIENT_CONTEXT["name"])

    if _asks_can_hear(text):
        return FAST_HEAR_YOU_TEXT, None, {}, "greeting", "repair_fast_hear_you"

    if _asks_agent_identity(text):
        return FAST_RECEPTIONIST_TEXT, None, {}, state, "repair_fast_identity"

    if _asks_why_verifying(text):
        return FAST_VERIFYING_TEXT, None, {}, state, "repair_fast_verifying"

    if _looks_like_goodbye(text):
        return (
            f"Thanks so much, {patient_name}. Have a wonderful day!",
            None,
            {},
            "ended",
            "wrap_up_fast_goodbye",
        )

    if _looks_dissatisfied_or_profane(text):
        return (
            "I'm sorry about that. I'll make sure the appointment cancellation is noted. Take care.",
            None,
            {},
            "ended",
            "wrap_up_fast_dissatisfied",
        )

    if state == "greeting":
        if _looks_affirmative(text):
            return FAST_CONFIRM_TEXT, None, {}, "availability_check", "greeting_fast_confirm"
        if _looks_negative_or_unavailable(text) or _contains_any(text, ("wrong number", "wrong person", "not me")):
            return (
                "I'm sorry, I must have the wrong number. Have a great day!",
                None,
                {},
                "ended",
                "greeting_bypass",
            )

    if state == "availability_check":
        if _looks_negative_or_unavailable(text):
            return FAST_CALLBACK_TEXT, None, {}, "ended", "availability_fast_callback"
        if _looks_affirmative(text):
            appointment = patient_context.get("appointment", "your appointment")
            provider = patient_context.get("provider_name", "your doctor")
            speech = FAST_APPOINTMENT_TEXT
            if appointment != DEFAULT_PATIENT_CONTEXT["appointment"] or provider != DEFAULT_PATIENT_CONTEXT["provider_name"]:
                speech = f"I'm calling about your appointment on {appointment} with {provider}. Does that still work for you?"
            return speech, None, {}, "appointment_review", "availability_fast_yes"

    if state == "appointment_review":
        provider = patient_context.get("provider_name", DEFAULT_PATIENT_CONTEXT["provider_name"])
        appointment = patient_context.get("appointment", DEFAULT_PATIENT_CONTEXT["appointment"])
        if _looks_like_cancel_request(text):
            return (
                f"I understand. I can cancel your {appointment} appointment with {provider}.",
                "cancel",
                {},
                "cancel",
                "cancel_fast_request",
            )
        if _appointment_does_not_work(text) or _contains_any(text, ("reschedule", "change it", "move it")):
            if _has_new_schedule_preference(text):
                reschedule_response = _reschedule_fast_response(text, patient_context)
                if reschedule_response:
                    speech, action, data, next_state, node = reschedule_response
                    return f"No problem. {speech}", action, data, next_state, node
            return (
                "No problem. What day or time works better?",
                None,
                {},
                "reschedule",
                "appointment_review_fast_reschedule",
            )
        if _appointment_confirmed(text):
            return (
                f"Everything looks all set then. We'll see you {appointment}.",
                "confirm",
                {},
                "confirm",
                "appointment_review_fast_confirm",
            )

    if state == "cancel":
        appointment = patient_context.get("appointment", DEFAULT_PATIENT_CONTEXT["appointment"])
        provider = patient_context.get("provider_name", DEFAULT_PATIENT_CONTEXT["provider_name"])
        if _looks_like_final_cancel(text) or _contains_any(text, ("no just cancel", "just cancel", "no need")):
            return (
                f"Understood. I've cancelled your {appointment} appointment with {provider}. Take care.",
                "cancel",
                {},
                "ended",
                "cancel_fast_final",
            )
        if _looks_like_cancel_request(text):
            return (
                f"I understand. I'll keep the {appointment} appointment with {provider} cancelled.",
                "cancel",
                {},
                "cancel",
                "cancel_fast_repeat",
            )

    if state in {"confirm", "wrap_up"} and _appointment_does_not_work(text):
        return (
            "No problem. What day or time works better?",
            None,
            {},
            "reschedule",
            "confirm_fast_reschedule",
        )

    if state in {"reschedule", "confirm", "wrap_up"} and _looks_like_cancel_request(text):
        appointment = patient_context.get("appointment", DEFAULT_PATIENT_CONTEXT["appointment"])
        provider = patient_context.get("provider_name", DEFAULT_PATIENT_CONTEXT["provider_name"])
        return (
            f"Understood. I've cancelled your {appointment} appointment with {provider}. Take care.",
            "cancel",
            {},
            "ended",
            "cancel_fast_from_active_flow",
        )

    if state == "reschedule":
        grounded = _grounded_demo_response(text, patient_context, state)
        if grounded and _is_rag_grounded_node(grounded[4]):
            return grounded

        reschedule_response = _reschedule_fast_response(text, patient_context)
        if reschedule_response:
            return reschedule_response

    if state != "reschedule" and _contains_any(text, ("let's reschedule", "lets reschedule", "reschedule")):
        reschedule_response = _reschedule_fast_response(text, patient_context)
        if reschedule_response:
            return reschedule_response

    grounded = _grounded_demo_response(text, patient_context, state)
    if grounded:
        return grounded

    return None


def _make_streaming_wav_header(sample_rate: int = DEMO_TTS_SAMPLE_RATE) -> bytes:
    byte_rate = sample_rate * 2
    block_align = 2
    header = struct.pack("<4sI4s", b"RIFF", 0xFFFFFFFF, b"WAVE")
    header += struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
    header += struct.pack("<4sI", b"data", 0xFFFFFFFF)
    return header


async def _cartesia_tts_pcm_stream(text: str):
    clean_text = clean_text_for_tts(text)
    if not clean_text or not _cartesia_available():
        return

    import websockets

    context_id = str(uuid.uuid4())
    try:
        async with websockets.connect(
            "wss://api.cartesia.ai/tts/websocket",
            additional_headers={
                "X-API-Key": CARTESIA_API_KEY,
                "Cartesia-Version": "2024-06-10",
            },
            open_timeout=15,
        ) as ws:
            await ws.send(json.dumps({
                "model_id": "sonic-3",
                "transcript": clean_text,
                "voice": {"mode": "id", "id": DEMO_CARTESIA_VOICE_ID},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": DEMO_TTS_SAMPLE_RATE,
                },
                "context_id": context_id,
                "__experimental_controls": {
                    "speed": "normal",
                    "emotion": ["positivity:high", "curiosity:low"],
                },
            }))
            async for message in ws:
                event = json.loads(message)
                if event.get("type") == "chunk":
                    yield base64.b64decode(event["data"])
                elif event.get("type") == "done":
                    break
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "Cartesia TTS error"))
    except Exception as exc:
        _record_cartesia_failure(str(exc))
        logger.warning("[TTS] Cartesia demo stream failed: %s", exc)


class PersistentCartesiaTTS:
    """One Cartesia WebSocket reused for a single live call."""

    def __init__(self) -> None:
        self.ws = None
        self.lock = asyncio.Lock()
        self.connect_lock = asyncio.Lock()

    async def _connect(self):
        if self.ws is not None:
            return self.ws

        async with self.connect_lock:
            if self.ws is not None:
                return self.ws

            import websockets

            self.ws = await websockets.connect(
                "wss://api.cartesia.ai/tts/websocket",
                additional_headers={
                    "X-API-Key": CARTESIA_API_KEY,
                    "Cartesia-Version": "2024-06-10",
                },
                open_timeout=15,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("[TTS] opened persistent Cartesia WebSocket")
            return self.ws

    async def close(self) -> None:
        ws = self.ws
        self.ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def stream_pcm(self, text: str):
        clean_text = clean_text_for_tts(text)
        if not clean_text or not _cartesia_available():
            return

        async with self.lock:
            for attempt in range(2):
                context_id = str(uuid.uuid4())
                try:
                    ws = await self._connect()
                    await ws.send(json.dumps({
                        "model_id": "sonic-3",
                        "transcript": clean_text,
                        "voice": {"mode": "id", "id": DEMO_CARTESIA_VOICE_ID},
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": DEMO_TTS_SAMPLE_RATE,
                        },
                        "context_id": context_id,
                        "__experimental_controls": {
                            "speed": "normal",
                            "emotion": ["positivity:high", "curiosity:low"],
                        },
                    }))

                    async for message in ws:
                        event = json.loads(message)
                        event_context = event.get("context_id")
                        if event_context and event_context != context_id:
                            continue
                        if event.get("type") == "chunk":
                            _record_cartesia_success()
                            yield base64.b64decode(event["data"])
                        elif event.get("type") == "done":
                            return
                        elif event.get("type") == "error":
                            raise RuntimeError(event.get("message", "Cartesia TTS error"))
                    return
                except Exception as exc:
                    _record_cartesia_failure(str(exc))
                    logger.warning("[TTS] persistent Cartesia stream failed: %s", exc)
                    await self.close()
                    if attempt == 1:
                        return


async def _cartesia_tts_stream(text: str):
    yielded_header = False
    async for chunk in _cartesia_tts_pcm_stream(text):
        if not yielded_header:
            yielded_header = True
            yield _make_streaming_wav_header()
        yield chunk


async def _deepgram_tts_stream(text: str):
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return
    async with httpx.AsyncClient(timeout=20.0) as client:
        async with client.stream(
            "POST",
            (
                "https://api.deepgram.com/v1/speak"
                f"?model={DEMO_DEEPGRAM_TTS_VOICE}"
                f"&encoding=linear16&sample_rate={DEMO_TTS_SAMPLE_RATE}&container=wav"
            ),
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": clean_text},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk


async def _deepgram_tts_pcm_stream(text: str):
    skipped_header = False
    async for chunk in _deepgram_tts_stream(text):
        if not chunk:
            continue
        if not skipped_header:
            skipped_header = True
            if len(chunk) <= 44:
                continue
            yield chunk[44:]
        else:
            yield chunk


async def _tts_stream(text: str):
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return

    if _cartesia_available():
        yielded = False
        async for chunk in _cartesia_tts_stream(clean_text):
            yielded = True
            yield chunk
        if yielded:
            _record_cartesia_success()
            return

    async for chunk in _deepgram_tts_stream(clean_text):
        yield chunk


async def _tts_pcm_stream_for_browser(text: str, cartesia_session: Optional[Any] = None):
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return

    key = _cache_key(clean_text)
    cached_wav = BUFFER_AUDIO_CACHE.get(key)
    if cached_wav and len(cached_wav) > 44:
        yield cached_wav[44:]
        return

    chunks: list[bytes] = []
    if _cartesia_available():
        yielded = False
        cartesia_stream = (
            cartesia_session.stream_pcm(clean_text)
            if cartesia_session is not None
            else _cartesia_tts_pcm_stream(clean_text)
        )
        try:
            first_chunk = await asyncio.wait_for(
                cartesia_stream.__anext__(),
                timeout=DEMO_CARTESIA_FIRST_CHUNK_TIMEOUT,
            )
        except StopAsyncIteration:
            first_chunk = b""
        except asyncio.TimeoutError:
            logger.warning(
                "[TTS] Cartesia first chunk exceeded %.1fs; falling back to Deepgram",
                DEMO_CARTESIA_FIRST_CHUNK_TIMEOUT,
            )
            _record_cartesia_failure("first chunk timeout")
            if cartesia_session is not None:
                await cartesia_session.close()
            else:
                try:
                    await cartesia_stream.aclose()
                except Exception:
                    pass
            first_chunk = b""

        if first_chunk:
            yielded = True
            chunks.append(first_chunk)
            yield first_chunk
            async for chunk in cartesia_stream:
                if chunk:
                    chunks.append(chunk)
                    yield chunk
        if yielded:
            _record_cartesia_success()
            if key:
                BUFFER_AUDIO_CACHE[key] = _make_streaming_wav_header() + b"".join(chunks)
            return

    async for chunk in _deepgram_tts_pcm_stream(clean_text):
        if chunk:
            chunks.append(chunk)
            yield chunk
    if key and chunks:
        BUFFER_AUDIO_CACHE[key] = _make_streaming_wav_header() + b"".join(chunks)


def _cache_key(text: str) -> str:
    return clean_text_for_tts(text).strip()


def _cartesia_available() -> bool:
    return bool(
        CARTESIA_API_KEY
        and DEMO_CARTESIA_VOICE_ID
        and time.time() >= CARTESIA_DISABLED_UNTIL
    )


def _record_cartesia_success() -> None:
    global CARTESIA_FAILURES, CARTESIA_DISABLED_UNTIL
    CARTESIA_FAILURES = 0
    CARTESIA_DISABLED_UNTIL = 0.0


def _record_cartesia_failure(reason: str) -> None:
    global CARTESIA_FAILURES, CARTESIA_DISABLED_UNTIL
    CARTESIA_FAILURES += 1
    if CARTESIA_FAILURES >= CARTESIA_FAILURE_LIMIT:
        CARTESIA_DISABLED_UNTIL = time.time() + CARTESIA_COOLDOWN_SECONDS
        logger.warning(
            "[TTS] disabling Cartesia for %ss after %s failures; reason=%s",
            CARTESIA_COOLDOWN_SECONDS,
            CARTESIA_FAILURES,
            reason,
        )


async def _buffer_tts_audio(text: str) -> bytes:
    chunks = []
    async for chunk in _tts_stream(text):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


async def _prewarm_common_audio() -> None:
    for text in COMMON_DEMO_SPEECH:
        key = _cache_key(text)
        if not key or key in BUFFER_AUDIO_CACHE:
            continue
        try:
            started = time.perf_counter()
            wav = await _buffer_tts_audio(text)
            if len(wav) > 44:
                BUFFER_AUDIO_CACHE[key] = wav
                logger.info(
                    "[DEMO-TTS] cached common phrase (%s bytes, %.0fms): %s",
                    len(wav),
                    (time.perf_counter() - started) * 1000,
                    key[:60],
                )
        except Exception as exc:
            logger.warning("[DEMO-TTS] common phrase prewarm failed: %s", exc)


@app.on_event("startup")
async def startup_demo_cache() -> None:
    asyncio.create_task(_prewarm_common_audio())


async def _send_tts_audio_stream(
    websocket: WebSocket,
    text: str,
    metadata: Dict[str, Any],
    cartesia_session: Optional[Any] = None,
) -> Dict[str, Any]:
    stream_id = str(uuid.uuid4())
    started = time.perf_counter()
    await websocket.send_json({
        "type": "agent_audio_start",
        "stream_id": stream_id,
        "text": text,
        "sample_rate": DEMO_TTS_SAMPLE_RATE,
        "encoding": "pcm_s16le",
        **metadata,
    })

    bytes_sent = 0
    chunk_count = 0
    first_chunk_ms: Optional[int] = None
    first_chunk_at: Optional[float] = None
    async for chunk in _tts_pcm_stream_for_browser(text, cartesia_session=cartesia_session):
        if not chunk:
            continue
        if first_chunk_ms is None:
            first_chunk_at = time.perf_counter()
            first_chunk_ms = round((first_chunk_at - started) * 1000)
            await websocket.send_json({
                "type": "agent_audio_first_chunk",
                "stream_id": stream_id,
                "tts_first_chunk_ms": first_chunk_ms,
            })
        chunk_count += 1
        bytes_sent += len(chunk)
        await websocket.send_bytes(chunk)

    total_ms = round((time.perf_counter() - started) * 1000)
    await websocket.send_json({
        "type": "agent_audio_end",
        "stream_id": stream_id,
        "text": text,
        "bytes": bytes_sent,
        "chunks": chunk_count,
        "tts_first_chunk_ms": first_chunk_ms,
        "tts_stream_ms": total_ms,
        **metadata,
    })
    return {
        "stream_id": stream_id,
        "bytes": bytes_sent,
        "chunks": chunk_count,
        "tts_first_chunk_ms": first_chunk_ms,
        "tts_stream_ms": total_ms,
        "first_chunk_at": first_chunk_at,
    }


def _start_streaming_audio(text: str) -> str:
    token = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    key = _cache_key(text)
    STREAM_AUDIO_CACHE[token] = {
        "queue": queue,
        "created_at": time.time(),
        "chunks": [],
        "bytes_sent": 0,
    }

    async def _producer():
        try:
            async for chunk in _tts_stream(text):
                entry = STREAM_AUDIO_CACHE.get(token)
                if entry is None:
                    break
                entry["chunks"].append(chunk)
                entry["bytes_sent"] += len(chunk)
                await queue.put(chunk)
        except Exception as exc:
            logger.warning("[TTS] stream failed: %s", exc)
        finally:
            await queue.put(None)
            entry = STREAM_AUDIO_CACHE.get(token)
            if entry is not None:
                buffer = b"".join(entry["chunks"])
                if key and len(buffer) > 44:
                    BUFFER_AUDIO_CACHE[key] = buffer
                STREAM_AUDIO_CACHE[token] = {
                    "buffer": buffer,
                    "expires_at": time.time() + AUDIO_CACHE_TTL,
                }

    asyncio.create_task(_producer())
    return token


def _start_audio_response(text: str) -> str:
    key = _cache_key(text)
    wav = BUFFER_AUDIO_CACHE.get(key)
    if wav:
        token = str(uuid.uuid4())
        STREAM_AUDIO_CACHE[token] = {
            "buffer": wav,
            "expires_at": time.time() + AUDIO_CACHE_TTL,
        }
        return token
    return _start_streaming_audio(text)


async def _fast_llm_response(
    transcript: str,
    patient_context: Dict[str, Any],
    current_state: str,
    history: List[Dict[str, str]],
) -> tuple[str, Optional[str], dict, str, str]:
    if LLM_BACKEND.lower() == "groq":
        api_key, model, backend = GROQ_API_KEY, GROQ_MODEL, "groq"
    else:
        api_key, model, backend = GOOGLE_API_KEY, GEMINI_MODEL, "gemini"

    state = current_state or "greeting"
    local_response = _local_turn_response(transcript, patient_context, state)
    if local_response:
        return local_response

    if state == "greeting":
        node = await decide_node(transcript, state, history, api_key, model, backend)
    elif state == "availability_check":
        node = await decide_node(transcript, state, history, api_key, model, backend)
    else:
        node = await decide_node(transcript, state, history, api_key, model, backend)

    agent_state = build_agent_state(
        transcript,
        {**patient_context, "agent_name": AGENT_NAME},
        state,
        history,
        api_key,
        model,
        backend,
        previous_node=current_state,
    )
    speech, action, action_data, next_state = await run_node_for_state(node, agent_state)
    return speech, action, action_data, next_state, node


@app.get("/")
async def index():
    html = Path("web_voice_demo.html").read_text(encoding="utf-8")
    return FastAPIResponse(
        content=html,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/demo-config")
async def demo_config():
    return {
        "greeting": GREETING_TEXT,
        "patient_context": DEFAULT_PATIENT_CONTEXT,
        "stt_engine": "deepgram_streaming",
        "stt_sample_rate": DEMO_STT_SAMPLE_RATE,
        "tts_engine": "cartesia" if _cartesia_available() else "deepgram",
        "tts_voice": DEMO_CARTESIA_VOICE_ID if _cartesia_available() else DEMO_DEEPGRAM_TTS_VOICE,
        "tts_sample_rate": DEMO_TTS_SAMPLE_RATE,
        "cached_common_replies": len(BUFFER_AUDIO_CACHE),
        "hospital_rag_documents": HOSPITAL_RAG.document_count,
        "hospital_rag_cache_entries": HOSPITAL_RAG.cache_size,
        "demo_version": str(int(time.time())),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "deepgram": bool(DEEPGRAM_API_KEY),
        "llm_backend": LLM_BACKEND,
        "tts_engine": "cartesia" if _cartesia_available() else "deepgram",
        "cartesia_disabled_seconds": max(0, round(CARTESIA_DISABLED_UNTIL - time.time())),
        "hospital_rag_documents": HOSPITAL_RAG.document_count,
        "hospital_rag_cache_entries": HOSPITAL_RAG.cache_size,
    }


@app.get("/api/rag-debug")
async def rag_debug(q: str = "doctor specialty"):
    answer = HOSPITAL_RAG.answer(q, DEFAULT_PATIENT_CONTEXT)
    hits = HOSPITAL_RAG.search(q)
    return {
        "query": q,
        "answer": answer.speech if answer else None,
        "node": answer.node if answer else None,
        "documents": HOSPITAL_RAG.document_count,
        "cache_entries": HOSPITAL_RAG.cache_size,
        "hits": [
            {"id": hit["id"], "kind": hit["kind"], "score": hit["score"]}
            for hit in hits
        ],
    }


@app.get("/audio-worklet.js")
async def audio_worklet():
    script = Path("web_voice_worklet.js").read_text(encoding="utf-8")
    return FastAPIResponse(
        content=script,
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/audio/{token}")
async def serve_audio(token: str):
    entry = STREAM_AUDIO_CACHE.get(token)
    if entry is None:
        return FastAPIResponse(status_code=404, content=b"Audio not found")

    if isinstance(entry, dict) and "queue" in entry:
        async def chunk_generator():
            queue = entry["queue"]
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        return StreamingResponse(
            chunk_generator(),
            media_type="audio/wav",
            headers={"Transfer-Encoding": "chunked"},
        )

    if isinstance(entry, dict) and "buffer" in entry:
        if time.time() > entry.get("expires_at", 0):
            STREAM_AUDIO_CACHE.pop(token, None)
            return FastAPIResponse(status_code=404, content=b"Audio expired")
        return FastAPIResponse(content=entry["buffer"], media_type="audio/wav")

    return FastAPIResponse(status_code=404, content=b"Audio unavailable")


@app.post("/api/speak")
async def speak(req: SpeakRequest):
    audio_token = _start_audio_response(req.text)
    return {
        "text": req.text,
        "audio_url": f"/api/audio/{audio_token}",
        "tts_ms": 0,
    }


@app.post("/api/transcribe")
async def transcribe(req: TranscribeRequest):
    started = time.perf_counter()
    if not DEEPGRAM_API_KEY:
        return {"transcript": "", "error": "DEEPGRAM_API_KEY is not set"}

    try:
        audio = base64.b64decode(req.audio_base64)
    except Exception:
        return {"transcript": "", "error": "Invalid audio payload"}

    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&smart_format=true"
        "&punctuate=true"
        "&no_delay=true"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": req.content_type or "audio/webm",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, content=audio)
        response.raise_for_status()
        data = response.json()
        alternative = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        return {
            "transcript": (alternative.get("transcript") or "").strip(),
            "confidence": alternative.get("confidence"),
            "stt_ms": round((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        return {"transcript": "", "error": str(exc)}


@app.post("/api/voice-turn")
async def voice_turn(req: VoiceTurnRequest):
    started = time.perf_counter()
    transcript = req.transcript.strip()
    if not transcript:
        return {
            "speech": "",
            "action": None,
            "action_data": {},
            "next_state": req.current_state,
            "audio_url": None,
        }

    llm_started = time.perf_counter()
    speech, action, action_data, next_state, node = await _fast_llm_response(
        transcript,
        req.patient_context or DEFAULT_PATIENT_CONTEXT,
        req.current_state,
        req.history or [],
    )
    speech = _humanize_reply(transcript, node, speech, req.current_state)
    llm_ms = round((time.perf_counter() - llm_started) * 1000)

    tts_started = time.perf_counter()
    audio_token = _start_audio_response(speech) if speech else None
    tts_ms = round((time.perf_counter() - tts_started) * 1000)

    return {
        "speech": speech,
        "action": action,
        "action_data": action_data or {},
        "next_state": next_state,
        "node": node,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
        "total_ms": round((time.perf_counter() - started) * 1000),
        "audio_url": f"/api/audio/{audio_token}" if audio_token else None,
    }


@app.websocket("/ws/demo-stream")
async def demo_stream(websocket: WebSocket):
    await websocket.accept()
    if not DEEPGRAM_API_KEY:
        await websocket.send_json({"type": "error", "message": "DEEPGRAM_API_KEY is not set"})
        await websocket.close()
        return

    import websockets

    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&smart_format=true"
        "&encoding=linear16"
        f"&sample_rate={DEMO_STT_SAMPLE_RATE}"
        "&channels=1"
        "&interim_results=true"
        "&endpointing=120"
        "&utterance_end_ms=1000"
    )

    current_state = "greeting"
    history: List[Dict[str, str]] = []
    patient_context: Dict[str, Any] = dict(DEFAULT_PATIENT_CONTEXT)
    processing_turn = False
    last_audio_at = time.perf_counter()
    final_parts: List[str] = []
    cartesia_session = (
        PersistentCartesiaTTS()
        if _cartesia_available()
        else None
    )

    async def process_transcript(transcript: str):
        nonlocal current_state, history, processing_turn
        if processing_turn:
            return
        text = transcript.strip()
        if not text:
            return

        processing_turn = True
        turn_started = time.perf_counter()
        history_for_llm = history[-8:]
        await websocket.send_json({
            "type": "user_final",
            "transcript": text,
            "stt_ms": round((time.perf_counter() - last_audio_at) * 1000),
        })
        asyncio.create_task(
            HOSPITAL_RAG.prefetch_for_turn(text, current_state, history_for_llm)
        )
        try:
            local_response = _local_turn_response(text, patient_context, current_state)
            if local_response:
                speech, action, action_data, next_state, node = local_response
                speech = _humanize_reply(text, node, speech, current_state)

                history.append({"role": "user", "content": text})
                if speech:
                    history.append({"role": "assistant", "content": speech})
                current_state = next_state or current_state

                audio_stats = {"tts_first_chunk_ms": None}
                if speech:
                    audio_stats = await _send_tts_audio_stream(
                        websocket,
                        speech,
                        {
                            "node": node,
                            "sentence_index": 1,
                            "llm_ms": 0,
                            "ttft_ms": 0,
                            "next_state": current_state,
                        },
                        cartesia_session=cartesia_session,
                    )

                await websocket.send_json({
                    "type": "agent_response_done",
                    "speech": speech,
                    "action": action,
                    "action_data": action_data or {},
                    "next_state": current_state,
                    "node": node,
                    "llm_ms": 0,
                    "ttft_ms": 0,
                    "ttfa_ms": round((audio_stats["first_chunk_at"] - turn_started) * 1000) if audio_stats.get("first_chunk_at") else None,
                    "total_ms": round((time.perf_counter() - turn_started) * 1000),
                    "sentence_count": 1 if speech else 0,
                    "audio_transport": "websocket_pcm",
                })
                return

            if LLM_BACKEND.lower() == "groq":
                api_key, model, backend = GROQ_API_KEY, GROQ_MODEL, "groq"
            else:
                api_key, model, backend = GOOGLE_API_KEY, GEMINI_MODEL, "gemini"

            llm_started = time.perf_counter()
            node = await decide_node(text, current_state, history_for_llm, api_key, model, backend)
            agent_state = build_agent_state(
                text,
                {**patient_context, "agent_name": AGENT_NAME},
                current_state,
                history_for_llm,
                api_key,
                model,
                backend,
                previous_node=current_state,
            )

            await websocket.send_json({
                "type": "agent_response_start",
                "node": node,
                "routing_ms": round((time.perf_counter() - llm_started) * 1000),
            })

            first_audio_ready_at: Optional[float] = None
            done_event: Dict[str, Any] = {
                "speech": "",
                "action": None,
                "action_data": {},
                "next_state": current_state,
                "ttft_ms": None,
                "llm_ms": 0,
            }
            sentence_index = 0
            first_token_ms: Optional[int] = None
            spoken_sentences: List[str] = []

            async for event in stream_node_speech_for_state(node, agent_state):
                if event.get("type") == "first_token":
                    first_token_ms = event.get("ttft_ms")
                    await websocket.send_json({
                        "type": "agent_first_token",
                        "node": node,
                        "ttft_ms": first_token_ms,
                    })
                    continue

                if event.get("type") == "sentence":
                    sentence = (event.get("text") or "").strip()
                    if not sentence:
                        continue
                    sentence_index += 1
                    if sentence_index == 1:
                        sentence = _humanize_reply(text, node, sentence, current_state)
                    spoken_sentences.append(sentence)
                    audio_stats = await _send_tts_audio_stream(
                        websocket,
                        sentence,
                        {
                            "node": node,
                            "sentence_index": sentence_index,
                            "llm_ms": round((time.perf_counter() - llm_started) * 1000),
                            "ttft_ms": first_token_ms,
                            "elapsed_ms": round((time.perf_counter() - turn_started) * 1000),
                        },
                        cartesia_session=cartesia_session,
                    )
                    if first_audio_ready_at is None:
                        first_audio_ready_at = audio_stats.get("first_chunk_at")
                    continue

                if event.get("type") == "done":
                    done_event = event

            speech = " ".join(spoken_sentences).strip() or (done_event.get("speech") or "").strip()
            current_state = done_event.get("next_state") or current_state
            history.append({"role": "user", "content": text})
            if speech:
                history.append({"role": "assistant", "content": speech})

            await websocket.send_json({
                "type": "agent_response_done",
                "speech": speech,
                "action": done_event.get("action"),
                "action_data": done_event.get("action_data") or {},
                "next_state": current_state,
                "node": node,
                "llm_ms": done_event.get("llm_ms") or round((time.perf_counter() - llm_started) * 1000),
                "ttft_ms": done_event.get("ttft_ms"),
                "ttfa_ms": round((first_audio_ready_at - turn_started) * 1000) if first_audio_ready_at else None,
                "total_ms": round((time.perf_counter() - turn_started) * 1000),
                "sentence_count": sentence_index,
            })
        except Exception as exc:
            logger.exception("[DEMO-WS] turn failed")
            await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            processing_turn = False

    try:
        async with websockets.connect(
            dg_url,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            open_timeout=25,
            ping_interval=20,
            ping_timeout=10,
        ) as dg_ws:
            await websocket.send_json({
                "type": "ready",
                "stt_engine": "deepgram_streaming",
                "stt_sample_rate": DEMO_STT_SAMPLE_RATE,
            })
            if cartesia_session is not None:
                async def warm_cartesia_session():
                    try:
                        await cartesia_session._connect()
                    except Exception as exc:
                        logger.warning("[TTS] persistent Cartesia preconnect failed: %s", exc)

                asyncio.create_task(warm_cartesia_session())

            async def browser_to_deepgram():
                nonlocal current_state, history, patient_context, last_audio_at
                try:
                    while True:
                        message = await websocket.receive()
                        if "bytes" in message and message["bytes"] is not None:
                            last_audio_at = time.perf_counter()
                            await dg_ws.send(message["bytes"])
                        elif "text" in message and message["text"] is not None:
                            payload = json.loads(message["text"])
                            msg_type = payload.get("type")
                            if msg_type == "start":
                                current_state = payload.get("current_state") or "greeting"
                                history = payload.get("history") or []
                                patient_context = payload.get("patient_context") or dict(DEFAULT_PATIENT_CONTEXT)
                            elif msg_type == "reset":
                                current_state = "greeting"
                                history = []
                                patient_context = dict(DEFAULT_PATIENT_CONTEXT)
                            elif msg_type == "stop":
                                await dg_ws.close()
                                return
                except (WebSocketDisconnect, RuntimeError):
                    await dg_ws.close()

            async def deepgram_to_browser():
                nonlocal final_parts
                async for raw in dg_ws:
                    data = json.loads(raw)
                    if data.get("type") == "UtteranceEnd":
                        final_text = " ".join(final_parts).strip()
                        final_parts = []
                        if final_text:
                            await process_transcript(final_text)
                        continue
                    if data.get("type") not in (None, "Results"):
                        continue
                    alternative = (
                        data.get("channel", {})
                        .get("alternatives", [{}])[0]
                    )
                    transcript = (alternative.get("transcript") or "").strip()
                    if not transcript:
                        continue

                    if data.get("is_final"):
                        final_parts.append(transcript)
                    else:
                        await websocket.send_json({
                            "type": "interim",
                            "transcript": transcript,
                        })

                    if data.get("speech_final"):
                        final_text = " ".join(final_parts).strip() or transcript
                        final_parts = []
                        await process_transcript(final_text)

            async def deepgram_keepalive():
                while True:
                    await asyncio.sleep(4)
                    try:
                        await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        return

            await asyncio.gather(
                browser_to_deepgram(),
                deepgram_to_browser(),
                deepgram_keepalive(),
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("[DEMO-WS] stream failed")
        message = str(exc)
        if "timed out during opening handshake" in message:
            message = "Deepgram STT connection timed out while starting. Please retry in a few seconds."
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if cartesia_session is not None:
            await cartesia_session.close()
