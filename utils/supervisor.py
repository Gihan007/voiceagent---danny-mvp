"""
LangGraph Supervisor Agent for Voice Call Flow — v2

Fixes over v1:
  • Increased max_tokens to 150 for node calls → prevents unterminated-string JSON errors
  • Robust JSON repair (_repair_json) handles truncated/malformed output
  • Per-node fallback next_state instead of blindly copying call_state
  • wrap_up node — "Is there anything else I can help you with?" after every action
  • Supervisor transitions are intent-based, not rigidly state-gated
  • Lambdas for fallback speech so they access live state at error time

Call flow:
  greeting → availability_check → appointment_review
      → confirm   → wrap_up → ended
      → reschedule → wrap_up → ended
      → cancel     → wrap_up → ended
      → general_question (any time) → back to appointment_review or wrap_up
  availability_check → callback_request → ended
"""

import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator, TypedDict, Optional

import httpx
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

_client = httpx.AsyncClient(timeout=20.0)


# ──────────────────────────────────────────────────────────────────────────────
# Gemini helper
# ──────────────────────────────────────────────────────────────────────────────

async def _gemini(
    system: str,
    user: str,
    api_key: str,
    model: str,
    max_tokens: int = 150,
    json_mode: bool = True,
) -> str:
    """Single Gemini call with exponential backoff on 429.

    thinkingBudget=0 disables Gemini 2.5 Flash's internal reasoning — thinking
    tokens count against maxOutputTokens, so without this the model thinks for
    ~40 tokens and only has ~10 left for the actual JSON reply → MAX_TOKENS.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    gen_cfg: dict = {
        "temperature": 0.2,
        "maxOutputTokens": max_tokens,
        "topP": 0.9,
        "thinkingConfig": {"thinkingBudget": 0},   # ← disable thinking; saves tokens + ~300ms
    }
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_cfg,
    }
    for attempt in range(3):
        r = await _client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        if r.status_code == 429:
            wait = 2**attempt
            logger.warning(f"[GEMINI] 429 — retry {attempt+1}/3 after {wait}s")
            if attempt < 2:
                await asyncio.sleep(wait)
                continue
            raise RuntimeError(f"Gemini 429: {r.text[:200]}")
        if r.status_code != 200:
            raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in Gemini response: {data.get('promptFeedback', data)}")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            finish = candidates[0].get("finishReason", "unknown")
            raise ValueError(f"Empty parts in Gemini response (finishReason={finish})")
        return parts[0]["text"].strip()
    raise RuntimeError("Gemini 429 after 3 retries")


# ──────────────────────────────────────────────────────────────────────────────
# Groq helper (Llama 3.1 8B — ultra-fast inference)
# ──────────────────────────────────────────────────────────────────────────────

# Per-process cache: models confirmed to not support json_object mode.
# Avoids the 400→retry round-trip on every single call after the first failure.
_GROQ_NO_JSON_MODE: set[str] = set()


async def _groq(
    system: str,
    user: str,
    api_key: str,
    model: str,
    max_tokens: int = 150,
    json_mode: bool = True,
) -> str:
    """Single Groq call with exponential backoff on 429.

    Handles two quirks of non-standard Groq models (e.g. openai/gpt-oss-20b):
      1. json_object response_format → 400  (cached after first failure)
      2. content is null/empty but reasoning_content has the output (thinking models)

    For models that don't support system messages, we fold system+user into a
    single user message so the prompt still reaches the model intact.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _build_payload(use_json_mode: bool) -> dict:
        # Some models reject system role — fold into user turn as fallback
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        p: dict = {
            "model":       model,
            "messages":    messages,
            "temperature": 0.2,
            "max_tokens":  max_tokens,
            "top_p":       0.9,
        }
        if use_json_mode:
            p["response_format"] = {"type": "json_object"}
        return p

    use_json = json_mode and model not in _GROQ_NO_JSON_MODE

    for attempt in range(4):   # extra slot for the json→no-json retry
        r = await _client.post(url, json=_build_payload(use_json), headers=headers)

        if r.status_code == 429:
            wait = 2 ** min(attempt, 2)
            logger.warning(f"[GROQ] 429 — retry {attempt+1} after {wait}s")
            if attempt < 3:
                await asyncio.sleep(wait)
                continue
            raise RuntimeError(f"Groq 429: {r.text[:200]}")

        if r.status_code == 400 and use_json:
            # Permanently cache: this model never supports json_object
            _GROQ_NO_JSON_MODE.add(model)
            logger.warning(f"[GROQ] 400 json_mode unsupported — cached for {model}, retrying")
            use_json = False
            continue

        if r.status_code != 200:
            raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")

        data     = r.json()
        choices  = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in Groq response: {data}")

        msg = choices[0].get("message", {})

        # Standard content field
        content = (msg.get("content") or "").strip()

        # Thinking/reasoning models (e.g. gpt-oss-20b) may put their answer in
        # reasoning_content and leave content null — use it as fallback.
        if not content:
            content = (msg.get("reasoning_content") or "").strip()

        if not content:
            # Log the raw message so we can diagnose unknown model formats
            logger.warning(f"[GROQ] Empty content from {model} — raw msg: {str(msg)[:300]}")
            raise ValueError(f"Empty content in Groq response for model {model}")

        return content

    raise RuntimeError("Groq retries exhausted")


async def _groq_stream_text(
    system: str,
    user: str,
    api_key: str,
    model: str,
    max_tokens: int = 180,
) -> AsyncIterator[str]:
    """Stream plain-text Groq deltas for low-latency spoken responses."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": True,
    }

    async with _client.stream("POST", url, json=payload, headers=headers, timeout=20.0) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise RuntimeError(f"Groq stream {response.status_code}: {body[:200]!r}")

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                delta = event.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or ""
            except Exception:
                token = ""
            if token:
                yield token


# ──────────────────────────────────────────────────────────────────────────────
# LLM dispatcher — routes to Gemini or Groq based on state
# ──────────────────────────────────────────────────────────────────────────────

async def _call_llm(
    system: str,
    user: str,
    api_key: str,
    model: str,
    backend: str = "groq",
    max_tokens: int = 150,
    json_mode: bool = True,
) -> str:
    """Route to Groq or Gemini based on backend flag."""
    if backend.lower() == "groq":
        return await _groq(system, user, api_key, model, max_tokens, json_mode)
    else:  # gemini
        return await _gemini(system, user, api_key, model, max_tokens, json_mode)


# ──────────────────────────────────────────────────────────────────────────────
# Robust JSON parsing with repair
# ──────────────────────────────────────────────────────────────────────────────

def _repair_json(text: str) -> dict:
    """
    Extract and repair common LLM JSON issues:
      - Markdown fences
      - Unterminated strings (most frequent with low max_tokens)
      - Missing closing braces
      - Trailing commas
    """
    # 1. Strip markdown fences
    text = re.sub(r"```json\s*|```\s*", "", text).strip()

    # 2. Find JSON object start
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    text = text[start:]

    # 3. Fast path — valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Close unclosed braces
    opens  = text.count("{")
    closes = text.count("}")
    if opens > closes:
        text += "}" * (opens - closes)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        pass

    # 5. Fix unterminated string: truncate at error pos, close string + braces
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        if e.pos and e.pos > 0:
            truncated = text[: e.pos]
            # Close any open string
            if truncated.count('"') % 2 != 0:
                truncated += '"'
            # Remove trailing comma before we close
            truncated = re.sub(r",\s*$", "", truncated.rstrip())
            # Close braces
            op = truncated.count("{")
            cl = truncated.count("}")
            truncated += "}" * max(0, op - cl)
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

    # 6. Strip trailing commas (another common failure)
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return json.loads(text)   # raises if still broken


def _parse_json(text: str) -> dict:
    result = _repair_json(text)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result)}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    transcript:       str
    call_state:       str
    previous_node:    str   # state we were in BEFORE the current one (used by wrap_up)
    filler_phrase:    str   # bridging phrase already spoken; node must continue from it
    patient_name:     str
    appointment_date: str
    appointment_time: str
    provider_name:    str
    agent_name:       str
    history:          list[dict]
    api_key:          str
    model:            str
    backend:          str   # "gemini" or "groq"
    # outputs (filled by supervisor + nodes)
    node:        str
    speech:      str
    action:      Optional[str]
    action_data: dict
    next_state:  str


_STATE_RANK = {
    "greeting": 0,
    "availability_check": 1,
    "appointment_review": 2,
    "confirm": 3,
    "reschedule": 3,
    "cancel": 3,
    "general_question": 3,
    "wrap_up": 4,
    "callback_request": 4,
    "ended": 5,
}


def _looks_like_new_scheduling_request(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(r"\b(reschedule|change|move|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|march|april|may|june|july|august|september|october|november|december|\d+\s*(am|pm))\b", t)
    )


def guard_node_choice(call_state: str, node: str, transcript: str = "") -> str:
    """Block obvious state regressions while still allowing new user intent."""
    if node == "greeting" and call_state != "greeting":
        return call_state

    if call_state in {"confirm", "reschedule", "cancel", "wrap_up"}:
        if node in {"greeting", "availability_check", "appointment_review"}:
            if _looks_like_new_scheduling_request(transcript):
                return "reschedule"
            return "wrap_up" if call_state == "wrap_up" else call_state

    return node


def guard_next_state(call_state: str, node_key: str, proposed: str, transcript: str = "") -> str:
    """Keep the call moving forward unless the user clearly introduces a new task."""
    next_state = proposed or call_state
    if next_state not in _STATE_RANK:
        return call_state

    if call_state in {"confirm", "reschedule", "cancel", "wrap_up"}:
        if next_state in {"greeting", "availability_check", "appointment_review"}:
            if _looks_like_new_scheduling_request(transcript):
                return "reschedule"
            return "wrap_up" if node_key in {"general_question", "wrap_up"} else call_state

    if _STATE_RANK.get(next_state, 0) < _STATE_RANK.get(call_state, 0):
        return call_state

    return next_state


def _node_defaults(node_key: str, state: AgentState) -> tuple[Optional[str], dict, str]:
    action = None
    action_data = {}
    next_state = _FALLBACK_NEXT.get(node_key, state["call_state"])
    if node_key == "confirm":
        action = "confirm_appointment"
        next_state = "wrap_up"
    elif node_key == "cancel":
        action = "cancel_appointment"
        next_state = "wrap_up"
    elif node_key == "reschedule":
        if _looks_like_new_scheduling_request(state["transcript"]):
            action = "reschedule_request"
        next_state = "reschedule"
    return action, action_data, next_state


def _build_node_messages(state: AgentState, node_key: str, *, json_mode: bool) -> tuple[str, str]:
    prompt_tmpl = _NODE_PROMPTS.get(node_key, _NODE_PROMPTS["general_question"])
    system = prompt_tmpl.format(
        agent=state["agent_name"],
        name=state["patient_name"],
        date=state["appointment_date"],
        time=state["appointment_time"],
        provider=state["provider_name"],
        previous_node=state.get("previous_node", ""),
        call_state=state["call_state"],
    )

    if not json_mode:
        node_tasks = {
            "greeting": "confirm you are speaking with the right patient",
            "availability_check": "ask whether the patient has a quick minute to talk",
            "appointment_review": "review the appointment and ask whether it still works",
            "confirm": "confirm the appointment is all set and move toward wrap-up",
            "reschedule": "collect the patient's preferred new day/time for rescheduling; do not claim you checked live availability",
            "cancel": "handle the cancellation with empathy",
            "general_question": "answer only using known appointment facts and avoid inventing details",
            "wrap_up": "ask if anything else is needed or say goodbye if the patient is done",
            "callback_request": "arrange a better time to call back",
        }
        system = (
            f"You are {state['agent_name']}, a warm clinic receptionist on a live phone call.\n"
            f"Patient name: {state['patient_name']}\n"
            f"Appointment: {state['appointment_date']} at {state['appointment_time']} with {state['provider_name']}\n"
            f"Current call state: {state['call_state']}\n"
            f"Your current task: {node_tasks.get(node_key, node_tasks['general_question'])}.\n\n"
            "Output ONLY the exact words to say aloud. Do not output JSON, labels, markdown, bullets, or metadata. "
            "Keep it concise: one or two short spoken sentences. "
            "If asked about facts you do not know, say you only have the appointment details. "
            "Never say you checked the schedule, found availability, or found unavailability. "
            "For rescheduling, only say you can note the preference or ask for a backup time."
        )

    filler = state.get("filler_phrase", "").strip()
    if filler:
        system += (
            f"\n\nYou already said \"{filler}\" to the patient just now. "
            f"Your response must flow naturally from that, don't repeat it, "
            f"and don't contradict it."
        )

    recent = state["history"][-6:]
    history_lines = []
    for m in recent:
        role = "Patient" if m["role"] == "user" else "Agent"
        content = m["content"]
        if len(content) > 120:
            content = content[:117] + "..."
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)
    user_msg = (
        f"{history_text}\nPatient: {state['transcript']}"
        if history_text
        else f"Patient: {state['transcript']}"
    )
    return system, user_msg


def _extract_complete_sentences(buffer: str) -> tuple[list[str], str]:
    sentences = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", buffer):
        sentence = buffer[start:match.end()].strip()
        if re.search(r"\b(?:Dr|Mr|Mrs|Ms)\.$", sentence):
            continue
        if sentence:
            sentences.append(sentence)
        start = match.end()
    return sentences, buffer[start:].lstrip()


# ──────────────────────────────────────────────────────────────────────────────
# Supervisor node — intent-based routing, not rigid state-gating
# ──────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are routing a live phone call for a clinic appointment reminder. Listen to what the patient just said and pick the most natural next step.

Nodes you can route to:
  greeting          — confirm you've reached the right person
  availability_check — check if they have a moment to talk
  appointment_review — go over their upcoming appointment and ask if it still works
  confirm           — they're keeping the appointment, wrap it up warmly
  reschedule        — they want a different time, help them sort it out
  cancel            — they want to cancel, handle it with empathy
  general_question  — they asked something, or said yes to "anything else?" — let them ask
  wrap_up           — appointment is sorted, check if they need anything else before goodbye
  callback_request  — they're busy or driving, offer to call back later

HOW TO DECIDE (go through these in order):
0. Once the call has moved past the greeting, never go back to it. If the patient says "hi", "hello", "yeah", "okay" mid-call, treat it as a reply to what the agent just said — not a fresh start.
1. Busy / driving / bad time / can't talk right now → callback_request
2. "Who is this?" / "Which clinic?" / "Who's calling?" → general_question (they want information, not a re-greeting)
3. "Yes" / "sure" / "okay" / "go ahead" when agent just asked if they can talk (availability_check state) → appointment_review
4. When you're in wrap_up:
   - Short farewell or acknowledgment (bye, thanks, okay, that's fine, all good, nothing else, I'm good) → wrap_up (node handles the goodbye)
   - "Yes" / "yeah" / "sure" with no actual question → wrap_up (ask again if they need anything)
   - A real question (where, how, what, directions, etc.) → general_question
5. A specific question about the clinic, appointment, directions, parking, etc. → general_question
6. Wants to change the date or time → reschedule
7. Wants to cancel → cancel
8. Agreed to keep the appointment (yes, sounds good, that works, confirmed) → confirm
9. Not sure? Stay in the current state.

Reply ONLY with JSON — no extra text: {"node": "<node_name>"}"""


_VALID_NODES = frozenset({
    "greeting", "availability_check", "appointment_review",
    "confirm", "reschedule", "cancel", "general_question",
    "wrap_up", "callback_request",
})


# ──────────────────────────────────────────────────────────────────────────────
# Smart filler — "conversation continue" LLM
# Runs in parallel with the node LLM. Generates a short context-aware bridge
# phrase (~200ms) instead of a generic static "One moment...".
# No thinking, no JSON, max 20 output tokens → very fast.
# ──────────────────────────────────────────────────────────────────────────────

# Per-node labels describe what action is about to happen — used in filler prompt
# so the LLM can generate a phrase that naturally bridges to the node's response.
_NODE_ACTION_LABELS: dict[str, str] = {
    "greeting":           "greet and confirm who they are speaking with",
    "availability_check": "ask if they can talk right now",
    "appointment_review": "review their upcoming appointment",
    "confirm":            "confirm their appointment is all set",
    "reschedule":         "help them reschedule their appointment",
    "cancel":             "process their appointment cancellation",
    "general_question":   "answer their question",
    "wrap_up":            "wrap up the call",
    "callback_request":   "arrange to call them back",
}

# Static fallback per node — used when LLM call fails or times out
_FILLER_FALLBACK: dict[str, str] = {
    "greeting":           "Thanks for confirming",
    "availability_check": "Perfect",
    "appointment_review": "Great",
    "confirm":            "Wonderful",
    "reschedule":         "Of course",
    "cancel":             "I understand",
    "general_question":   "Sure thing",
    "wrap_up":            "Of course",
    "callback_request":   "No problem at all",
}

_FILLER_SYSTEM = (
    "You are a warm clinic receptionist on a live phone call.\n"
    "The patient just replied and you need to say ONE short bridging phrase (3–6 words) that:\n"
    "  1. Reacts naturally to what the patient said\n"
    "  2. Leads smoothly into what the agent will say next (next_action)\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the phrase — nothing else, no punctuation at the end\n"
    "- Never ask a question\n"
    "- Never use the patient's name, dates, or times\n"
    "- If next_action is wrap_up and patient is still talking, use a continuing phrase (Perfect / Of course / Sure thing)\n"
    "- ONLY use farewell words (thank you, take care, have a great day) when next_action=wrap_up AND patient gave a clear goodbye signal\n"
    "\n"
    "-- EXAMPLES (next_action = the node name shown exactly) --\n"
    "\n"
    "call_stage=greeting, next_action=availability_check:\n"
    "  patient says 'yes speaking' / 'that's me' / 'yes I'm John'  -> Thanks for confirming\n"
    "  patient says 'hello' / 'yes'                                -> Great to hear from you\n"
    "\n"
    "call_stage=availability_check, next_action=appointment_review:\n"
    "  patient says 'yes I can talk' / 'go ahead'                  -> Perfect\n"
    "  patient says 'yeah sure' / 'I have a minute'                -> Wonderful\n"
    "\n"
    "call_stage=appointment_review, next_action=confirm:\n"
    "  patient says 'yes that works' / 'that's fine'               -> Glad to hear that\n"
    "  patient says 'sounds good' / 'that's working'               -> That's great\n"
    "\n"
    "call_stage=appointment_review, next_action=reschedule:\n"
    "  patient says 'I need to reschedule'                         -> Of course I can help\n"
    "\n"
    "call_stage=appointment_review, next_action=cancel:\n"
    "  patient says 'I want to cancel'                             -> I completely understand\n"
    "\n"
    "call_stage=confirm, next_action=wrap_up:\n"
    "  patient says 'okay' / 'great'                               -> Perfect\n"
    "\n"
    "call_stage=wrap_up, next_action=wrap_up (call continuing):\n"
    "  patient says 'okay' / 'sure'                                -> Of course\n"
    "  patient says 'yes please' / 'I have a question'             -> Sure thing\n"
    "\n"
    "call_stage=wrap_up, next_action=wrap_up (patient saying goodbye):\n"
    "  patient says 'no all good' / 'that's fine' / 'bye'          -> Wonderful\n"
    "\n"
    "call_stage=any, next_action=general_question:\n"
    "  patient asks a question                                      -> Sure let me check\n"
    "\n"
    "call_stage=any, next_action=callback_request:\n"
    "  patient says busy / driving / bad time                      -> No problem at all\n"
)


async def generate_smart_filler(
    transcript: str,
    call_state: str,
    target_node: str,
    api_key: str,
    model: str,
    backend: str = "groq",
    history: list = [],
) -> str:
    """
    Generate a transcript-aware bridging phrase while the node LLM processes.
    Reads what the patient actually said to produce a meaningful instant reaction.
    Plain-text output, max 25 tokens → ~150-250ms.
    """
    last_agent = next(
        (m["content"][:80] for m in reversed(history) if m["role"] == "assistant"),
        None,
    )
    ctx = f"Agent just said: \"{last_agent}\"\n" if last_agent else ""
    user_msg = f"{ctx}Call stage: {call_state}\nNext action: {target_node}\nPatient said: \"{transcript}\""
    try:
        phrase = await _call_llm(
            _FILLER_SYSTEM, user_msg, api_key, model,
            backend=backend, max_tokens=25, json_mode=False,
        )
        phrase = phrase.strip().strip('"').rstrip(".,!?")
        if phrase and "?" not in phrase and 2 <= len(phrase.split()) <= 10:
            return phrase
    except Exception as e:
        logger.debug(f"[FILLER-LLM] failed: {e}")
    return _FILLER_FALLBACK.get(target_node, _FILLER_FALLBACK.get(call_state, "Of course"))


def _format_history(history: list, max_pairs: int = 3) -> str:
    """Format the last N exchange pairs as 'Agent: ...\nPatient: ...' lines."""
    recent = history[-(max_pairs * 2):]
    lines = []
    for m in recent:
        role = "Agent" if m["role"] == "assistant" else "Patient"
        content = m["content"][:100]
        if len(m["content"]) > 100:
            content += "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def supervisor_node(state: AgentState) -> AgentState:
    history_text = _format_history(state.get("history", []))
    ctx = f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    user_msg = (
        f"{ctx}"
        f"Current state: {state['call_state']}\n"
        f"Patient just said: \"{state['transcript']}\""
    )
    node = state["call_state"]   # safe default
    try:
        raw    = await _call_llm(SUPERVISOR_SYSTEM, user_msg, state["api_key"], state["model"], state["backend"], max_tokens=50)
        parsed = _parse_json(raw)
        choice = parsed.get("node", "")
        if choice in _VALID_NODES:
            node = choice
        else:
            logger.warning(f"[SUPERVISOR] Unknown node '{choice}' — keeping {node}")
    except Exception as e:
        logger.warning(f"[SUPERVISOR] Failed: {e} — defaulting to {node}")

    logger.info(f"[SUPERVISOR] state={state['call_state']} → node={node}")
    return {**state, "node": node}


# ──────────────────────────────────────────────────────────────────────────────
# Node prompts
# ──────────────────────────────────────────────────────────────────────────────
# Prompts are kept concise to stay within max_tokens=150.
# Each prompt embeds the EXACT JSON schema so the model knows what to produce.

_NODE_PROMPTS: dict[str, str] = {

    "greeting": (
        "You're Jamie, a friendly receptionist at Talbot Health. You've just called {name}. "
        "Say a warm hello and gently check that you're speaking with the right person. "
        "Keep it to one natural sentence — the way you'd actually start a call.\n\n"
        "Reply ONLY with this exact JSON structure:\n"
        '{{"speech": "<one sentence>", "action": null, "action_data": {{}}, "next_state": "availability_check"}}'
    ),

    "availability_check": (
        "You're Jamie, a friendly receptionist at Talbot Health. You've already said hello to {name}. "
        "Now just check if they have a minute to talk — keep it short and friendly, like you would on a real call. "
        "If they sound busy, driving, or say it's a bad time, be understanding and let them know you'll try again later. "
        "One or two sentences at most.\n\n"
        "next_state must be EXACTLY one of: availability_check | appointment_review | callback_request\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": null, "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "appointment_review": (
        "You're Jamie, a friendly receptionist at Talbot Health. Jump straight into the reason you called — "
        "you're reminding {name} about their appointment on {date} at {time} with {provider}. "
        "Check if that still works for them. Sound warm and casual, like a helpful reminder call, not a formal announcement. "
        "One or two natural sentences.\n\n"
        "next_state must be EXACTLY one of: appointment_review | confirm | reschedule | cancel | general_question\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": null, "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "confirm": (
        "You're Jamie, a friendly receptionist at Talbot Health. "
        "{name} just said they're happy to keep their appointment on {date} at {time} with {provider}. "
        "Give them a genuine, warm acknowledgment — like you're actually pleased it works out. "
        "One or two easy sentences.\n\n"
        "next_state must be EXACTLY one of: confirm | wrap_up\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": "confirm_appointment", "action_data": {{}}, "next_state": "wrap_up"}}'
    ),

    "reschedule": (
        "You're Jamie, a friendly receptionist at Talbot Health. "
        "{name} wants to move their appointment from {date} at {time}. "
        "Do not claim you checked live availability, found an opening, or found that a day is unavailable. "
        "Only collect or confirm the patient's preferred day/time and ask for a backup if needed. "
        "If they've already told you a new day or time they'd prefer, acknowledge it warmly and let them know you'll sort it out. "
        "If they haven't mentioned a new time yet, just ask what works better — keep it light and easy. "
        "One or two sentences.\n\n"
        "next_state must be EXACTLY one of: reschedule | wrap_up\n"
        "Set action to 'reschedule_request' only once the patient has given you a preferred date or time. "
        "If you're still waiting for that, set action to null.\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": "<reschedule_request or null>", "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "cancel": (
        "You're Jamie, a friendly receptionist at Talbot Health. "
        "{name} wants to cancel their appointment on {date} at {time}. "
        "Check the conversation history first:\n"
        "- If the cancellation hasn't been processed yet: be warm and understanding, and gently ask if there's a reason — but don't push. Set action='cancel_appointment'.\n"
        "- If it's already been handled: just reassure them it's all taken care of. Set action=null.\n"
        "One or two easy sentences.\n\n"
        "next_state must be EXACTLY one of: cancel | wrap_up\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": "<cancel_appointment or null>", "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "general_question": (
        "You're Jamie, a friendly receptionist at Talbot Health. Call stage: {call_state}. "
        "If the patient said something vague like 'yes', 'yeah', 'sure', or 'okay' without asking anything specific, "
        "they're probably signalling they'd like to ask something — invite them warmly, like 'Of course! What would you like to know?' "
        "If they asked a real question, answer it helpfully and briefly (clinic hours, location, what to bring, parking, and so on). "
        "If the appointment is already sorted (call_state is wrap_up, confirm, reschedule, or cancel), don't circle back to appointment_review.\n\n"
        "next_state must be EXACTLY one of: appointment_review | wrap_up | ended\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": null, "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "wrap_up": (
        "You're Jamie, a friendly receptionist at Talbot Health, wrapping up a call with {name}. "
        "The main reason for the call is done (previous step: '{previous_node}'). "
        "Your job now is simple — check once if there's anything else they need, then say a warm goodbye when they're ready.\n\n"
        "First, look at what the patient just said:\n"
        "- If it sounds like a farewell or they're done (bye, thanks, okay, that's fine, all good, nothing else, I'm good, that's all) "
        "→ say goodbye warmly: 'Thanks so much, {name}. Have a wonderful day!' and set next_state='ended'.\n"
        "- If you haven't asked yet whether they need anything else → ask it now, naturally.\n"
        "- If you already asked and they said something vague or positive → invite them: 'Of course! What can I help with?'\n\n"
        "Keep it to one short sentence. Don't repeat appointment details.\n\n"
        "next_state must be EXACTLY one of: wrap_up | ended\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": null, "action_data": {{}}, "next_state": "<state>"}}'
    ),

    "callback_request": (
        "You're Jamie, a friendly receptionist at Talbot Health. "
        "{name} can't talk right now. Let them know you'll reach out again at a better time, and say a warm goodbye. "
        "One natural sentence.\n\n"
        "Reply ONLY with JSON:\n"
        '{{"speech": "<reply>", "action": null, "action_data": {{}}, "next_state": "ended"}}'
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-node fallback speech (lambdas so they read live state) + next_state
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_SPEECH: dict[str, object] = {
    "greeting":
        lambda s: f"Hi there! This is {s['agent_name']} calling from Talbot Health. Am I speaking with {s['patient_name']}?",
    "availability_check":
        lambda _: f"Do you have just a quick moment to chat?",
    "appointment_review":
        lambda s: (
            f"I'm calling about your appointment on {s['appointment_date']} at {s['appointment_time']} "
            f"with {s['provider_name']}. Does that still work for you?"
        ),
    "confirm":
        lambda s: f"That's great — you're all set for {s['appointment_date']} at {s['appointment_time']}!",
    "reschedule":
        lambda s: f"Of course, happy to help with that. What day and time would work better for you?",
    "cancel":
        lambda s: f"Absolutely, I'll take care of that for you. Is there anything in particular that came up?",
    "general_question":
        lambda s: f"Of course! What would you like to know?",
    "wrap_up":
        lambda s: f"Is there anything else I can help you with, {s['patient_name']}?",
    "callback_request":
        lambda s: f"No worries at all — we'll try you again at a better time. Take care!",
}

# Safe fallback next_state per node — never regresses to a past state
_FALLBACK_NEXT: dict[str, str] = {
    "greeting":           "availability_check",
    "availability_check": "availability_check",
    "appointment_review": "appointment_review",
    "confirm":            "wrap_up",
    "reschedule":         "reschedule",
    "cancel":             "cancel",
    "general_question":   "appointment_review",
    "wrap_up":            "ended",
    "callback_request":   "ended",
}


# ──────────────────────────────────────────────────────────────────────────────
# Generic node executor
# ──────────────────────────────────────────────────────────────────────────────

async def _run_node(state: AgentState, node_key: str) -> AgentState:
    system, user_msg = _build_node_messages(state, node_key, json_mode=True)
    speech = ""
    action, action_data, next_state = _node_defaults(node_key, state)

    try:
        raw = await _call_llm(
            system,
            user_msg,
            state["api_key"],
            state["model"],
            state["backend"],
            max_tokens=256,
        )
        parsed = _parse_json(raw)
        speech = parsed.get("speech", "").strip()

        parsed_action = parsed.get("action") or None
        if isinstance(parsed_action, str) and parsed_action.lower() not in ("null", "none", ""):
            action = parsed_action
        elif parsed_action is None:
            action = None

        action_data = parsed.get("action_data") or action_data
        proposed = parsed.get("next_state", "")
        if proposed:
            next_state = guard_next_state(
                state["call_state"], node_key, proposed, state["transcript"]
            )
    except Exception as e:
        logger.error(f"[NODE:{node_key}] LLM/parse failed: {e}")

    if not speech:
        fallback_fn = _FALLBACK_SPEECH.get(node_key)
        speech = fallback_fn(state) if callable(fallback_fn) else "Could you repeat that, please?"
        logger.warning(f"[NODE:{node_key}] Using fallback speech")

    logger.info(f"[NODE:{node_key}] speech={speech[:70]!r}  next_state={next_state}")
    return {
        **state,
        "speech": speech,
        "action": action,
        "action_data": action_data,
        "next_state": next_state,
    }

    prompt_tmpl = _NODE_PROMPTS.get(node_key, _NODE_PROMPTS["general_question"])
    system = prompt_tmpl.format(
        agent=state["agent_name"],
        name=state["patient_name"],
        date=state["appointment_date"],
        time=state["appointment_time"],
        provider=state["provider_name"],
        previous_node=state.get("previous_node", ""),
        call_state=state["call_state"],
    )

    # If a filler phrase was already spoken, tell the node to continue from it
    # naturally — avoid repeating the same sentiment or contradicting it.
    filler = state.get("filler_phrase", "").strip()
    if filler:
        system += (
            f"\n\nYou already said \"{filler}\" to the patient just now. "
            f"Your response must flow naturally from that — don't repeat it, "
            f"don't start with the same word or sentiment, and don't contradict it. "
            f"Continue as if \"{filler}\" was the opening of your sentence."
        )

    # Last 3 exchange pairs (6 messages), each capped at 120 chars to stay lean.
    # More history helps the node understand the full conversation context.
    recent = state["history"][-6:]
    history_lines = []
    for m in recent:
        role = "Patient" if m["role"] == "user" else "Agent"
        content = m["content"]
        if len(content) > 120:
            content = content[:117] + "..."
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)
    user_msg = (
        f"{history_text}\nPatient: {state['transcript']}"
        if history_text
        else f"Patient: {state['transcript']}"
    )

    # Safe defaults before we try the LLM
    speech     = ""
    action     = None
    action_data = {}
    next_state = _FALLBACK_NEXT[node_key]

    try:
        raw    = await _call_llm(system, user_msg, state["api_key"], state["model"], state["backend"], max_tokens=256)
        parsed = _parse_json(raw)

        speech = parsed.get("speech", "").strip()

        action = parsed.get("action") or None
        if isinstance(action, str) and action.lower() in ("null", "none", ""):
            action = None

        action_data = parsed.get("action_data") or {}

        proposed = parsed.get("next_state", "")
        if proposed and proposed != "ended":
            next_state = proposed
        elif proposed == "ended":
            next_state = "ended"

    except Exception as e:
        logger.error(f"[NODE:{node_key}] LLM/parse failed: {e}")

    # Use fallback speech if the model returned nothing
    if not speech:
        fallback_fn = _FALLBACK_SPEECH.get(node_key)
        speech = fallback_fn(state) if callable(fallback_fn) else "Could you repeat that, please?"
        logger.warning(f"[NODE:{node_key}] Using fallback speech")

    logger.info(f"[NODE:{node_key}] speech={speech[:70]!r}  next_state={next_state}")
    return {
        **state,
        "speech":      speech,
        "action":      action,
        "action_data": action_data,
        "next_state":  next_state,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Individual node callables (LangGraph needs named functions)
# ──────────────────────────────────────────────────────────────────────────────

async def node_greeting(s: AgentState)            -> AgentState: return await _run_node(s, "greeting")
async def node_availability(s: AgentState)        -> AgentState: return await _run_node(s, "availability_check")
async def node_appointment_review(s: AgentState)  -> AgentState: return await _run_node(s, "appointment_review")
async def node_confirm(s: AgentState)             -> AgentState: return await _run_node(s, "confirm")
async def node_reschedule(s: AgentState)          -> AgentState: return await _run_node(s, "reschedule")
async def node_cancel(s: AgentState)              -> AgentState: return await _run_node(s, "cancel")
async def node_general(s: AgentState)             -> AgentState: return await _run_node(s, "general_question")
async def node_wrap_up(s: AgentState)             -> AgentState: return await _run_node(s, "wrap_up")
async def node_callback(s: AgentState)            -> AgentState: return await _run_node(s, "callback_request")


# ──────────────────────────────────────────────────────────────────────────────
# Routing map
# ──────────────────────────────────────────────────────────────────────────────

_NODE_MAP: dict[str, str] = {
    "greeting":           "greeting",
    "availability_check": "availability_check",
    "appointment_review": "appointment_review",
    "confirm":            "confirm",
    "reschedule":         "reschedule",
    "cancel":             "cancel",
    "general_question":   "general_question",
    "wrap_up":            "wrap_up",
    "callback_request":   "callback_request",
}


def route_to_node(state: AgentState) -> str:
    return _NODE_MAP.get(state["node"], "appointment_review")


# ──────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("supervisor",          supervisor_node)
    g.add_node("greeting",            node_greeting)
    g.add_node("availability_check",  node_availability)
    g.add_node("appointment_review",  node_appointment_review)
    g.add_node("confirm",             node_confirm)
    g.add_node("reschedule",          node_reschedule)
    g.add_node("cancel",              node_cancel)
    g.add_node("general_question",    node_general)
    g.add_node("wrap_up",             node_wrap_up)
    g.add_node("callback_request",    node_callback)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_to_node, _NODE_MAP)

    for node_name in _NODE_MAP:
        g.add_edge(node_name, END)

    return g.compile()


_graph = _build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Public API — split pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def decide_node(
    transcript: str,
    call_state: str,
    history: list,
    api_key: str,
    model: str,
    backend: str = "groq",
) -> str:
    """
    Run ONLY the supervisor LLM to decide which node to execute.
    Returns the node name string (~300ms). Used to run filler + node in parallel.
    """
    # Include recent conversation so the supervisor can disambiguate intent
    # (e.g. "yes" means something different after "is this John?" vs "does that still work?")
    history_text = _format_history(history, max_pairs=3)
    ctx = f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    user_msg = (
        f"{ctx}"
        f"Current state: {call_state}\n"
        f"Patient just said: \"{transcript}\""
    )
    node = call_state  # safe default
    try:
        raw    = await _call_llm(SUPERVISOR_SYSTEM, user_msg, api_key, model, backend, max_tokens=50)
        parsed = _parse_json(raw)
        choice = parsed.get("node", "")
        if choice in _VALID_NODES:
            node = choice
        else:
            logger.warning(f"[SUPERVISOR] Unknown node '{choice}' — keeping {node}")
    except Exception as e:
        logger.warning(f"[SUPERVISOR] Failed: {e} — defaulting to {node}")

    # Hard guard: never regress to greeting once the call has moved past it.
    # The LLM sometimes mis-routes "Hello?" or "Hi" mid-call back to greeting,
    # which causes the agent to re-introduce herself. Block it in code.
    if node == "greeting" and call_state != "greeting":
        logger.warning(f"[SUPERVISOR] Blocked regression to greeting from {call_state} — routing to {call_state}")
        node = call_state

    node = guard_node_choice(call_state, node, transcript)

    logger.info(f"[SUPERVISOR] state={call_state} → node={node}")
    return node


def build_agent_state(
    transcript: str,
    patient_context: dict,
    call_state: str,
    history: list,
    api_key: str,
    model: str,
    backend: str = "groq",
    previous_node: str = "",
    filler_phrase: str = "",
) -> AgentState:
    """Build an AgentState dict from patient_context for direct node execution."""
    appointment = patient_context.get("appointment", "your appointment")
    date_s = appointment.split(" at ")[0] if " at " in appointment else appointment
    time_s = appointment.split(" at ")[1] if " at " in appointment else ""
    return {
        "transcript":       transcript,
        "call_state":       call_state,
        "previous_node":    previous_node,
        "filler_phrase":    filler_phrase,
        "patient_name":     patient_context.get("name", "there"),
        "appointment_date": date_s,
        "appointment_time": time_s,
        "provider_name":    patient_context.get("provider_name", "your doctor"),
        "agent_name":       patient_context.get("agent_name", "Jamie"),
        "history":          history,
        "api_key":          api_key,
        "model":            model,
        "backend":          backend,
        "node":             call_state,
        "speech":           "",
        "action":           None,
        "action_data":      {},
        "next_state":       call_state,
    }


async def run_node_for_state(
    node_key: str,
    state: AgentState,
) -> tuple[str, Optional[str], dict, str]:
    """
    Run a specific node directly without supervisor or LangGraph.
    Used in the split pipeline after decide_node() has returned.
    Returns (speech, action, action_data, next_state).
    """
    result = await _run_node(state, node_key)
    return (
        result["speech"],
        result["action"],
        result["action_data"],
        result["next_state"],
    )


async def stream_node_speech_for_state(
    node_key: str,
    state: AgentState,
) -> AsyncIterator[dict]:
    """
    Stream a node response as sentence-sized speech chunks.

    Groq is streamed token by token and flushed on sentence boundaries so TTS can
    begin before the full LLM answer is complete. Other backends fall back to the
    normal node call but keep the same event shape.
    """
    started = time.perf_counter()
    action, action_data, next_state = _node_defaults(node_key, state)

    if state["backend"].lower() != "groq":
        result = await _run_node(state, node_key)
        speech = result["speech"].strip()
        if speech:
            yield {"type": "sentence", "text": speech}
        yield {
            "type": "done",
            "speech": speech,
            "action": result["action"],
            "action_data": result["action_data"],
            "next_state": result["next_state"],
            "ttft_ms": None,
            "llm_ms": round((time.perf_counter() - started) * 1000),
        }
        return

    system, user_msg = _build_node_messages(state, node_key, json_mode=False)
    buffer = ""
    parts: list[str] = []
    first_token_at: Optional[float] = None

    try:
        async for token in _groq_stream_text(
            system,
            user_msg,
            state["api_key"],
            state["model"],
            max_tokens=180,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
                yield {
                    "type": "first_token",
                    "ttft_ms": round((first_token_at - started) * 1000),
                }

            buffer += token
            sentences, buffer = _extract_complete_sentences(buffer)
            for sentence in sentences:
                parts.append(sentence)
                yield {"type": "sentence", "text": sentence}

        tail = buffer.strip()
        if tail:
            parts.append(tail)
            yield {"type": "sentence", "text": tail}

        speech = " ".join(parts).strip()
        if not speech:
            fallback_fn = _FALLBACK_SPEECH.get(node_key)
            speech = fallback_fn(state) if callable(fallback_fn) else "Could you repeat that, please?"
            yield {"type": "sentence", "text": speech}

        next_state = guard_next_state(
            state["call_state"], node_key, next_state, state["transcript"]
        )
        yield {
            "type": "done",
            "speech": speech,
            "action": action,
            "action_data": action_data,
            "next_state": next_state,
            "ttft_ms": round((first_token_at - started) * 1000) if first_token_at else None,
            "llm_ms": round((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        logger.warning(f"[NODE:{node_key}] Stream failed, falling back to full node call: {exc}")
        result = await _run_node(state, node_key)
        speech = result["speech"].strip()
        if speech:
            yield {"type": "sentence", "text": speech}
        yield {
            "type": "done",
            "speech": speech,
            "action": result["action"],
            "action_data": result["action_data"],
            "next_state": result["next_state"],
            "ttft_ms": None,
            "llm_ms": round((time.perf_counter() - started) * 1000),
        }


async def run_supervisor(
    transcript:       str,
    patient_context:  dict,
    call_state:       str,
    history:          list,
    api_key:          str,
    model:            str = "gemini-2.5-flash-lite",
    backend:          str = "groq",
    previous_node:    str = "",
) -> tuple[str, Optional[str], dict, str]:
    """
    Run one conversation turn through the supervisor + node graph.
    Returns (speech, action, action_data, next_state).
    """
    appointment = patient_context.get("appointment", "your appointment")
    date_s = appointment.split(" at ")[0] if " at " in appointment else appointment
    time_s = appointment.split(" at ")[1] if " at " in appointment else ""

    initial: AgentState = {
        "transcript":       transcript,
        "call_state":       call_state,
        "previous_node":    previous_node,
        "patient_name":     patient_context.get("name", "there"),
        "appointment_date": date_s,
        "appointment_time": time_s,
        "provider_name":    patient_context.get("provider_name", "your doctor"),
        "agent_name":       patient_context.get("agent_name", "Jamie"),
        "history":          history,
        "api_key":          api_key,
        "model":            model,
        "backend":          backend,
        # outputs — filled by supervisor + chosen node
        "node":        call_state,
        "speech":      "",
        "action":      None,
        "action_data": {},
        "next_state":  call_state,
    }

    result = await _graph.ainvoke(initial)
    return (
        result["speech"],
        result["action"],
        result["action_data"],
        result["next_state"],
    )
