"""Text helpers for the browser voice demo."""

import re


def clean_text_for_tts(text: str) -> str:
    """Normalize text before sending it to a TTS provider."""
    cleaned = re.sub(r"\([^)]*\)", "", text)
    cleaned = re.sub(r"\s*[-]\s*", ", ", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\bDr\.", "Doctor", cleaned)
    cleaned = re.sub(r"\bMr\.", "Mister", cleaned)
    cleaned = re.sub(r"\bMs\.", "Miss", cleaned)
    cleaned = re.sub(r"\bMrs\.", "Missus", cleaned)
    cleaned = re.sub(r"\bSt\.", "Street", cleaned)
    cleaned = re.sub(r"\bApt\.\s*(\d)", r"Apartment \1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
