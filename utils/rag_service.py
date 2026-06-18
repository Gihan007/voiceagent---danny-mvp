import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class RAGAnswer:
    speech: str
    node: str
    metadata: dict[str, Any]


class HospitalRAGService:
    """Small cache-first RAG layer for the local hospital demo knowledge base."""

    def __init__(self, kb_path: Union[str, Path] = "data/hospital_kb.json", ttl_seconds: int = 300) -> None:
        self.kb_path = Path(kb_path)
        self.ttl_seconds = ttl_seconds
        self._kb: dict[str, Any] = {}
        self._docs: list[dict[str, Any]] = []
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()
        self.load()

    def load(self) -> None:
        self._kb = json.loads(self.kb_path.read_text(encoding="utf-8"))  #data/hospital_kb.json
        self._docs = self._build_docs(self._kb)
        self._cache.clear()

    @property
    def document_count(self) -> int:
        return len(self._docs)

    @property
    def cache_size(self) -> int:
        self._evict_expired()
        return len(self._cache)

    def answer(self, query: str, patient_context: Optional[dict[str, Any]] = None) -> Optional[RAGAnswer]:
        text = self._normalize(query)
        if not text:
            return None

        patient_context = patient_context or {}
        provider = patient_context.get("provider_name", "Dr. Smith")
        doctor = self._doctor_for_query(text, provider)
        clinic = self._kb.get("clinic", {})

        if self._asks_doctor_full_name(text) and doctor:
            return RAGAnswer(
                speech=(
                    f"{doctor['display_name']}'s full name is {doctor['full_name']}. "
                    f"{doctor['display_name']} is listed under {doctor['specialization']}."
                ),
                node="rag_doctor_full_name",
                metadata={"doctor": doctor["display_name"], "source": "hospital_kb"},
            )

        specialist_answer = self._answer_specialist_question(text)
        if specialist_answer:
            return specialist_answer

        if self._asks_doctor_specialty(text) and doctor:
            return RAGAnswer(
                speech=(
                    f"{doctor['display_name']} specializes in {doctor['specialization']} "
                    f"in the {doctor['department']} department."
                ),
                node="rag_doctor_specialization",
                metadata={"doctor": doctor["display_name"], "source": "hospital_kb"},
            )

        if self._asks_doctor_details(text) and doctor:
            return RAGAnswer(
                speech=doctor["bio"],
                node="rag_doctor_bio",
                metadata={"doctor": doctor["display_name"], "source": "hospital_kb"},
            )

        if self._asks_available_doctors(text):
            doctors = self._kb.get("doctors", [])
            day = self._day_in_text(text)
            if day:
                doctors = [
                    doctor for doctor in doctors
                    if day in {available.lower() for available in doctor.get("available_days", [])}
                ]
            parts = [
                f"{d['full_name']} for {d['specialization']}"
                for d in doctors
            ]
            if not parts:
                return RAGAnswer(
                    speech="I do not see a matching doctor in the local demo knowledge base for that day.",
                    node="rag_available_doctors",
                    metadata={"source": "hospital_kb"},
                )
            day_phrase = f" for {day.title()}" if day else ""
            return RAGAnswer(
                speech=f"City Clinic lists{day_phrase}: " + "; ".join(parts) + ".",
                node="rag_available_doctors",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_doctor_availability(text) and doctor:
            days = ", ".join(doctor.get("available_days", []))
            windows = ", ".join(doctor.get("available_windows", []))
            return RAGAnswer(
                speech=(
                    f"{doctor['display_name']} is generally listed for {days}, "
                    f"with windows around {windows}. The scheduling desk still needs to confirm the exact slot."
                ),
                node="rag_doctor_availability",
                metadata={"doctor": doctor["display_name"], "source": "hospital_kb"},
            )

        if self._asks_clinic_hours(text):
            hours = clinic.get("hours", {})
            return RAGAnswer(
                speech=(
                    f"{clinic.get('name', 'The clinic')} is open Monday to Friday, "
                    f"{hours.get('monday_friday', 'business hours')}. "
                    f"Saturday hours are {hours.get('saturday', 'limited')}, and Sunday is {hours.get('sunday', 'closed')}."
                ),
                node="rag_clinic_hours",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_clinic_location(text):
            return RAGAnswer(
                speech=(
                    f"{clinic.get('name', 'The clinic')} is at {clinic.get('address')}. "
                    f"{clinic.get('parking')}"
                ),
                node="rag_clinic_location",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_clinic_overview(text):
            return RAGAnswer(
                speech=(
                    f"{clinic.get('name', 'The clinic')} is an outpatient medical clinic. "
                    f"It has departments for Cardiology, Primary Care, Dermatology, and Orthopedics."
                ),
                node="rag_clinic_overview",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_arrival(text):
            return RAGAnswer(
                speech=clinic.get("arrival_guidance", ""),
                node="rag_arrival_guidance",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_policy(text, "reschedule"):
            return RAGAnswer(
                speech=clinic.get("reschedule_policy", ""),
                node="rag_reschedule_policy",
                metadata={"source": "hospital_kb"},
            )

        if self._asks_policy(text, "cancel"):
            return RAGAnswer(
                speech=clinic.get("cancellation_policy", ""),
                node="rag_cancellation_policy",
                metadata={"source": "hospital_kb"},
            )

        self.search(query, top_k=2)
        return None

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        key = self._cache_key(query)
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1][:top_k]

        query_tokens = self._tokens(query)
        scored: list[dict[str, Any]] = []
        for doc in self._docs:
            score = self._score(query_tokens, doc)
            if score > 0:
                scored.append({**doc, "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        result = scored[:top_k]
        self._cache[key] = (now + self.ttl_seconds, result)
        return result

    async def prefetch_for_turn(self, transcript: str, current_state: str, history: list[dict[str, str]]) -> None:
        topics = self._predict_followups(transcript, current_state, history)
        async with self._lock:
            await asyncio.gather(*(asyncio.to_thread(self.search, topic, 3) for topic in topics))

    def _build_docs(self, kb: dict[str, Any]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        clinic = kb.get("clinic", {})
        clinic_text = " ".join(
            str(value)
            for value in [
                clinic.get("name"),
                clinic.get("type"),
                clinic.get("phone"),
                clinic.get("address"),
                clinic.get("arrival_guidance"),
                clinic.get("parking"),
                clinic.get("reschedule_policy"),
                clinic.get("cancellation_policy"),
                clinic.get("emergency_guidance"),
                json.dumps(clinic.get("hours", {})),
            ]
            if value
        )
        docs.append({
            "id": "clinic",
            "kind": "clinic",
            "summary": clinic_text,
            "tokens": self._tokens(clinic_text),
        })

        for doctor in kb.get("doctors", []):
            text = " ".join(
                str(value)
                for value in [
                    doctor.get("display_name"),
                    doctor.get("full_name"),
                    doctor.get("specialization"),
                    doctor.get("department"),
                    " ".join(doctor.get("available_days", [])),
                    " ".join(doctor.get("available_windows", [])),
                    doctor.get("bio"),
                    doctor.get("appointment_notes"),
                ]
                if value
            )
            docs.append({
                "id": doctor["display_name"],
                "kind": "doctor",
                "summary": text,
                "tokens": self._tokens(text),
            })

        for department in kb.get("departments", []):
            text = " ".join(
                str(value)
                for value in [
                    department.get("name"),
                    department.get("description"),
                    " ".join(department.get("doctors", [])),
                ]
                if value
            )
            docs.append({
                "id": department["name"],
                "kind": "department",
                "summary": text,
                "tokens": self._tokens(text),
            })
        return docs

    def _doctor_for_query(self, text: str, provider_name: str) -> Optional[dict[str, Any]]:
        doctors = self._kb.get("doctors", [])
        for doctor in doctors:
            names = {
                self._normalize(doctor.get("display_name", "")),
                self._normalize(doctor.get("full_name", "")),
                self._normalize(doctor.get("full_name", "").replace("Dr.", "")),
            }
            if any(name and name in text for name in names):
                return doctor

        provider_norm = self._normalize(provider_name)
        for doctor in doctors:
            if provider_norm and provider_norm == self._normalize(doctor.get("display_name", "")):
                return doctor
        return doctors[0] if doctors else None

    def _predict_followups(self, transcript: str, current_state: str, history: list[dict[str, str]]) -> list[str]:
        text = self._normalize(transcript)
        topics = [
            "doctor full name specialization availability",
            "clinic hours address parking arrival guidance",
        ]
        if "doctor" in text or "smith" in text or "him" in text:
            topics.extend([
                "Dr. Smith full name",
                "Dr. Smith cardiology specialization",
                "Dr. Smith available days",
            ])
        if current_state == "reschedule" or any(day in text for day in ["monday", "weekend", "weekday"]):
            topics.extend([
                "reschedule policy",
                "available doctors Monday Wednesday Friday",
                "appointment availability backup day",
            ])
        if "where" in text or "parking" in text or "arrive" in text:
            topics.extend(["clinic location parking", "arrival guidance"])
        return list(dict.fromkeys(topics))[:8]

    def _score(self, query_tokens: set[str], doc: dict[str, Any]) -> int:
        tokens = doc["tokens"]
        overlap = query_tokens & tokens
        score = len(overlap)
        if doc["kind"] == "doctor" and {"doctor", "dr", "smith"} & query_tokens:
            score += 2
        if doc["kind"] == "clinic" and {"clinic", "where", "hours", "parking"} & query_tokens:
            score += 2
        return score

    def _tokens(self, text: str) -> set[str]:
        normalized = self._normalize(text)
        synonyms = {
            "speciality": "specialization",
            "specialized": "specialization",
            "specialised": "specialization",
            "specialist": "specialization",
            "specialists": "specialization",
            "availability": "available",
            "doctor": "dr",
            "physician": "doctor",
        }
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        tokens.update(synonyms.get(token, token) for token in list(tokens))
        return {token for token in tokens if len(token) > 1}

    def _normalize(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _cache_key(self, query: str) -> str:
        return " ".join(sorted(self._tokens(query)))

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [key for key, value in self._cache.items() if value[0] <= now]
        for key in expired:
            self._cache.pop(key, None)

    def _asks_doctor_full_name(self, text: str) -> bool:
        return "full name" in text or "first name" in text or "last name" in text

    def _asks_doctor_specialty(self, text: str) -> bool:
        return any(term in text for term in [
            "specialty", "speciality", "specialization", "specializations",
            "specialized", "specialised", "specialist", "specialists",
            "what kind of doctor", "cardiologist", "dermatologist", "orthopedic",
        ])

    def _asks_doctor_details(self, text: str) -> bool:
        return any(term in text for term in [
            "doctor details", "about him", "about her", "more about", "background",
            "experience", "qualification", "bio", "biography",
        ])

    def _asks_doctor_availability(self, text: str) -> bool:
        return ("doctor" in text or "smith" in text or "dr" in text) and any(term in text for term in [
            "available", "availability", "which day", "what day", "schedule",
        ])

    def _asks_available_doctors(self, text: str) -> bool:
        if any(term in text for term in [
            "available doctors", "which doctors", "what doctors", "doctors available",
            "all doctors", "list doctors", "doctors are available",
            "doctor are available", "doctors do you have",
        ]):
            return True
        return bool(re.search(r"\b(doctor|doctors|dr)\b.*\b(available|availability)\b", text)) or bool(
            re.search(r"\b(available|availability)\b.*\b(doctor|doctors|dr)\b", text)
        )

    def _answer_specialist_question(self, text: str) -> Optional[RAGAnswer]:
        specialty_terms = {
            "cardiology": ("cardiology", "cardiologist", "heart"),
            "dermatology": ("dermatology", "dermatologist", "skin"),
            "orthopedics": ("orthopedics", "orthopedic", "bone", "joint"),
            "family medicine": ("family medicine", "primary care", "general doctor"),
        }
        if not any(term in text for term in [
            "have", "available", "doctor", "specialist", "specialists",
            "best", "who is", "which", "recommend",
        ]):
            return None

        departments = {department["name"].lower(): department for department in self._kb.get("departments", [])}
        doctors = self._kb.get("doctors", [])
        for department_name, terms in specialty_terms.items():
            if not any(term in text for term in terms):
                continue

            matched_doctors = [
                doctor for doctor in doctors
                if doctor.get("specialization", "").lower() == department_name
                or department_name in doctor.get("department", "").lower()
            ]
            if not matched_doctors:
                return RAGAnswer(
                    speech=f"I do not see a {department_name} doctor in the local demo knowledge base.",
                    node="rag_specialist_lookup",
                    metadata={"source": "hospital_kb"},
                )

            names = ", ".join(
                f"{doctor['full_name']} for {doctor['specialization']}"
                for doctor in matched_doctors
            )
            department = departments.get(department_name)
            description = f" {department['description']}" if department else ""
            return RAGAnswer(
                speech=f"For {department_name.title()}, City Clinic lists {names}.{description}",
                node="rag_specialist_lookup",
                metadata={"source": "hospital_kb", "specialty": department_name},
            )
        return None

    def _day_in_text(self, text: str) -> Optional[str]:
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            if day in text:
                return day
        return None

    def _asks_clinic_hours(self, text: str) -> bool:
        return "hours" in text or "open" in text or "opening" in text or "closing" in text

    def _asks_clinic_location(self, text: str) -> bool:
        return "address" in text or "location" in text or "where is" in text or "parking" in text

    def _asks_clinic_overview(self, text: str) -> bool:
        return ("clinic" in text or "hospital" in text) and any(term in text for term in [
            "about", "details", "information", "info", "what is",
        ])

    def _asks_arrival(self, text: str) -> bool:
        return "arrive" in text or "early" in text or "lobby" in text or "check in" in text

    def _asks_policy(self, text: str, policy: str) -> bool:
        if policy == "reschedule":
            return "reschedule policy" in text or "change policy" in text
        return "cancel policy" in text or "cancellation policy" in text
