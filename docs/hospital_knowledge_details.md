# Hospital Knowledge Base Details

This demo uses a local hospital knowledge base for grounded appointment-call answers.

The knowledge base is stored in:

```text
data/hospital_kb.json
```

It contains:

- Clinic name, address, phone, hours, parking, arrival guidance, reschedule policy, cancellation policy, and emergency guidance.
- Doctor full names, display names, departments, specializations, available days, available windows, bios, and appointment notes.
- Department descriptions and doctor mappings.

## Current Demo Facts

Clinic:

```text
City Clinic
125 Market Street, Suite 400, Springfield
Monday-Friday: 8 AM to 6 PM
Saturday: 9 AM to 1 PM for limited appointments
Sunday: Closed
```

Doctors:

| Display Name | Full Name | Specialization | Available Days |
| --- | --- | --- | --- |
| Dr. Smith | Dr. Michael Smith | Cardiology | Monday, Wednesday, Friday |
| Dr. Patel | Dr. Anika Patel | Family Medicine | Tuesday, Thursday |
| Dr. Nguyen | Dr. Laura Nguyen | Dermatology | Monday, Thursday |
| Dr. Brown | Dr. Emily Brown | Orthopedics | Wednesday, Friday |

## RAG Design

The project uses a lightweight cache-first RAG service rather than a remote vector database.

Flow:

```text
user question
  -> local fast-path rules
  -> hospital RAG cache lookup
  -> lightweight KB search
  -> grounded answer
  -> background prefetch for likely follow-up topics
```

This follows the idea from the VoiceAgentRAG paper in a small-project form:

- `Fast Talker`: answer from local rules or RAG cache first.
- `Slow Thinker`: prefetch likely follow-up facts in the background after each user turn.
- Cache-first retrieval: repeated doctor, clinic, and availability questions avoid repeated lookup work.

This demo does not use FAISS/Qdrant yet because the hospital KB is small. If the KB grows, the same `utils/rag_service.py` API can be backed by embeddings and a vector index later.
