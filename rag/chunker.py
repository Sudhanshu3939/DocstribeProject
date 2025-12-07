from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standard chunking strategy for small RAG systems
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", "?", "!"]
)

def chunk_field(text: str) -> list:
    """
    Splits a single patient field (e.g., care_plan, notes)
    into overlapping chunks using LangChain text splitter.
    """
    if not text:
        return []
    return text_splitter.split_text(text)


def build_corpus(patient: dict) -> list:
    """
    Convert patient dictionary into a list of chunk dictionaries:
    Each dict has:
        - text   : chunk string
        - source : which field (timeline/care_plan/notes/summary)
        - meta   : patient ID, and timeline date if applicable
    """
    docs = []

    # Summary section: combine basic info
    summary_text = (
        f"Name: {patient.get('name')}. "
        f"Age/Sex: {patient.get('age')}/{patient.get('sex')}. "
        f"Conditions: {', '.join(patient.get('conditions', []))}. "
        + (patient.get('notes', '') or '')
    )
    for chunk in chunk_field(summary_text):
        docs.append({
            "text": chunk,
            "source": "summary",
            "meta": {"patient_id": patient.get("id")}
        })

    # Care plan
    for chunk in chunk_field(patient.get("care_plan", "")):
        docs.append({
            "text": chunk,
            "source": "care_plan",
            "meta": {"patient_id": patient.get("id")}
        })

    # Timeline — treat each event as separate text but still chunk it
    for event in patient.get("timeline", []):
        date = event.get("date")
        text = f"{date}: {event.get('text', '')}"
        for chunk in chunk_field(text):
            docs.append({
                "text": chunk,
                "source": "timeline",
                "meta": {"patient_id": patient.get("id"), "date": date}
            })

    # Notes
    for chunk in chunk_field(patient.get("notes", "")):
        docs.append({
            "text": chunk,
            "source": "notes",
            "meta": {"patient_id": patient.get("id")}
        })

    return docs
