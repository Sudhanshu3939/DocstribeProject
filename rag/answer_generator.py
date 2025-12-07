import os
from typing import List, Tuple, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
# ----- HF CONFIG -----
HF_API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
MODEL = os.environ.get("HF_MODEL", "openai/gpt-oss-20b:groq")

client = InferenceClient(api_key=HF_API_KEY)

# ----- SIMPLE CACHE -----
LLM_CACHE = {}   # key: (patient_id, question_text)


# ----- PROMPT BUILDER -----
def build_prompt(question: str, retrieved_chunks: List[Tuple[Dict, float]]):
    context = "\n".join(
        f"- [{doc['source']}] {doc['text']}"
        for doc, _ in retrieved_chunks
    ) or "- No context available."

    system_prompt = (
        "You are a clinical assistant. Use ONLY the patient context provided. "
        "Return factual, concise answers (1–3 sentences). "
        "If the answer is not explicitly in the context, reply: "
        "'Information not found in patient record.'"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]


# ----- LLM CALL -----
def call_llm(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.2,
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"LLM Error: {e}"


# ----- MAIN FUNCTION -----
def generate_answer(question: str, retrieved_chunks, route: str, patient_meta: Dict) -> str:
    cache_key = (patient_meta.get("id"), question.strip().lower())

    # CACHE HIT
    if cache_key in LLM_CACHE:
        return LLM_CACHE[cache_key]

    # BUILD PROMPT
    messages = build_prompt(question, retrieved_chunks)

    # CALL LLM
    answer = call_llm(messages)

    # STORE IN CACHE
    LLM_CACHE[cache_key] = answer

    return answer
