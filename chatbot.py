#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import build_corpus
from rag.embedder import embed_texts, get_model
from rag.retriever import retrieve
from rag.router import route_question
from rag.answer_generator import generate_answer


def load_patient(patient_id: str):
    pfile = Path("patients") / f"{patient_id}.json"
    if not pfile.exists():
        raise FileNotFoundError(f"Patient file not found: {pfile}")
    return json.loads(pfile.read_text(encoding="utf-8"))


def prepare_patient(patient: dict):
    pid = patient["id"]

    docs = build_corpus(patient)
    texts = [d["text"] for d in docs]
    vectors = embed_texts(texts, pid)

    return docs, vectors


def interactive_chat(patient_id: str):
    patient = load_patient(patient_id)
    docs, doc_vectors = prepare_patient(patient)
    embedder = get_model()

    print(f"\nLoaded patient {patient_id}: {patient.get('name')}")
    print(f"Number of chunks: {len(docs)}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Q> ").strip()
        if query.lower() in ("exit", "quit"):
            print("bye")
            break
        print(f"query: {query}")

        route = route_question(query)
        print(f"Route: {route}")
        query_vector = embedder.encode([query])[0]

        retrieved = retrieve(
            query_vector=query_vector,
            docs=docs,
            doc_vectors=doc_vectors,
            top_k=3,
            filter_source=route
        )
        print(f"Retrieved: {retrieved}")
        print("\n")
        answer = generate_answer(
            question=query,
            retrieved_chunks=retrieved,
            route=route,
            patient_meta=patient
        )

        print("\n--- Retrieved Context ---")
        for doc, score in retrieved:
            print(f"[{doc['source']}] score={score:.4f} | {doc['text']}")

        print("\n--- Answer ---")
        print(answer)
        print("\n-------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True,
                        help="Patient ID (P001, P002...)")
    args = parser.parse_args()

    pid = args.patient.upper()
    interactive_chat(pid)


if __name__ == "__main__":
    main()
