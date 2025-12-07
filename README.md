# Patient-Specific RAG Chatbot (Terminal-Based)

This project implements a terminal-based Retrieval-Augmented Generation (RAG) chatbot that answers patient-specific questions using:

- Custom retrieval over structured patient JSON data  
- Chunking via LangChain text splitters  
- Embedding-based vector search  
- Rule-based routing logic  
- LLM answer generation using Hugging Face Inference Providers 


## Features

- Multiple patient records (JSON files)  
- Chunking of care plans, notes, timeline events, summaries  
- Local embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)  
- Cosine similarity–based retrieval (`scikit-learn`)  
- Rule-based intent router (summary, care_plan, timeline, notes, or all)  
- LLM answer generation using `openai/gpt-oss-20b:groq` via Hugging Face InferenceClient  
- Caching of LLM responses to avoid repeated API calls  
- Clean modular structure, easy to extend

## Architecture

```
patients/*.json
        ↓
rag/chunker.py        (LangChain RecursiveCharacterTextSplitter)
        ↓
rag/embedder.py       (SentenceTransformer embeddings + caching)
        ↓
rag/retriever.py      (cosine similarity retrieval)
        ↓
rag/router.py         (intent detection → summary/care_plan/timeline/notes/all)
        ↓
rag/answer_generator.py (HF chat.completions → openai/gpt-oss-20b:groq + cache)
        ↓
chatbot.py            (terminal interface)
```

## Design Decisions

### 1. Chunking With LangChain Splitter  
Chunk formation affects retrieval quality. The project uses `RecursiveCharacterTextSplitter` to preserve sentence boundaries, reduce noise, handle edge cases, and simplify code. Only this utility from LangChain is used.

### 2. Manual Embedding + Retrieval  
Embedding and retrieval are implemented explicitly (SentenceTransformers + cosine similarity). This demonstrates an understanding of embedding pipelines and vector search rather than hiding mechanics behind high-level wrappers.

### 3. Rule-Based Router  
A priority-based router classifies user intent into `summary`, `care_plan`, `timeline`, `notes`, or `all`. This increases retrieval precision and ensures the LLM sees focused, relevant context.

### 4. LLM Answer Generation (Hugging Face Inference Providers)  
The system uses Hugging Face’s OpenAI-compatible chat completion interface for `openai/gpt-oss-20b:groq`.

LLM is used for final answer composition; retrieval remains deterministic.

### 5. Caching  
LLM outputs for identical (patient_id, question) pairs are cached in memory to reduce API usage and speed repeated queries during evaluation.

## Project Structure

```
.
├── chatbot.py
├── requirements.txt
├── patients/
│   ├── P001.json
│   ├── P002.json
│   └── P003.json
└── rag/
    ├── __init__.py
    ├── chunker.py
    ├── embedder.py
    ├── retriever.py
    ├── router.py
    └── answer_generator.py
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Export your Hugging Face token:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

3. Run the chatbot:

```bash
python chatbot.py --patient P001
```

## Example Questions

```
Summarize the patient’s conditions.
When is the next review?
What happened recently?
What medications are prescribed?
What is the care plan?
```

## Example Output

### Query:
```
When is the next review?
```

### Retrieved Context:
```
[care_plan] ... Next review scheduled on 2025-12-10 ...
```

### LLM Answer:
```
The next review is scheduled for 2025-12-10 according to the care plan.
```


## Future Improvements

- Add Chroma or FAISS for scalable vector retrieval  
- Deploy via FastAPI  
- Add timeline-aware retriever (explicit date sorting)  
- Add LangGraph for multi-step agent workflows  
- Add evaluation metrics (precision@k, grounding accuracy)
