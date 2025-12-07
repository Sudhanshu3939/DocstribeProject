from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json

MODEL_NAME = "all-MiniLM-L6-v2"
_cache_dir = Path(".cache")
_cache_dir.mkdir(exist_ok=True)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, patient_id: str):
    cache_e = _cache_dir / f"{patient_id}_embeddings.npy"
    cache_meta = _cache_dir / f"{patient_id}_meta.json"

    if cache_e.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if meta.get("n_texts") == len(texts):
            return np.load(cache_e)

    model = get_model()
    embs = model.encode(texts, convert_to_numpy=True)

    np.save(cache_e, embs)
    cache_meta.write_text(json.dumps({"n_texts": len(texts)}))

    return embs
