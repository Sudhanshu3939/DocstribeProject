import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(
    query_vector: np.ndarray,
    docs: List[Dict],
    doc_vectors: np.ndarray,
    top_k: int = 3,
    filter_source: str = None
) -> List[Tuple[Dict, float]]:
    """
    Retrieve top-k most relevant document chunks based on cosine similarity.

    Args:
        query_vector: Embedding for the query (1D numpy array).
        docs: List of document dictionaries containing 'text', 'source', 'meta'.
        doc_vectors: 2D numpy array of all doc embeddings.
        top_k: How many chunks to return.
        filter_source: Optional filter to restrict retrieval to a single source
                       like 'timeline', 'care_plan', etc.

    Returns:
        List of (doc, score) tuples sorted by similarity.
    """

    # Step 1: Apply routing-based filtering
    if filter_source:
        filtered_docs = []
        filtered_vectors = []

        for i, d in enumerate(docs):
            if d["source"] == filter_source:
                filtered_docs.append(d)
                filtered_vectors.append(doc_vectors[i])

        # If filter removed all candidates, fall back to full set
        if filtered_docs:
            docs = filtered_docs
            doc_vectors = np.array(filtered_vectors)

    # Step 2: Cosine similarity (sklearn handles normalization)
    scores = cosine_similarity([query_vector], doc_vectors)[0]  # shape: (N,)

    # Step 3: Top-K indices (descending)
    top_indices = scores.argsort()[-top_k:][::-1]

    # Step 4: Build result list
    results = [
        (docs[i], float(scores[i]))
        for i in top_indices
    ]

    return results
