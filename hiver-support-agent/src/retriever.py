"""
Step 6-7: Historical Retrieval.

Default backend is TF-IDF + cosine similarity - it's fast, has zero
external dependencies or network calls, and is genuinely a form of
"embeddings + similarity search" as the assignment asks for.

If you have internet access and want stronger semantic matching (catches
paraphrases TF-IDF misses, e.g. "my parcel is late" vs "delivery delayed"),
set backend="sentence-transformer" - it's a drop-in swap, same interface.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:

    def __init__(self, backend: str = "tfidf", model_name: str = "all-MiniLM-L6-v2"):
        self.backend = backend
        self.model_name = model_name
        self._vectorizer = None
        self._encoder = None
        self.data = None
        self.embeddings = None

    def fit(self, data: pd.DataFrame, text_col: str = "customer_text"):
        """Build the retrieval index over `data[text_col]`."""
        self.data = data.reset_index(drop=True)
        texts = self.data[text_col].tolist()

        if self.backend == "tfidf":
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.embeddings = self._vectorizer.fit_transform(texts)

        elif self.backend == "sentence-transformer":
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._encoder = SentenceTransformer(self.model_name)
            self.embeddings = self._encoder.encode(
                texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
        else:
            raise ValueError(f"Unknown retriever backend: {self.backend}")

        return self

    def _encode_query(self, query: str):
        if self.backend == "tfidf":
            return self._vectorizer.transform([query])
        else:
            return self._encoder.encode([query], normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 5):
        query_vec = self._encode_query(query)
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                "score": float(scores[i]),
                "customer": self.data.iloc[i]["customer_text"],
                "response": self.data.iloc[i]["response_text"],
            })
        return results

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Retriever":
        return joblib.load(path)


def build_retrieval_index(pairs: pd.DataFrame, out_path: str, max_size: int = 20000,
                           backend: str = "tfidf", seed: int = 42) -> Retriever:
    """Step 6: build and persist the retrieval index."""

    print("\n" + "=" * 60)
    print("BUILDING HISTORICAL RETRIEVAL INDEX")
    print("=" * 60)

    data = pairs.copy()
    if len(data) > max_size:
        data = data.sample(max_size, random_state=seed)

    retriever = Retriever(backend=backend).fit(data)
    retriever.save(out_path)

    print(f"\u2713 Retrieval index built over {len(data)} conversation pairs -> {out_path}")
    return retriever
