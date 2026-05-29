"""
FastText subword embedding retriever.

Uses pretrained fasttext-wiki-news-subwords-300 by default (~1 GB Gensim cache).
Subword n-grams improve handling of OOV tokens (identifiers, jargon) vs GloVe.
"""

import numpy as np

from db import Issue
from .base import Retriever, tokenize

# Default Gensim pretrained model (large download, cached under ~/gensim-data).
DEFAULT_FASTTEXT_MODEL = "fasttext-wiki-news-subwords-300"


def _mean_fasttext_vector(tokens: list[str], kv, dim: int) -> np.ndarray:
    """
    @param tokens: Token list from tokenize().
    @param kv: FastText KeyedVectors (supports get_mean_vector for OOV).
    @param dim: Vector dimensionality.
    @return: Document vector, or zeros if no tokens.
    """

    if not tokens:
        return np.zeros(dim, dtype=np.float64)
    return np.asarray(kv.get_mean_vector(tokens), dtype=np.float64)

    if not tokens:
        return np.zeros(dim, dtype=np.float64)
    if hasattr(kv, "get_mean_vector"):
        return np.asarray(kv.get_mean_vector(tokens), dtype=np.float64)
    vectors = []
    for word in tokens:
        try:
            vectors.append(kv.get_vector(word))
        except KeyError:
            continue
    if not vectors:
        return np.zeros(dim, dtype=np.float64)
    return np.mean(vectors, axis=0)


def _train_fasttext(texts: list[str]):
    """
    Train FastText on the issue corpus (offline fallback).

    @param texts: Issue.to_text() strings.
    @return: KeyedVectors from the trained model.
    """
    from gensim.models import FastText

    sentences = [tokenize(text) for text in texts]
    model = FastText(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10,
        seed=42,
    )
    return model.wv


class FastTextRetriever(Retriever):
    """
    Subword-aware static embeddings for software-engineering text.

    First run downloads and encodes all issues (slow on CPU). Use
    train_on_corpus=True to avoid the large pretrained download.
    """

    def __init__(
        self,
        issues: list[Issue],
        model_name: str = DEFAULT_FASTTEXT_MODEL,
        train_on_corpus: bool = True,
    ):
        """
        @param issues: Candidate issues to index.
        @param model_name: Gensim downloader FastText model name.
        @param train_on_corpus: If True, train on issue texts instead of downloading.
        """
        self.kv = self._load_vectors(model_name, issues, train_on_corpus)
        self.dim = self.kv.vector_size
        super().__init__(issues)

    def _load_vectors(self, model_name: str, issues: list[Issue], train_on_corpus: bool):
        """@return: KeyedVectors for encoding."""
        if train_on_corpus:
            print("Training FastText on issue corpus...")
            return _train_fasttext([issue.to_text() for issue in issues])

        try:
            import gensim.downloader as api

            print(f"Loading FastText vectors: {model_name} (first run may download)...")
            print(f"Gensim data directory: {api.BASE_DIR}")
            return api.load(model_name)
        except Exception as exc:
            print(
                f"Could not load pretrained model '{model_name}' ({exc}). "
                "Falling back to corpus-trained FastText."
            )
            return _train_fasttext([issue.to_text() for issue in issues])

    def _encode_documents(self) -> np.ndarray:
        """Pre-encode all issues; prints progress because this step is slow."""
        n = len(self.issue_texts)
        print(f"Encoding {n} issues with FastText (may take a few minutes)...")
        matrix = np.vstack(
            [_mean_fasttext_vector(tokenize(text), self.kv, self.dim) for text in self.issue_texts]
        )
        print("Issue encoding complete.")

        return matrix

    def _encode_query(self, text: str) -> np.ndarray:
        return _mean_fasttext_vector(tokenize(text), self.kv, self.dim)
    
    def batch_encode_queries(self, queries):
        matrix = np.vstack(
            [_mean_fasttext_vector(tokenize(text), self.kv, self.dim) for text in self.issue_texts]
        )
        return matrix
