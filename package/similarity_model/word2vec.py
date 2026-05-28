"""
Word2Vec / GloVe static embedding retriever.

Document vectors are the mean of in-vocabulary word vectors. OOV tokens are
skipped, which can yield a zero query vector and failed retrieval for some commits.
"""

import numpy as np

from db import Issue
from .base import Retriever, tokenize

# Default Gensim pretrained model (~128 MB, cached under ~/gensim-data).
DEFAULT_WORD2VEC_MODEL = "glove-wiki-gigaword-100"


def _mean_word_vector(tokens: list[str], kv, dim: int) -> np.ndarray:
    """
    @param tokens: Token list from tokenize().
    @param kv: Gensim KeyedVectors.
    @param dim: Vector dimensionality.
    @return: Mean word vector, or zeros if no in-vocab tokens.
    """
    vectors = [kv[word] for word in tokens if word in kv]
    if not vectors:
        return np.zeros(dim, dtype=np.float64)
    return np.mean(vectors, axis=0)


def _train_word2vec(texts: list[str]):
    """
    Train a small Word2Vec model on the issue corpus (offline fallback).

    @param texts: Issue.to_text() strings.
    @return: KeyedVectors from the trained model.
    """
    from gensim.models import Word2Vec

    sentences = [tokenize(text) for text in texts]
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10,
        seed=42,
    )
    return model.wv


class Word2VecRetriever(Retriever):
    """
    Static distributional embeddings via pretrained GloVe or corpus-trained Word2Vec.

    Default: glove-wiki-gigaword-100 via gensim.downloader. On load failure,
    falls back to training on the issue texts passed at construction.
    """

    def __init__(
        self,
        issues: list[Issue],
        model_name: str = DEFAULT_WORD2VEC_MODEL,
        train_on_corpus: bool = False,
    ):
        """
        @param issues: Candidate issues to index.
        @param model_name: Gensim downloader model name.
        @param train_on_corpus: If True, skip download and train on issue texts only.
        """
        self.kv = self._load_vectors(model_name, issues, train_on_corpus)
        self.dim = self.kv.vector_size
        super().__init__(issues)

    def _load_vectors(self, model_name: str, issues: list[Issue], train_on_corpus: bool):
        """@return: KeyedVectors for encoding."""
        if train_on_corpus:
            print("Training Word2Vec on issue corpus...")
            return _train_word2vec([issue.to_text() for issue in issues])

        try:
            import gensim.downloader as api

            print(f"Loading Word2Vec/GloVe vectors: {model_name} (first run may download)...")
            print(f"Gensim data directory: {api.BASE_DIR}")
            return api.load(model_name)
        except Exception as exc:
            print(
                f"Could not load pretrained model '{model_name}' ({exc}). "
                "Falling back to corpus-trained Word2Vec."
            )
            return _train_word2vec([issue.to_text() for issue in issues])

    def _encode_documents(self) -> np.ndarray:
        return np.vstack(
            [_mean_word_vector(tokenize(text), self.kv, self.dim) for text in self.issue_texts]
        )

    def _encode_query(self, text: str) -> np.ndarray:
        return _mean_word_vector(tokenize(text), self.kv, self.dim)
