"""
Factory for constructing retrieval backends by name.

Supported names: tfidf, word2vec, fasttext, sbert.
"""

from db import Issue

from .base import Retriever
from .fasttext_retriever import DEFAULT_FASTTEXT_MODEL, FastTextRetriever
from .sbert import DEFAULT_SBERT_MODEL, SbertRetriever
from .tfidf import TfidfRetriever
from .word2vec import DEFAULT_WORD2VEC_MODEL, Word2VecRetriever

# Names evaluated in package/main.py when run as __main__.
RETRIEVER_NAMES = ("tfidf", "word2vec", "fasttext", "sbert")


def build_retriever(name: str, issues: list[Issue], **kwargs) -> Retriever:
    """
    Instantiate a retriever implementation for comparison experiments.

    @param name: One of RETRIEVER_NAMES (case-insensitive).
    @param issues: Issue corpus to index and rank over.
    @param kwargs: Optional overrides:
        - model (str): Pretrained model id/path for word2vec, fasttext, or sbert.
        - train_on_corpus (bool): If True, train word2vec/fasttext on issue texts
          instead of downloading pretrained vectors.
    @return: Configured Retriever instance.
    @raises ValueError: If name is not recognized.
    """
    key = name.lower()
    if key == "tfidf":
        return TfidfRetriever(issues)
    if key == "word2vec":
        return Word2VecRetriever(
            issues,
            model_name=kwargs.get("model", DEFAULT_WORD2VEC_MODEL),
            train_on_corpus=kwargs.get("train_on_corpus", False),
        )
    if key == "fasttext":
        return FastTextRetriever(
            issues,
            model_name=kwargs.get("model", DEFAULT_FASTTEXT_MODEL),
            train_on_corpus=kwargs.get("train_on_corpus", True),
        )
    if key == "sbert":
        return SbertRetriever(
            kwargs.get("model", DEFAULT_SBERT_MODEL),
            issues,
        )
    raise ValueError(
        f"Unknown retriever '{name}'. Choose from: {', '.join(RETRIEVER_NAMES)}"
    )
