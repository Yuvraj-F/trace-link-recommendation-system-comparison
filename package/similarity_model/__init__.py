"""
Issue–commit trace link retrieval models.

Four interchangeable first-stage retrievers share the Retriever interface:
  - TfidfRetriever   (lexical VSM)
  - Word2VecRetriever (static GloVe / Word2Vec)
  - FastTextRetriever (subword static embeddings)
  - SbertRetriever   (contextual sentence embeddings)

Use build_retriever(name, issues) to construct by name.
"""

from .base import Retriever, rank_by_similarity, tokenize
from .factory import RETRIEVER_NAMES, build_retriever
from .fasttext_retriever import DEFAULT_FASTTEXT_MODEL, FastTextRetriever
from .sbert import DEFAULT_SBERT_MODEL, SbertRetriever
from .tfidf import TfidfRetriever
from .word2vec import DEFAULT_WORD2VEC_MODEL, Word2VecRetriever

__all__ = [
    "Retriever",
    "rank_by_similarity",
    "tokenize",
    "RETRIEVER_NAMES",
    "build_retriever",
    "TfidfRetriever",
    "Word2VecRetriever",
    "FastTextRetriever",
    "SbertRetriever",
    "DEFAULT_SBERT_MODEL",
    "DEFAULT_WORD2VEC_MODEL",
    "DEFAULT_FASTTEXT_MODEL",
]
