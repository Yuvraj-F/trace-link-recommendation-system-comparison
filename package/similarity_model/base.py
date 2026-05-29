"""
Shared retrieval contract for issue–commit trace link recommendation.

All retrievers encode issue texts (summary + description) at construction time,
encode commit messages at query time, and rank issues by cosine similarity.
"""

import re
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from db import Issue


def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase word tokens for bag-of-words embedders.

    @param text: Raw commit message or issue text.
    @return: List of alphanumeric tokens.
    """
    return re.findall(r"\w+", text.lower())


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

def rank_by_similarity(
    query_vec,
    doc_matrix,
    issues: list[Issue],
    threshold: float = 0,
) -> list[Issue]:
    """
    Rank issues by cosine similarity between one query vector and document rows.

    @param query_vec: Embedding of the commit message.
    @param doc_matrix: Row matrix of pre-encoded issue embeddings.
    @param issues: Issue objects aligned with doc_matrix rows.
    @param threshold: Minimum similarity to include an issue (default 0).
    @return: Issues sorted by descending similarity.
    """
    query_vec = _to_dense(query_vec)
    doc_matrix = _to_dense(doc_matrix)

    similarities = cosine_similarity(query_vec, doc_matrix)

    if similarities.ndim == 1:
        similarities = similarities.reshape(1, -1)

    results = []
    for row in similarities:
        ranked = sorted(
            [(issues[i], float(row[i])) for i in range(len(issues)) if row[i] >= threshold],
            key=lambda x: x[1],
            reverse=True
        )
        results.append([i for i, _ in ranked])

    return results


class Retriever(ABC):
    """
    Abstract first-stage retriever: commit message in, ranked issues out.

    Subclasses implement document/query encoding. The corpus is fixed at init
    (all Issue objects passed in). Inputs use Issue.to_text() for documents
    and the raw commit message string for queries.
    """

    def __init__(self, issues: list[Issue]):
        """
        @param issues: Candidate issue corpus (typically seoss33.get_issues()).
        """
        self.issues = issues
        self.issue_texts = [issue.to_text() for issue in issues]
        self._encoded_documents = self._encode_documents()

    @abstractmethod
    def _encode_documents(self) -> np.ndarray:
        """
        Pre-encode all issue texts in the corpus.

        @return: Matrix of shape (num_issues, embedding_dim).
        """
        pass

    @abstractmethod
    def _encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single commit message for retrieval.

        @param text: Commit message string.
        @return: Query vector of shape (embedding_dim,).
        """
        pass

    def batch_encode_queries(self, queries: list[str]):
        pass

    def get_relevant_issues(self, commit: str | list[str], threshold: float = 0) -> list[Issue]:
        """
        Return issues ranked by similarity to the commit message. It can accept a list of commits for batched 
        processing as well.

        @param commit: Commit message text (not the Commit object).
        @param threshold: Minimum cosine similarity to retain a candidate.
        @return: Ranked list of issues, highest similarity first.
        """
        if isinstance(commit, str):
            query_vec = self._encode_query(commit)
        else:
            query_vec = self.batch_encode_queries(commit)

        return rank_by_similarity(
            query_vec, self._encoded_documents, self.issues, threshold
        )
