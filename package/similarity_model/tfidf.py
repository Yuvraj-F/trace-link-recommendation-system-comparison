"""
TF-IDF vector space model retriever (lexical baseline).

Fits a TfidfVectorizer on the issue corpus at init; commit messages are
queried against the same vocabulary. No external model download.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from db import Issue
from .base import Retriever


class TfidfRetriever(Retriever):
    """
    Classical IR baseline: weighted term vectors and cosine similarity.

    Captures lexical overlap only; related words with different spellings
    (e.g. fix vs repair) are not aligned unless they share tokens.
    """

    def __init__(self, issues: list[Issue]):
        """
        @param issues: Candidate issues to index.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            sublinear_tf=True,
        )
        super().__init__(issues)

    def _encode_documents(self) -> np.ndarray:
        """Fit vectorizer on issue texts and return dense TF-IDF rows."""
        self.vectorizer.fit(self.issue_texts)
        return self.vectorizer.transform(self.issue_texts)

    def _encode_query(self, text: str) -> np.ndarray:
        """@param text: Commit message. @return: TF-IDF vector for the query."""
        return self.vectorizer.transform([text])
    
    def batch_encode_queries(self, queries: list[str]):
        return self.vectorizer.transform(queries)
