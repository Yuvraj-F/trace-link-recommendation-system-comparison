"""
Sentence-BERT (SBERT) contextual embedding retriever.

Fine-tuned sentence embeddings via sentence-transformers. Uses CUDA when
available. Hugging Face weights are cached under project .hf_cache (see config).
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import logging

from db import Issue
from .base import Retriever

logging.set_verbosity_error()

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SbertRetriever(Retriever):
    """
    Contextual sentence embeddings (Siamese fine-tuned BERT family).

    Strong semantic similarity; slower and heavier than lexical/static baselines.
    """

    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL, issues: list[Issue] | None = None):
        """
        @param model_name: Hugging Face sentence-transformers model id.
        @param issues: Candidate issues to encode at init.
        @raises TypeError: If issues is omitted.
        """
        if issues is None:
            raise TypeError("issues is required")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        super().__init__(issues)

    def is_cuda(self) -> bool:
        """@return: True if the model is running on a CUDA device."""
        return self.device == "cuda"

    def _encode_documents(self):
        return self.model.encode(self.issue_texts, convert_to_numpy=True)
  
    def _encode_query(self, text: str):
        return self.model.encode(text, convert_to_numpy=True)

    def batch_encode_queries(self, queries: list[str]):
        """
        Encode multiple commit messages in one forward pass.

        @param queries: List of commit message strings.
        @return: Matrix of shape (len(queries), embedding_dim).
        """
        return self.model.encode(queries, convert_to_numpy=True)
