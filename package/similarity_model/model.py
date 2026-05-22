import torch
from sentence_transformers import SentenceTransformer
from transformers import logging
from query import *
from db import *

logging.set_verbosity_error()

class SimModel:
    """
    This is a wrapper class that abstracts away loading models and pre-computing encodings. 
    NOTE: Current implementation assumes model exists and is publicly available on HuggingFace. If other models
    are needed, this implementation would need to be updated
    """
    def __init__(self, model, issues: list[Issue]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        self.documnets = issues
        self.encoded_documents = self.model.encode_document([issue.to_text() for issue in issues])

    def is_cuda(self):
        return self.device == 'cuda'
    
    def batch_encode_queries(self, queries):
        return self.model.encode_query(queries)

    def get_relevant_issues(self, commit: str, threshold=0) -> list[Issue]:
        """"
        commit: A commit containing the commit message
        threshold: The minimum required similarity in the range [0, 1]  
        
        result: A list of issues sorted by similarity. 
        """
        query = self.model.encode_query(commit)

        similarities = self.model.similarity(query, self.encoded_documents)[0]

        result = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                result.append((self.documnets[i], similarity))

        result.sort(key=lambda x: x[1], reverse=True)

        return [issue for issue, _ in result]