import torch
from sentence_transformers import SentenceTransformer
from db import SEOSS33
from query import *

class SimModel:
    """
    This is a wrapper class that abstracts away loading models and fetching required data from the database. 
    NOTE: Current implementation assumes model exists and is publicly available on HuggingFace. If other models
    are needed, this implementation would need to be updated
    """
    def __init__(self, model, db: SEOSS33):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        self.db = db
        self.issues_str, self.issue_embeddings = self._get_issues()
        
    def _get_issues(self):
        issues, keys = self.db.query(SELECT_ALL_ISSUES)   
        # print(f"Issue keys: {keys}\n")
        issues_str = []
        for issue in issues:
            issues_str.append((issue.summary or "") + ". "  + (issue.description or ""))
        print("encoding...")
        embeddings = self.model.encode_document(issues_str)
        print("done")
        return issues_str, embeddings
    
    def similarity(self, commit):
        """"
        commit: A commit containing the commit message
        db: Database isntance used to access issues

        result: A list of issues
        """
        query = self.model.encode_query(commit)

        similarities = self.model.similarity(query, self.issue_embeddings)[0]
        # print(len(similarities))
        result = []
        for i, similarity in enumerate(similarities):
            if similarity > 0.5:
                result.append(self.issues_str[i])

        # for candidate in result:
        #     print(candidate[:100] + "...")
        # print(f"{len(result)} Candidates for query: {commit}")

        return result