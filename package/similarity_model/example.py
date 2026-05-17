"""
This is a placeholder to assist development of the high level pipeline. All similarity models should align with this.
"""
from db import SEOSS33
from query import *

def example_model(commit, db: SEOSS33):
    """"
    commit: A commit containing the commit message
    db: Database isntance used to access issues

    result: A list of issues
    """
    issues, keys = db.query(SELECT_ALL_ISSUES)   
    print(f"Issue keys: {keys}\n")
    
    result = ["issue#1", "issue#2", "issue#3"]
    return result