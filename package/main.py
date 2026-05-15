from pathlib import Path
from llm import example_llm
from similarity_model import example_model
from config import *
from utils import *
from db import *

def setup_database():
    "TODO: create database connection"

def get_candidate_issues(model, commit):
    "TODO: pass commit and database connection to model and return the result"

def compute_recall_k(commit, issues, k):
    "TODO: fetch trace link for commit"
    "TODO: check if issues contains the ground truth issue based on the trace link"
    "TODO: compute and return recall value"

def get_ranked_issues(model, commit, issues):
    ranked_issues = model(commit, issues)
    return ranked_issues

def compute_precision_k(commit, issues, k):
    "TODO: fetch trace link for commit"
    "TODO: check if issues[:k] contains the ground truth issue based on the trace link"
    "TODO: compute and return recall value"

if __name__ == "__main__":
    DB_ROOT, DB_ZIP_PATH = init_config()
    SEOSS_PATH = DB_ROOT / "seoss33"
    
    if not Path.exists(SEOSS_PATH):
        unzip(DB_ZIP_PATH, SEOSS_PATH)

    bz2_files = list(SEOSS_PATH.glob("*.bz2"))
    for file in bz2_files:
        decompress_bz2(file)

    sqlite_files = list(SEOSS_PATH.glob("*.sqlite3"))
    print(f"Found SEOSS 33 dataset with {len(sqlite_files)} project databases")

    seoss33 = SEOSS33(sqlite_files[0])
    for row in seoss33.definition():
        print(row)

    for row in seoss33.get_issues().fetchone():
        print(row)
    seoss33.close()


    
    

