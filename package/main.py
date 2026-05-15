from pathlib import Path
from llm import example_llm
from similarity_model import example_model
from config import *
from utils import *
from db import *
from query import *

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

    # Gets issues and commits that have trace link
    trace_links, keys = seoss33.query(TRACE_LINKS_QUERY)
    print(f"Trace link keys: {keys}\n")

    # Extract issue ids or all issues that are linked to merge commits
    count = 0
    issue_ids = []
    for link in trace_links:
        if link.is_merge:
            count += 1
            issue_ids.append(link.issue_id)

    # Get all trace links for issues that were linked to a merge commit
    placeholders = ",".join(["?"] * len(issue_ids))
    issues, keys = seoss33.query(f"select * from change_set_link where issue_id in ({placeholders})", params=issue_ids)
    print(f"Issue keys: {keys}")
    
    # Count trace links for issues that were linked to a merge commit
    issue_counts = {}
    for issue in issues:
        issue_counts[issue.issue_id] = issue_counts.get(issue.issue_id, 0) + 1
    print(f"Commit count per issue: {issue_counts}")

    # Count how many trace links for these issues were merge commits
    issue_commits = {}
    for link in trace_links:
        if link.is_merge:
            issue_commits[link.issue_id] = issue_commits.get(link.issue_id, 0) + 1
    print(f"Merge commit count per issue: {issue_commits}")

    print("The issues that have only one linked commit and have one merge commit need to be investigated to see how that merge commit resolved the issue and wether they should be included in the experiment")

    seoss33.close()