from pathlib import Path
from llm import example_llm
from similarity_model import *
from config import *
from utils import *
from db import *
from query import *

def get_candidate_issues(model: SimModel, commit: Commit | str) -> list[Issue]:
    "TODO: batch queries"
    return model.get_relevant_issues(commit.message if isinstance(commit, Commit) else commit)


def compute_recall_k(pred_issues: list[Issue], true_issues: list[Issue], k=None) -> float:
    pred_issues = set(pred_issues[:k]) if k != None else set(pred_issues)
    true_issues = set(true_issues)

    count = 0
    for issue in true_issues:
        if issue in pred_issues:
            count += 1
        
    recall = count/len(true_issues)
    "TODO: Plot recall@k over multiple k values to get a curve showing how recall changes at different k's. Ideally it is just 1 but realistically probably looks like an increasing curve"
    return recall

def recall_curve():
    pred = [1, 2, 34, 36, 865, 3, 4, 5, 34]
    issues = [34, 865]
    for k in range(1, 10):
        print(f"Recall@{k}: {compute_recall_k(pred, issues, k=k)}")

def get_ranked_issues(model, commit, issues):
    ranked_issues = model(commit, issues)
    return ranked_issues

def compute_precision_k(commit, issues, k=5):
    "TODO: fetch trace link for commit"
    "TODO: check if issues[:k] contains the ground truth issue based on the trace link"
    "TODO: compute and return recall value"

def preprocess(seoss33: SEOSS33):
    # Gets issues and commits that have trace link
    trace_links, keys = seoss33.query(SELECT_ALL_TRACE_LINKS)
    print(f"Trace link keys: {keys}\n")

    # Extract issue ids for all issues that are linked to merge commits
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

def commit_counts(seoss33: SEOSS33):
    trace_links, keys = seoss33.query(SELECT_ALL_TRACE_LINKS)
    print(f"Trace link keys: {keys}\n")

    commit_counts = {}
    for link in trace_links:
        commit_counts[link.commit_hash] = commit_counts.get(link.commit_hash, 0) + 1

    for c, count in commit_counts.items():
        if count > 5:
            print(f"Issues per commit: {count} ({c})")

if __name__ == "__main__":
    seoss33 = init_db()

    # Pipeline
    # Load similarity model
    minilm_l6_v2 = SimModel('sentence-transformers/all-MiniLM-L6-v2', seoss33.get_issues())
    print(f"Loaded minilm_l6_v2 Successfully | Device: {minilm_l6_v2.device}")

    # Iterate over tests. Currntly only goes up to calculating recall
    recalls = []
    commits_count = len(seoss33.get_commits())
    for commit, issues in seoss33.get_trace_links():
        candidate_issues = get_candidate_issues(minilm_l6_v2, commit.message)
        recall = compute_recall_k(candidate_issues, issues)
        print(f"Testing...{min(len(recalls)/commits_count*100, 100):.2f}%\r", end="")
        recalls.append(recall)

    print(f"Recall: {sum(recalls)/len(recalls):.4f}")
    # get_ranked_issues()
    # compute_precision_k()

    seoss33.close()