from pathlib import Path
from llm import example_llm
from similarity_model import RETRIEVER_NAMES, Retriever, build_retriever
from config import *
from utils import *
from db import *
from query import *

def get_candidate_issues(model: Retriever, commit: Commit | str) -> list[Issue]:
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

def compute_recall_curve(pred_issues: list[Issue], true_issues: list[Issue], ks=None) -> float:
    if ks == None:
        ks = [1] + list(range(5, 101, 5))

    recalls = {}
    for k in ks:
        recalls[k] = compute_recall_k(pred_issues, true_issues, k=k)
    return recalls

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


def eval():
    # Iterate over tests. Currntly only goes up to calculating recall
    ks = [1] + list(range(5, 101, 5))
    recalls_per_commit = []
    commits_count = len(seoss33.get_commits())
    count = 0
    for commit, issues in seoss33.get_trace_links():
        candidate_issues = get_candidate_issues(minilm_l6_v2, commit.message)
        recall_curve = compute_recall_curve(candidate_issues, issues, ks=ks)
        recalls_per_commit.append(recall_curve)
        print(f"Testing...{min(len(recalls_per_commit)/commits_count*100, 100):.2f}%\r", end="")
        count += 1
        

    recalls_k = {}
    for recall_curve in recalls_per_commit:
        for k, recall in recall_curve.items():
            recalls_k.setdefault(k, []).append(recall)
    
    average_recalls_k = []
    for k, recalls in recalls_k.items():
        average_recall = sum(recalls)/len(recalls)
        average_recalls_k.append(average_recall)
        print(f"Recall@{k}: {average_recall:.4f}", end=" | ")

    print(f"\nRecall: {sum(average_recalls_k)/len(average_recalls_k):.4f}")

    
def evaluate_retriever(retriever: Retriever, seoss33: SEOSS33) -> float:
    """
    Average per-commit recall over all trace links in the loaded project DB.

    @param retriever: Configured first-stage model (issues indexed at build time).
    @param seoss33: Open SEOSS33 dataset handle.
    @return: Mean of compute_recall_k scores across commits.
    """
    recalls = []
    trace_links = list(seoss33.get_trace_links())
    total = len(trace_links)
    for i, (commit, issues) in enumerate(trace_links, start=1):
        candidate_issues = get_candidate_issues(retriever, commit)
        recall = compute_recall_k(candidate_issues, issues)
        recalls.append(recall)
        print(f"Testing...{i / total * 100:.2f}%\r", end="")
    print()
    return sum(recalls) / len(recalls) if recalls else 0.0


if __name__ == "__main__":
    # Compare all four retrieval backends on the same issue corpus and trace links.
    seoss33 = init_db()
    issues = seoss33.get_issues()

    print(f"Evaluating {len(RETRIEVER_NAMES)} retrieval techniques on {len(issues)} issues\n")

    for name in RETRIEVER_NAMES:
        print(f"--- {name} ---")
        print("Building retriever (download/load/encode as needed)...")
        retriever = build_retriever(name, issues)
        if hasattr(retriever, "device"):
            print(f"Ready | Device: {retriever.device}")
        else:
            print("Ready")
        mean_recall = evaluate_retriever(retriever, seoss33)
        print(f"Mean recall: {mean_recall:.4f}\n")

    # get_ranked_issues()
    # compute_precision_k()
    seoss33.close()