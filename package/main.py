from pathlib import Path
from llm import example_llm
from similarity_model import RETRIEVER_NAMES, Retriever, build_retriever
from config import *
from utils import *
from db import *
from query import *

def get_candidate_issues_batched(get_relevant_issues, commits: list[Commit], topk=None):
    queries = [c.message for c in commits]
    results = get_relevant_issues(queries)

    if topk is not None:
        results = [r[:topk] for r in results]

    return results

def compute_recall_precision_curves(pred_issues: list[Issue], true_issues: list[Issue]):
    true_set = set(true_issues)

    recalls = {}
    precisions = {}

    found = 0

    for i, issue in enumerate(pred_issues, start=1):
        if issue in true_set:
            found += 1

        recalls[i] = found / len(true_set)
        precisions[i] = found / i

    return recalls, precisions

def commit_counts(seoss33: SEOSS33):
    trace_links, keys = seoss33.query(SELECT_ALL_TRACE_LINKS)
    print(f"Trace link keys: {keys}\n")

    commit_counts = {}
    for link in trace_links:
        commit_counts[link.commit_hash] = commit_counts.get(link.commit_hash, 0) + 1

    for c, count in commit_counts.items():
        if count > 5:
            print(f"Issues per commit: {count} ({c})")

def evaluate_retriever(retriever: Retriever, seoss33: SEOSS33) -> float:
    """
    Average per-commit recall over all trace links in the loaded project DB.

    @param retriever: Configured first-stage model (issues indexed at build time).
    @param seoss33: Open SEOSS33 dataset handle.
    @return: Mean recall over entire test set.
    """

    recalls_per_commit = []
    precisions_per_commit = []
    trace_links = list(seoss33.get_trace_links())
    total = len(trace_links)

    commits = [c for c, _ in trace_links]
    issues = [i for _, i in trace_links]
    pred_batches = get_candidate_issues_batched(retriever.get_relevant_issues, commits, topk=100)
    for preds, true_issues in zip(pred_batches, issues):
        recall_curve, precision_curve = compute_recall_precision_curves(preds, true_issues)
        recalls_per_commit.append(recall_curve)
        precisions_per_commit.append(precision_curve)

    print()

    recalls_k = {} 
    precisions_k = {}
    for recall_curve, precision_curve in zip(recalls_per_commit, precisions_per_commit):
        for k, recall in recall_curve.items():
            recalls_k.setdefault(k, []).append(recall)
            precisions_k.setdefault(k, []).append(precision_curve[k])

    
    average_recalls_k = {}
    
    for k, recalls in recalls_k.items():
        average_recall = sum(recalls)/len(recalls)
        average_recalls_k[k] = average_recall
        if k in [1, 5, 10, 20, 50, 100]:
            print(f"Recall@{k}: {average_recall:.4f}", end=" | ")
    print()

    average_precisions_k = {}
    for k, precisions in precisions_k.items():
        average_precision = sum(precisions)/len(precisions)
        average_precisions_k[k] = average_precision
        if k in [1, 5, 10, 20, 50, 100]:
            print(f"Precision@{k}: {average_precision:.4f}", end=" | ")
    print()

    return average_recalls_k, average_precisions_k


if __name__ == "__main__":
    # Compare all four retrieval backends on the same issue corpus and trace links.
    seoss33 = init_db()

    recalls_across_projects = {}
    precisions_across_projects = {}
    for project in seoss33:
        issues = project.get_issues()

        print("------------------------------------------------------")
        print(f"\nProject {project.name}: {len(issues)} trace links\n")
        print("------------------------------------------------------")

        retriever_recalls = {}
        retriever_precisions = {}
        for name in RETRIEVER_NAMES:
            print(f"--- {name} ---")
            print("Building retriever (download/load/encode as needed)...")
            retriever = build_retriever(name, issues)
            if hasattr(retriever, "device"):
                print(f"Ready | Device: {retriever.device}")
            else:
                print("Ready")
            average_recalls, average_precisions = evaluate_retriever(retriever, project)
            mean_recall = sum(average_recalls.values())/len(average_recalls.values())
            retriever_recalls[name] = average_recalls
            retriever_precisions[name] = average_precisions
        
        recalls_across_projects[project.name] = retriever_recalls
        precisions_across_projects[project.name] = retriever_precisions

    save_recalls_data(recalls_across_projects)
    save_precision_data(precisions_across_projects)

    for project in seoss33:
        project.close()