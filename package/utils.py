import bz2
import shutil
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

from config import DATA_DIR
RECALL_DATA_PREFIX = "recall_data"
PRECISION_DATA_PREFIX = "precision_data"

def decompress_bz2(in_path, out_path=None, keep=False):
    if out_path is None:
        out_path = in_path.with_suffix("")

    print(f"Decompressing {in_path.parent.name}/{in_path.name} → {out_path.parent.name}/{out_path.name}")
    with bz2.open(in_path, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    if not keep:
        in_path.unlink()
    
def unzip(in_path, out_path):
    print(f"Extracting {in_path.parent.name}/{in_path.name} → {out_path.parent.name}/{out_path.name}")
    with zipfile.ZipFile(in_path, "r") as zip_ref:
        zip_ref.extractall(out_path)

def get_latest_data_file_index(prefix: str):
    """
    Returns the highest x value from files in the data directory that are named <prefix_000x>
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\..+")

    max_idx = None

    for file in Path(DATA_DIR).iterdir():
        match = pattern.match(file.name)
        if match:
            idx = int(match.group(1))
            max_idx = idx if max_idx is None else max(max_idx, idx)

    return max_idx

def get_latest_data_file(prefix: str):
    idx = get_latest_data_file_index(prefix)
    if idx is None:
        raise FileNotFoundError()
    return DATA_DIR / f"{prefix}_{idx:04d}.file"

def get_next_data_file(prefix: str):
    idx = get_latest_data_file_index(prefix)
    if idx is None:
        idx = -1
    return DATA_DIR / f"{prefix}_{idx+1:04d}.file"

def save_dict(data: dict, name: str):
    data_path = get_next_data_file(name)
    with open(data_path, "w") as f:
        json.dump(data, f)

def load_dict(name: str):
    data_path = get_latest_data_file(name)
    with open(data_path, "r") as f:
        return json.load(f)

def save_recalls_data(recalls: dict):
    save_dict(recalls, RECALL_DATA_PREFIX)

def save_precision_data(recalls: dict):
    save_dict(recalls, PRECISION_DATA_PREFIX)

def load_recall_data():
    return load_dict(RECALL_DATA_PREFIX)

def load_precision_data():
    return load_dict(PRECISION_DATA_PREFIX)

def plot(ax, x, label=""):
    ax.plot(range(len(x)), x, label=label)
    return ax

def plot_retrievers(axis, retrievers:dict, title="", xlabel="", ylabel=""):
    for retriever, metric in retrievers.items():
        plot(axis, metric.values(), label=retriever)
    axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return axis

def plot_projects(recalls: dict, precisions: dict):
    for project in recalls.keys():
        fig, axs = plt.subplots(6, 6, figsize=(18, 18))

        for i, (project, _) in enumerate(recalls.items()):
            row = i // 6
            col = i % 6
            plot_retrievers(axs[row][col], recalls[project], title=project, xlabel="K", ylabel="Recall")
        plt.show()

    for project in precisions.keys():
        fig, axs = plt.subplots(6, 6, figsize=(18, 18))

        for i, (project, _) in enumerate(precisions.items()):
            row = i // 6
            col = i % 6
            plot_retrievers(axs[row][col], precisions[project], title=project, xlabel="K", ylabel="Precision")
        plt.show()

def plot_average_across_projects(recalls: dict, precisions: dict):
    for project in recalls.keys():
        fig, axs = plt.subplots(1, 2)
        plot_retrievers(axs[0], recalls[project], title=project, xlabel="K", ylabel="Recall")
        plot_retrievers(axs[1], precisions[project], title=project, xlabel="K", ylabel="Precision")
        plt.show()

    for i, (project, _) in enumerate(recalls.items()):
        row = i // 6
        col = i % 6
        plot_retrievers(axs[row][col], recalls[project], title=project, xlabel="K", ylabel="Recall")

    plt.show()
