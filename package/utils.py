import bz2
import shutil
import zipfile
import matplotlib.pyplot as plt

compressed_file = "db.sqlite3.bz2"
output_file = "db.sqlite3"

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

def remove_issue_id_from_commit(issue_id: str, commit: str):
    return commit.replace(issue_id, "")

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
        plot_retrievers(axs[0], recalls[project], title=project, xlabel="K", ylabel="Recall")
        plot_retrievers(axs[1], precisions[project], title=project, xlabel="K", ylabel="Precision")
        plt.show()

    for i, (project, _) in enumerate(recalls.items()):
        row = i // 6
        col = i % 6
        plot_retrievers(axs[row][col], recalls[project], title=project, xlabel="K", ylabel="Recall")

    plt.show()
