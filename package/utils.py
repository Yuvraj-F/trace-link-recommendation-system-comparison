import bz2
import shutil
import zipfile

compressed_file = "db.sqlite3.bz2"
output_file = "db.sqlite3"

def decompress_bz2(in_path, out_path=None, keep=False):
    if out_path is None:
        out_path = in_path.with_suffix("")

    print(f"Decompressing {in_path.parent.name}/{in_path.name} → {out_path.parent.name}/")
    with bz2.open(in_path, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    if not keep:
        in_path.unlink()
    
def unzip(in_path, out_path):
    print(f"Extracting {in_path.parent.name}/{in_path.name} → {out_path.parent.name}/")
    with zipfile.ZipFile(in_path, "r") as zip_ref:
        zip_ref.extractall(out_path)

def remove_issue_id_from_commit(issue_id: str, commit: str):
    return commit.replace(issue_id, "")