from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
PACKAGE_DIR = ROOT_DIR / "package"
CACHE_DIR = ROOT_DIR / ".hf_cache"
DATA_DIR = ROOT_DIR / "data"

def _load_from_env():
    if not Path.exists(ROOT_DIR / ".env"):
        raise FileNotFoundError(f".env file not found in {ROOT_DIR}\nRefer to README.md")
    with open(ROOT_DIR / ".env", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("COMPRESSED_DATASET_PATH"):
                compressed_path = Path(line.split("=", 1)[1].strip())
                dataset_dir = compressed_path.parent
                print(f"Found compressed dataset at {compressed_path}")
                return dataset_dir, compressed_path

def _load_dataset_path():
    zip_files = list(ROOT_DIR.glob("*.zip"))
    if len(zip_files) == 1:
        dataset_dir = ROOT_DIR
        compressed_path = ROOT_DIR / zip_files[0].name
        print(f"Found compressed dataset at {compressed_path}")
        return dataset_dir, compressed_path
    else:
        print(f"None or Multiple zip files found where one was expected")
        print(f"Loading path from .env")
        return _load_from_env()

def _setup_local_cache():
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)


def init_config():
    print(f"Loading configuration...")

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    
    _setup_local_cache()
    return _load_dataset_path()

__all__ = ["init_config", "ROOT_DIR", "PACKAGE_DIR"]