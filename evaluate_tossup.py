# %%
import glob
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.zshenv"))

import argparse

from datasets import Dataset
from huggingface_hub import snapshot_download
from numpy import require
from transformers import pipeline


# ------------------------------------------------------------------ #
def preload_python_files(repo_dir: str):
    """Exec every *.py in the repo so PIPELINE_REGISTRY entries are created."""
    sys.path.insert(0, repo_dir)  # let plain `import` work
    for fname in glob.glob(os.path.join(repo_dir, "**/*.py"), recursive=True):
        if fname.endswith("__init__.py"):
            continue
        print(f"🐍 loading {fname}")
        spec = importlib.util.spec_from_file_location(Path(fname).stem, fname)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"⚠️  Skipping {fname}: {exc}")


# %%
if __name__ == "__main__":
    # p = pipeline("quizbowl-tossup", model="gpt2", device=0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        "-r",
        type=str,
        required=True,
        help="repo_id of the model submitted to HF",
    )

    args = parser.parse_args()
    # repo_name = args.repo_id.replace("/", "__")
    # local_dir = f"models/tossups/{repo_name}"
    # repo_dir = snapshot_download(repo_id=args.repo_id, local_dir=None)
    # print("⬇️ Downloaded model to", repo_dir)

    # preload_python_files(repo_dir)

    # %%
    p = pipeline("quizbowl-tossup", args.repo_id, device=0, trust_remote_code=True)

    small_data = [
        {
            "question_text": "Name this fashion capital that is home to the painting Mona Lisa."
        },
        {"question_text": "What is the capital of France?"},
        {"question_text": "What is the capital of Italy?"},
        {"question_text": "What is the capital of Spain?"},
        {"question_text": "What is the capital of Germany?"},
        {"question_text": "What is the capital of Japan?"},
        {"question_text": "What is the capital of China?"},
        {"question_text": "What is the capital of India?"},
        {"question_text": "What is the capital of Brazil?"},
        {"question_text": "What is the capital of Canada?"},
    ]

    dataset = Dataset.from_list(small_data)
    dataset = dataset.map(lambda x: p(x), remove_columns=["question_text"])
    dataset = dataset.rename_column("answer", "guess")
    print(dataset)

# %%
