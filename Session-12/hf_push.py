"""Push local files to a Hugging Face repo using a git-based Repository.

This clones the target repo (creating it if necessary), copies selected files
from the current working directory into the repo, commits and pushes.

Usage:
  set HF_TOKEN=...  # on Windows (PowerShell: $env:HF_TOKEN = '...')
  python hf_push.py --repo-id username/repo --repo-type model

Note: be careful when running from a large workspace; this copies files into a
temporary folder before pushing.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Repository


def copy_workspace(src: Path, dst: Path, include=None, exclude=None):
    include = include or []
    exclude = exclude or {".git", "__pycache__", ".ipynb_checkpoints"}
    for p in src.iterdir():
        name = p.name
        if name in exclude:
            continue
        # if include is provided, only copy those (files or directories)
        if include and name not in include:
            continue
        target = dst / name
        if p.is_dir():
            shutil.copytree(p, target)
        else:
            shutil.copy2(p, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="username/repo")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--repo-type", choices=["model", "space"], default="model")
    p.add_argument(
        "--include", nargs="*", help="specific files to include (default: common files)"
    )
    p.add_argument("--commit-msg", default="Add repo files from local workspace")
    args = p.parse_args()

    if not args.token:
        raise SystemExit("HF token required via --token or HF_TOKEN env var")

    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id, token=args.token, repo_type=args.repo_type
        )
        print("Repository created")
    except Exception:
        print("Repository may already exist; continuing")

    # clone into a temp dir
    tmpdir = Path(tempfile.mkdtemp())
    print(f"cloning into {tmpdir}")
    repo = Repository(
        local_dir=str(tmpdir),
        clone_from=args.repo_id,
        repo_type=args.repo_type,
        use_auth_token=args.token,
    )

    # decide which files to copy; default common files
    default_files = {
        "model.py",
        "inference.py",
        "demo.py",
        "README.md",
        "requirements.txt",
        "app.py",
        "requirements_space.txt",
    }
    include = set(args.include) if args.include else default_files

    # copy
    copy_workspace(Path.cwd(), tmpdir, include=include)

    # commit and push
    repo.git_add(pattern="*")
    repo.git_commit(commit_message=args.commit_msg)
    repo.git_push()

    print("Pushed files to Hugging Face Hub")


if __name__ == "__main__":
    main()
