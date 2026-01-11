"""Small helper to upload selected files to a Hugging Face Hub repo.

Usage:
  export HF_TOKEN=... (or set env var on Windows)
  python upload_to_hf.py --repo-id your-username/your-repo
"""

import os
import argparse
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="repo id like username/repo")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()
    if args.token is None:
        raise SystemExit("HF token required via --token or HF_TOKEN env var")

    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo_id, token=args.token)
    except Exception:
        print("Repo may already exist, continuing to upload files")

    files_to_upload = [
        "model.py",
        "inference.py",
        "requirements.txt",
        "README.md",
        "demo.py",
    ]

    for f in files_to_upload:
        if not os.path.exists(f):
            print(f"skipping missing {f}")
            continue
        print(f"uploading {f} -> {args.repo_id}")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=os.path.basename(f),
            repo_id=args.repo_id,
            token=args.token,
        )


if __name__ == "__main__":
    main()
