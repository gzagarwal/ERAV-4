import os
from huggingface_hub import HfApi
import os
from huggingface_hub import HfApi


def main():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    username = os.environ.get("HF_USERNAME")
    if not token or not username:
        raise SystemExit(
            "Set HF_USERNAME and HUGGINGFACE_HUB_TOKEN environment variables before running."
        )

    repo_name = os.environ.get("HF_REPO_NAME", "hindi-bpe-tokenizer")
    repo_id = f"{username}/{repo_name}"

    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

    files = [
        ("hindi_bpe.model", "src/hindi_bpe.model"),
        ("hindi_bpe.vocab", "src/hindi_bpe.vocab"),
        ("README.md", "README.md"),
        ("tokenizer_config.json", "tokenizer_config.json"),
    ]

    for target_name, local_path in files:
        if not os.path.exists(local_path):
            print(f"Skipping missing: {local_path}")
            continue
        print(f"Uploading {local_path} -> {repo_id}/{target_name}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=target_name,
            repo_id=repo_id,
            token=token,
        )

    print("Upload complete. Visit https://huggingface.co/" + repo_id)


if __name__ == "__main__":
    main()
            print("Upload complete. Visit https://huggingface.co/" + repo_id)
