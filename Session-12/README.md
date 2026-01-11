# Hugging Face-ready GPT helper

This repository contains a small custom GPT implementation plus helper scripts to run inference and upload the package to the Hugging Face Hub.

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo using an official `gpt2` model (fast):

```bash
python demo.py
```

3. To upload the repository files to the Hub:

```bash
set HF_TOKEN=your_token_here  # on Windows (PowerShell: $env:HF_TOKEN = "...")
python upload_to_hf.py --repo-id your-username/your-repo
```

Notes
- `model.py` contains a clean import-safe implementation of the custom GPT used in training.
- `inference.py` wraps either a Hugging Face `transformers` model (preferred) or the custom checkpoint.
- If you want to publish a trained `pytorch_model.bin`, upload it alongside these files and set `Generator` to point at it.
