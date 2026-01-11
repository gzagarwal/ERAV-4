# Model Card / Hub README

Model: Custom GPT implementation (minGPT-like) + HF helpers

Overview
--------
This repository contains a small custom GPT implementation (`model.py`) and
helpers to run inference (`inference.py`) and deploy to the Hugging Face Hub
(`hf_push.py`, `upload_to_hf.py`). It also includes a Gradio app (`app.py`)
for quick deployment to Hugging Face Spaces.

Usage (Model Hub)
------------------
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. If you want to test with an official GPT-2 model locally:

```bash
python demo.py
```

3. To push files to the Hub (git style):

```powershell
set HF_TOKEN=your_token_here
python hf_push.py --repo-id your-username/your-repo --repo-type model
```

4. Alternatively, use the simple uploader:

```powershell
set HF_TOKEN=your_token_here
python upload_to_hf.py --repo-id your-username/your-repo
```

Usage (Spaces)
---------------
1. Prepare the Space requirements:

```bash
pip install -r requirements_space.txt
```

2. Push to a Space (repo_type=space) with the same `hf_push.py`:

```powershell
set HF_TOKEN=your_token_here
python hf_push.py --repo-id your-username/your-space --repo-type space
```

Notes
-----
- If you have a trained checkpoint, upload `pytorch_model.bin` alongside these files.
- `inference.py` supports loading a Hugging Face `transformers` model by name
  (e.g., `gpt2`) or a local PyTorch checkpoint path.
- For Spaces, the `app.py` uses Gradio and expects the same environment as
  `requirements_space.txt`.

Security
--------
Ensure your `HF_TOKEN` is kept secret. Do not commit it to the repository.

License
-------
Add your chosen license file before publishing.
