# Hindi BPE SentencePiece tokenizer
This repository contains a SentencePiece BPE tokenizer trained on a Hindi corpus.
**Files included**
- `src/hindi_bpe.model` — SentencePiece model
- `src/hindi_bpe.vocab` — SentencePiece vocab
- `tokenizer_config.json` — tokenizer metadata
- `app.py` — minimal Gradio demo (added)
**Demo (Hugging Face Spaces)**
This repo includes a simple Gradio app at `app.py` so it can be deployed as a Hugging Face Space (Gradio).
To run locally:
```bash
pip install -r requirements.txt
python app.py
```
On Hugging Face Spaces: create a new Space (Gradio), push this repository, and the app will run automatically.
**Usage (load model from Hugging Face Hub)**
```python
from huggingface_hub import hf_hub_download
import sentencepiece as spm

model_path = hf_hub_download(repo_id="your-username/hindi-bpe-tokenizer", filename="hindi_bpe.model")
sp = spm.SentencePieceProcessor()
sp.load(model_path)
print(sp.encode("यह एक परीक्षण है।", out_type=str))
```
If you prefer to load the local model shipped in this repo, use `src/hindi_bpe.model` directly:
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("src/hindi_bpe.model")
print(sp.encode("यह एक परीक्षण है।", out_type=str))
```
**Notes**
- The `requirements.txt` includes `gradio`, `sentencepiece`, and `huggingface-hub` required for the demo.
- If you plan to use large frameworks (e.g. `torch`) add them to `requirements.txt` only when needed.

```

