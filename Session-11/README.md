# Hindi BPE SentencePiece tokenizer

This repository contains a SentencePiece BPE tokenizer trained on a Hindi corpus.

Files included
- `hindi_bpe.model` — SentencePiece model
- `hindi_bpe.vocab` — SentencePiece vocab
- `tokenizer_config.json` — tokenizer metadata

Usage (load model from Hugging Face Hub)

```python
from huggingface_hub import hf_hub_download
import sentencepiece as spm

model_path = hf_hub_download(repo_id="your-username/hindi-bpe-tokenizer", filename="hindi_bpe.model")
sp = spm.SentencePieceProcessor()
sp.load(model_path)
print(sp.encode("यह एक परीक्षण है।", out_type=str))
```

