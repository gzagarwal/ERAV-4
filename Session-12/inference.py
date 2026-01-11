"""Simple inference wrapper supporting transformers GPT-2 or the custom `model.GPT`.

Usage:
    from inference import Generator
    g = Generator(model_name_or_path="gpt2")
    print(g.generate("Hello world", max_length=50))
"""

import os
from typing import Optional

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

from model import GPT, GPTConfig


class Generator:
    def __init__(self, model_name_or_path: str = "gpt2", device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # prefer transformers if the identifier looks like a pretrained model
        if _HAS_TRANSFORMERS and (
            model_name_or_path.startswith("gpt2") or os.path.isdir(model_name_or_path)
        ):
            self._use_transformers = True
        else:
            self._use_transformers = False

        if self._use_transformers:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(
                self.device
            )
        else:
            # expect a torch state_dict path (e.g., pytorch_model.bin)
            self.tokenizer = None
            if _HAS_TIKTOKEN:
                self.enc = tiktoken.get_encoding("gpt2")
            else:
                self.enc = None
            # create a default config unless user provided config file next to checkpoint
            config = GPTConfig()
            self.model = GPT(config)
            if os.path.exists(model_name_or_path):
                sd = torch.load(model_name_or_path, map_location=self.device)
                try:
                    self.model.load_state_dict(sd)
                except Exception:
                    # maybe it's a dict with key 'model_state_dict'
                    if isinstance(sd, dict) and "model_state_dict" in sd:
                        self.model.load_state_dict(sd["model_state_dict"])
                    else:
                        raise
            self.model.to(self.device)

    def generate(
        self, prompt: str, max_length: int = 50, num_return_sequences: int = 1
    ):
        if self._use_transformers:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                num_return_sequences=num_return_sequences,
            )
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]
        else:
            if self.enc is None:
                raise RuntimeError(
                    "tiktoken required for custom GPT inference when not using transformers tokenizer"
                )
            tokens = self.enc.encode(prompt)
            x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            while x.size(1) < max_length:
                with torch.no_grad():
                    logits = self.model(x)[0]
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=1)
            return [self.enc.decode(x[0, :max_length].tolist())]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("model", help="model name or path", nargs="?", default="gpt2")
    p.add_argument("--prompt", default="Hello world")
    p.add_argument("--max_length", type=int, default=50)
    args = p.parse_args()

    g = Generator(args.model)
    out = g.generate(args.prompt, max_length=args.max_length)
    for o in out:
        print(o)
