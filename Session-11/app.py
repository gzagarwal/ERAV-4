import os
import sentencepiece as spm
import gradio as gr


def load_model():
    # Prefer model in src/ shipped with repo
    local_path = os.path.join("src", "hindi_bpe.model")
    sp = spm.SentencePieceProcessor()
    if os.path.exists(local_path):
        sp.load(local_path)
    else:
        # Fall back to just attempting to load from current dir
        if os.path.exists("hindi_bpe.model"):
            sp.load("hindi_bpe.model")
        else:
            raise FileNotFoundError(
                "SentencePiece model not found. Place `hindi_bpe.model` in `src/` or repo root."
            )
    return sp


SP = None


def tokenize(text: str):
    global SP
    if SP is None:
        SP = load_model()
    if not text:
        return "", ""
    tokens = SP.encode(text, out_type=str)
    ids = SP.encode(text, out_type=int)
    return " ".join(tokens), str(ids)


title = "Hindi BPE SentencePiece Tokenizer"
description = "Enter Hindi text to see the SentencePiece BPE tokens and ids."

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    with gr.Row():
        txt = gr.Textbox(lines=3, placeholder="यहाँ हिंदी टेक्स्ट लिखें...", label="Input")
    with gr.Row():
        out_tokens = gr.Textbox(label="Tokens")
        out_ids = gr.Textbox(label="Token IDs")
    btn = gr.Button("Tokenize")
    btn.click(tokenize, inputs=txt, outputs=[out_tokens, out_ids])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
