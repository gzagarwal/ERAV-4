import gradio as gr
from inference import Generator


def infer(prompt, model="gpt2", max_length=64):
    g = Generator(model)
    out = g.generate(prompt, max_length=max_length, num_return_sequences=1)
    return out[0]


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=2, label="Prompt"),
        gr.Textbox(value="gpt2", label="Model"),
        gr.Slider(minimum=8, maximum=512, step=1, value=64, label="Max length"),
    ],
    outputs=[gr.Textbox(label="Generated")],
    title="Simple GPT Demo",
    description="Use an HF `gpt2` model or a custom checkpoint path.",
)


if __name__ == "__main__":
    demo.launch()
