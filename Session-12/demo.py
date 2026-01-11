from inference import Generator


def main():
    g = Generator("gpt2")
    outputs = g.generate("The future of AI is", max_length=60, num_return_sequences=2)
    for i, out in enumerate(outputs):
        print(f"--- RESULT {i} ---")
        print(out)


if __name__ == "__main__":
    main()
