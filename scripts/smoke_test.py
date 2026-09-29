import torch
from transformers import GPT2Model, GPT2Tokenizer


def inspect(text: str, tokenizer: GPT2Tokenizer, model: GPT2Model, device: str) -> None:
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    token_ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"\nInput: {text!r}")
    print("Tokens:", tokens)
    print("Input shape:", tuple(encoded["input_ids"].shape))
    print("Hidden-state shape:", tuple(output.last_hidden_state.shape))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    inspect("52fa", tokenizer, model, device)
    inspect("5 2 f a", tokenizer, model, device)


if __name__ == "__main__":
    main()