import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from model import GnatGPT


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    device="cpu",
    context_len: int = 128,
):
    model.eval()
    ids = tokenizer.encode(prompt)
    # Model expects (B, T): we use B=1
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if the sequence has grown to long
            idx_cond = idx[:, -context_len:]

            logits, _ = model(idx_cond)  # (1, T, vocab_size)
            logits = logits[:, -1, :]  # last position -> (1, vocab_size)
            logits = logits / temperature

            if top_k is not None:
                # Zero out everything below the top-k logits
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)  # k-th largest
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = torch.cat([idx, next_id], dim=1)  # (1, T+1)

    return tokenizer.decode(idx[0].tolist())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate text from a GnatGPT checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("prompt", help="Text prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--device",
        type=str,
        default="default",
        help="Device to run on: cpu, mps, cuda, or default",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = cfg.device if args.device == "default" else args.device
    device = torch.device(device)

    with open("data/shakespeare.txt", "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    model = GnatGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        context_len=cfg.context_len,
    )
    print(output)
