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
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_04999.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "ROMEO:"

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device(cfg.device)

    with open("data/shakespeare.txt", "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    model = GnatGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=300,
        temperature=0.8,
        top_k=40,
        device=device,
        context_len=cfg.context_len,
    )
    print(output)
