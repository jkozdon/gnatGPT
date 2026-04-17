import torch
import os
from config import Config
from tokenizer import CharTokenizer
from dataset import make_dataloader
from model import GnatGPT


@torch.no_grad()
def estimate_loss(model, val_loader, eval_batches, device):
    """Average loss over a fixed nmumber of validation batches"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    cfg = Config()
    device = torch.device(cfg.device)

    # Data
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    cfg.vocab_size = tokenizer.vocab_size

    split = int(0.9 * len(text))
    train_loader = make_dataloader(
        text[:split], tokenizer, cfg.context_len, cfg.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        text[split:], tokenizer, cfg.context_len, cfg.batch_size, shuffle=False
    )

    # Model
    model = GnatGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer + schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_steps, eta_min=cfg.lr / 10
    )

    # Training loop
    model.train()
    loader_iter = iter(train_loader)
    os.makedirs("checkpoints", exist_ok=True)

    for step in range(cfg.max_steps):
        # Refill the iterator when the epoch ends
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Periodical eval + checkpoint
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_loss = estimate_loss(model, val_loader, eval_batches=20, device=device)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step:5d} | train loss {loss.item():.4f}"
                f"| val loss {val_loss:.4f} | lr {lr_now:.2e}"
            )
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                },
                f"checkpoints/ckpt_{step:05d}.pt",
            )
    print("Training complete")


if __name__ == "__main__":
    train()
