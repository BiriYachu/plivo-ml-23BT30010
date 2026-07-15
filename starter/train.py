import argparse
import time
import math
import torch
from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, max_steps, peak_lr, warmup_steps):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    min_lr = peak_lr * 0.1
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    
    assert args.steps <= MAX_STEPS
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens (vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    t0 = time.time()
    losses = []
    warmup_steps = 200
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.lr, warmup_steps)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
            
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  ({(time.time()-t0)/step*1000:.0f} ms/step)  lr {lr:.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))},
            "steps": args.steps,
            "train_loss_curve": losses,
        },
        args.out,
    )
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()