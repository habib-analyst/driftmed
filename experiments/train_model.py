"""Train a small CNN on BloodMNIST; dump reference embeddings+probs.

Resumable: --resume continues from checkpoints/ckpt_last.pt.
Writes DONE marker on completion.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import numpy as np
import torch
import torch.nn as nn

from driftmed.model import SmallCNN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "bloodmnist.npz")
CKPT = os.path.join(ROOT, "checkpoints")
DONE = os.path.join(ROOT, "TRAIN_DONE")


def load_npz():
    d = np.load(DATA)
    # medmnist npz keys: train_images, train_labels, val_images, ...
    tr_x = d["train_images"].astype(np.float32) / 255.0
    tr_y = d["train_labels"].astype(np.int64).ravel()
    va_x = d["val_images"].astype(np.float32) / 255.0
    va_y = d["val_labels"].astype(np.int64).ravel()
    te_x = d["test_images"].astype(np.float32) / 255.0
    te_y = d["test_labels"].astype(np.int64).ravel()
    return (tr_x, tr_y), (va_x, va_y), (te_x, te_y)


def to_torch(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = load_npz()
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start = 0
    ckpt_path = os.path.join(CKPT, "ckpt_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        print(f"resumed at epoch {start}", flush=True)

    tr_xt, tr_yt = to_torch(tr_x), torch.from_numpy(tr_y)
    va_xt = to_torch(va_x)
    bs = 256
    for epoch in range(start, args.epochs):
        model.train()
        perm = torch.randperm(len(tr_xt))
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(tr_xt), bs):
            idx = perm[i:i + bs]
            xb, yb = tr_xt[idx], tr_yt[idx]
            opt.zero_grad()
            _, logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            _, vl = model(va_xt)
            vacc = (vl.argmax(1) == torch.from_numpy(va_y)).float().mean().item()
        print(f"epoch {epoch}: loss {loss_sum/tot:.4f} "
              f"train_acc {correct/tot:.4f} val_acc {vacc:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, ckpt_path)

    # dump reference embeddings + probs on TEST (deployment reference)
    model.eval()
    embs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(te_x), bs):
            xb = to_torch(te_x[i:i + bs])
            e, logits = model(xb)
            embs.append(e.numpy())
            probs.append(torch.softmax(logits, 1).numpy())
    np.savez(os.path.join(ROOT, "results", "reference.npz"),
             emb=np.vstack(embs), probs=np.vstack(probs), y=te_y,
             x=te_x)
    torch.save(model.state_dict(), os.path.join(CKPT, "model_final.pt"))
    open(DONE, "w").write("ok\n")
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
