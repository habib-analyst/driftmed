"""Reference CNN for driftmed experiments (BloodMNIST, 8 classes)."""
import numpy as np
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes=8, emb_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Linear(128, emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        e = torch.relu(self.embed(f))
        return e, self.head(e)


class ModelFn:
    """Callable(images [N,H,W,C] float32) -> (embeddings, probs) numpy."""

    def __init__(self, weights_path, batch=256):
        self.model = SmallCNN()
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.batch = batch

    def __call__(self, x):
        embs, probs = [], []
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).permute(0, 3, 1, 2)
        with torch.no_grad():
            for i in range(0, len(xt), self.batch):
                e, logits = self.model(xt[i:i + self.batch])
                embs.append(e.numpy())
                probs.append(torch.softmax(logits, 1).numpy())
        return np.vstack(embs), np.vstack(probs)
