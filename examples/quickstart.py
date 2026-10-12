"""End-to-end driftmed example on synthetic data (no downloads needed).

Builds a tiny synthetic classification problem, a canary table, fits the
mapper, and runs the streaming monitor — demonstrating the full
label-free accuracy-drop estimation loop.
"""
import numpy as np

from driftmed import (CanaryProbe, DropMapper, DriftMonitor,
                      CANARY_CORRUPTIONS)
from driftmed import shift as S

rng = np.random.default_rng(0)


def synthetic_model_fn_factory(n_classes=3, emb_dim=8):
    """A fake 'model': embeddings = class-conditional Gaussians + noise;
    probs = softmax of negative distances (well-calibrated by design)."""
    centers = rng.normal(0, 3, (n_classes, emb_dim))

    def model_fn(x):
        # x: (N, H, W, C) in [0,1]; use mean pixel as a pseudo-signal
        sig = x.mean(axis=(1, 2, 3))
        emb = centers[(sig * n_classes).astype(int).clip(0, n_classes - 1)]
        emb = emb + rng.normal(0, 0.5, emb.shape)
        d2 = ((emb[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        logits = -d2
        e = np.exp(logits - logits.max(1, keepdims=True))
        return emb, e / e.sum(1, keepdims=True)

    return model_fn


def main():
    n_classes, n_ref = 3, 1200
    x_ref = rng.uniform(0, 1, (n_ref, 28, 28, 1)).astype(np.float32)
    y_ref = (x_ref.mean(axis=(1, 2, 3)) * n_classes).astype(int).clip(
        0, n_classes - 1)
    model_fn = synthetic_model_fn_factory(n_classes)

    emb_ref, probs_ref = model_fn(x_ref)
    acc = (probs_ref.argmax(1) == y_ref).mean()
    print(f"reference accuracy: {acc:.4f}")

    probe = CanaryProbe(model_fn, n_classes)
    canary = probe.build_table(x_ref[:600], y_ref[:600], CANARY_CORRUPTIONS)
    mapper = DropMapper(n_bootstrap=50, seed=0)
    mapper.fit_null(emb_ref, probs_ref, n_classes, n_resamples=20, batch=256)
    mapper.fit(canary)

    monitor = DriftMonitor(mapper, emb_ref, probs_ref, n_classes,
                           alert_threshold=0.03)
    # clean batch
    rec = monitor.update(emb_ref[:256], probs_ref[:256])
    print(f"clean batch: est_drop={rec['est_drop']:.4f} "
          f"[{rec['ci_lo']:.4f}, {rec['ci_hi']:.4f}] alert={rec['alert']}")
    # shifted batch (heavy noise)
    from driftmed.canary import gaussian_noise
    emb_s, pr_s = model_fn(gaussian_noise(x_ref[:256], 5))
    rec = monitor.update(emb_s, pr_s)
    true_drop = acc - (pr_s.argmax(1) == y_ref[:256]).mean()
    print(f"noisy batch: est_drop={rec['est_drop']:.4f} "
          f"[{rec['ci_lo']:.4f}, {rec['ci_hi']:.4f}] alert={rec['alert']} "
          f"(true drop {true_drop:.4f})")


if __name__ == "__main__":
    main()
