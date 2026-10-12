"""Canary probe: synthetic corruptions of reference data with LABELED
degradation curves, used to calibrate the unlabeled shift->drop mapping.

The canary set is the labeled anchor: we know exactly how much each
synthetic shift hurts accuracy, so at deployment we can map measured
(unlabeled) shift statistics to an estimated accuracy drop.
"""
import numpy as np

from . import shift as S


def gaussian_noise(x, severity):
    rng = np.random.default_rng(1000 + severity)
    sigma = 0.02 * severity
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)


def blur(x, severity):
    # box blur with radius ~ severity, pure numpy
    r = int(severity)
    if r <= 0:
        return x
    k = 2 * r + 1
    pad = r
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for i in range(k):
        for j in range(k):
            out += xp[:, i:i + x.shape[1], j:j + x.shape[2], :]
    return out / (k * k)


def brightness(x, severity):
    return np.clip(x + 0.06 * severity, 0, 1)


def contrast(x, severity):
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip((x - mean) * (1 - 0.12 * severity) + mean, 0, 1)


def rotation(x, severity):
    # 90-degree rotations only (exact, no interpolation artifacts)
    k = severity % 4
    return np.rot90(x, k, axes=(1, 2)).copy()


def cutout(x, severity):
    rng = np.random.default_rng(2000 + severity)
    out = x.copy()
    h, w = x.shape[1], x.shape[2]
    s = min(h, w) // (5 - min(severity, 4))
    for i in range(len(x)):
        top = rng.integers(0, h - s + 1)
        left = rng.integers(0, w - s + 1)
        out[i, top:top + s, left:left + s, :] = 0.0
    return out


# Canary families: corruptions the mapper is CALIBRATED on.
CANARY_CORRUPTIONS = {
    "gaussian_noise": (gaussian_noise, [1, 2, 3, 4, 5]),
    "blur": (blur, [1, 2, 3]),
    "brightness": (brightness, [1, 2, 3, 4]),
}
# Held-out families: used ONLY at evaluation, never for calibration.
# The kill-gate lives here — if the mapping generalizes to unseen
# corruption families, it is a real mapping, not curve-fitting.
HELDOUT_CORRUPTIONS = {
    "contrast": (contrast, [1, 2, 3, 4]),
    "rotation": (rotation, [1, 2, 3]),
    "cutout": (cutout, [1, 2, 3]),
}


class CanaryProbe:
    """Build the labeled shift->drop calibration table.

    model_fn: callable(images) -> (embeddings, probs) as numpy arrays.
    """

    def __init__(self, model_fn, n_classes):
        self.model_fn = model_fn
        self.n_classes = n_classes

    def _stats_for(self, x_ref, x_shifted):
        emb_r, pr_r = self.model_fn(x_ref)
        emb_s, pr_s = self.model_fn(x_shifted)
        return S.shift_statistics(emb_r, emb_s, pr_r, pr_s, self.n_classes)

    def build_table(self, x_ref, y_ref, corruptions, batch=512):
        """Returns list of dicts: {family, severity, acc_drop, stats...}.

        acc_drop = acc(clean) - acc(corrupted), measured WITH labels.
        """
        rows = []
        emb_r, pr_r = self.model_fn(x_ref)
        acc_clean = float((pr_r.argmax(axis=1) == y_ref).mean())
        for family, (fn, severities) in corruptions.items():
            for sev in severities:
                x_c = fn(x_ref, sev)
                emb_c, pr_c = self.model_fn(x_c)
                acc_c = float((pr_c.argmax(axis=1) == y_ref).mean())
                stats = S.shift_statistics(emb_r, emb_c, pr_r, pr_c,
                                           self.n_classes)
                rows.append({
                    "family": family,
                    "severity": sev,
                    "acc_clean": acc_clean,
                    "acc": acc_c,
                    "acc_drop": acc_clean - acc_c,
                    **stats,
                })
        return rows
