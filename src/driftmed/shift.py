"""Shift statistics for driftmed: MMD, CORAL, confidence-distribution shift.

All functions operate on numpy arrays. Embeddings are expected as
(n_samples, n_features); probability vectors as (n_samples, n_classes).
"""
import numpy as np


def _median_bandwidth(X, Y, subsample=2000):
    rng = np.random.default_rng(0)
    Z = np.vstack([X, Y])
    if len(Z) > subsample:
        Z = Z[rng.choice(len(Z), subsample, replace=False)]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    triu = d2[np.triu_indices(len(Z), k=1)]
    med = np.median(triu)
    return float(med) if med > 0 else 1.0


def mmd_rbf(X, Y, gamma=None):
    """Unbiased squared MMD with RBF kernel. ~0 for identical distributions."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = len(X), len(Y)
    if gamma is None:
        gamma = 1.0 / (2.0 * _median_bandwidth(X, Y))
    def K(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    Kxx = K(X, X)
    Kyy = K(Y, Y)
    Kxy = K(X, Y)
    # unbiased: exclude diagonals
    term_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    term_xy = Kxy.sum() / (n * m)
    return float(term_xx + term_yy - 2 * term_xy)


def coral_distance(X, Y):
    """Frobenius norm of covariance difference (CORAL-style)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Cx = np.cov(X, rowvar=False)
    Cy = np.cov(Y, rowvar=False)
    return float(np.linalg.norm(Cx - Cy, ord="fro") / (Cx.shape[0]))


def confidence_stats(probs):
    """Max-confidence and entropy summary of a probability matrix."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    return conf, ent


def ks_distance(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def prediction_hist_shift(probs_ref, probs_new, n_classes):
    """Symmetric KL between predicted-class histograms (smoothed)."""
    pr = np.asarray(probs_ref).argmax(axis=1)
    pn = np.asarray(probs_new).argmax(axis=1)
    hr, _ = np.histogram(pr, bins=n_classes, range=(0, n_classes))
    hn, _ = np.histogram(pn, bins=n_classes, range=(0, n_classes))
    hr = (hr + 1) / (hr.sum() + n_classes)
    hn = (hn + 1) / (hn.sum() + n_classes)
    m = 0.5 * (hr + hn)
    js = 0.5 * (hr * np.log(hr / m)).sum() + 0.5 * (hn * np.log(hn / m)).sum()
    return float(js)


def shift_statistics(emb_ref, emb_new, probs_ref, probs_new, n_classes):
    """Full shift-statistic vector used by the drop mapper.

    Returns dict with: mmd, coral, conf_ks, entropy_ks, pred_js.
    """
    cr, er = confidence_stats(probs_ref)
    cn, en = confidence_stats(probs_new)
    return {
        "mmd": mmd_rbf(emb_ref, emb_new),
        "coral": coral_distance(emb_ref, emb_new),
        "conf_ks": ks_distance(cr, cn),
        "entropy_ks": ks_distance(er, en),
        "pred_js": prediction_hist_shift(probs_ref, probs_new, n_classes),
    }
