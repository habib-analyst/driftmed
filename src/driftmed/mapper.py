"""DropMapper: the invention core of driftmed.

Maps measurable, LABEL-FREE shift statistics to an estimated accuracy drop,
calibrated on the canary probe's labeled degradation curve.

Method (all non-parametric, no distributional assumptions):
1. NULL NORMALIZATION — each shift statistic is converted to a z-score
   against its own null distribution (reference-vs-reference bootstraps).
   This puts MMD, CORAL, confidence-KS etc. on one comparable scale.
2. SHIFT INDEX — mean of the z-scores. One number summarizing "how far".
3. ISOTONIC CALIBRATION — monotone regression from shift index to accuracy
   drop on the canary table. More shift -> more drop; no parametric form.
4. HONEST UNCERTAINTY — bootstrap over (a) the deployment batch and
   (b) the canary calibration rows, giving a CI on the estimated drop,
   not a point estimate.

If the mapping does not generalize to held-out corruption families
(Spearman < 0.7), the kill-gate fires: the estimate is not trustworthy.
"""
import numpy as np

from . import shift as S

STAT_KEYS = ["mmd", "coral", "conf_ks", "entropy_ks", "pred_js"]


def _isotonic_fit(x, y):
    """PAVA isotonic regression (non-decreasing). Returns (xs, ys) step fn."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    # pool adjacent violators
    blocks = [[xi, yi, 1] for xi, yi in zip(x, y)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] / blocks[i][2] <= blocks[i + 1][1] / blocks[i + 1][2]:
            i += 1
        else:
            x_new = (blocks[i][0] * blocks[i][2] + blocks[i + 1][0] * blocks[i + 1][2]) / (blocks[i][2] + blocks[i + 1][2])
            y_new = blocks[i][1] + blocks[i + 1][1]
            w_new = blocks[i][2] + blocks[i + 1][2]
            blocks[i] = [x_new, y_new, w_new]
            del blocks[i + 1]
            if i > 0:
                i -= 1
    xs = np.array([b[0] / b[2] for b in blocks])
    ys = np.array([b[1] / b[2] for b in blocks])
    return xs, ys


def _isotonic_predict(xs, ys, xq):
    return np.interp(xq, xs, ys, left=ys[0], right=ys[-1])


class DropMapper:
    def __init__(self, n_bootstrap=200, ci=0.95, seed=0):
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.rng = np.random.default_rng(seed)
        self.null_mean_ = None
        self.null_std_ = None
        self.iso_x_ = None
        self.iso_y_ = None

    def fit_null(self, emb_ref, probs_ref, n_classes, n_resamples=50,
                 batch=512):
        """Null distribution of each stat: reference vs reference."""
        n = len(emb_ref)
        vals = {k: [] for k in STAT_KEYS}
        for _ in range(n_resamples):
            i = self.rng.choice(n, batch, replace=False)
            j = self.rng.choice(n, batch, replace=False)
            s = S.shift_statistics(emb_ref[i], emb_ref[j],
                                   probs_ref[i], probs_ref[j], n_classes)
            for k in STAT_KEYS:
                vals[k].append(s[k])
        self.null_mean_ = {k: float(np.mean(vals[k])) for k in STAT_KEYS}
        self.null_std_ = {k: float(np.std(vals[k])) + 1e-12
                          for k in STAT_KEYS}
        return self

    def zscores(self, stats):
        return {k: (stats[k] - self.null_mean_[k]) / self.null_std_[k]
                for k in STAT_KEYS}

    def shift_index(self, stats):
        z = self.zscores(stats)
        return float(np.mean([z[k] for k in STAT_KEYS]))

    def fit(self, canary_rows):
        """Fit isotonic calibration on the canary table."""
        idx = np.array([self.shift_index(r) for r in canary_rows])
        drop = np.array([r["acc_drop"] for r in canary_rows])
        self.iso_x_, self.iso_y_ = _isotonic_fit(idx, drop)
        self._canary_idx = idx
        self._canary_drop = drop
        return self

    def estimate(self, stats):
        """Point estimate of accuracy drop for a shift-statistic dict."""
        return float(_isotonic_predict(self.iso_x_, self.iso_y_,
                                       self.shift_index(stats)))

    def estimate_ci(self, emb_new, probs_new, emb_ref, probs_ref, n_classes,
                    batch=512):
        """Estimate + bootstrap CI. Resamples the deployment batch and the
        canary calibration rows."""
        n = len(emb_new)
        nr = len(emb_ref)
        # subsample the reference: full 3421-point MMD is O(n^2) per
        # bootstrap and too slow; 2*batch reference points are plenty
        ridx = self.rng.choice(nr, min(2 * batch, nr), replace=False)
        emb_r = emb_ref[ridx]
        pr_r = probs_ref[ridx]
        preds = []
        n_can = len(self._canary_idx)
        alpha = 1 - self.ci
        for _ in range(self.n_bootstrap):
            bi = self.rng.choice(n, min(batch, n), replace=True)
            s = S.shift_statistics(emb_r, emb_new[bi],
                                   pr_r, probs_new[bi], n_classes)
            ci_ = self.rng.choice(n_can, n_can, replace=True)
            xs_b, ys_b = _isotonic_fit(self._canary_idx[ci_],
                                       self._canary_drop[ci_])
            preds.append(_isotonic_predict(xs_b, ys_b, self.shift_index(s)))
        preds = np.array(preds)
        lo, hi = np.quantile(preds, [alpha / 2, 1 - alpha / 2])
        return float(np.mean(preds)), float(lo), float(hi)

    def evaluate(self, rows):
        """Spearman correlation between estimated and true drop on rows
        (typically HELD-OUT families). Returns (spearman, n)."""
        est = np.array([self.estimate(r) for r in rows])
        true = np.array([r["acc_drop"] for r in rows])
        return float(_spearman(est, true)), len(rows)


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = _rank(a)
    rb = _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _rank(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # tie-average
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    return avg[inv]
