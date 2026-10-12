"""Streaming drift monitor: alerts with honest uncertainty.

The monitor consumes deployment batches (embeddings, probabilities,
optional metadata) and emits drift alerts when the LOWER bound of the
estimated accuracy-drop CI exceeds a threshold — i.e. we alert only when
we are confident the drop is real, controlling false alarms.
"""
import numpy as np

from . import shift as S
from .mapper import DropMapper, STAT_KEYS
from .metadata import metadata_drift


class DriftMonitor:
    def __init__(self, mapper, emb_ref, probs_ref, n_classes,
                 meta_ref=None, alert_threshold=0.03):
        self.mapper = mapper
        self.emb_ref = emb_ref
        self.probs_ref = probs_ref
        self.n_classes = n_classes
        self.meta_ref = meta_ref
        self.alert_threshold = alert_threshold
        self.history = []

    def update(self, emb_new, probs_new, meta_new=None, batch=512):
        """Process one deployment batch. Returns alert dict."""
        s = S.shift_statistics(self.emb_ref, emb_new,
                               self.probs_ref, probs_new, self.n_classes)
        est, lo, hi = self.mapper.estimate_ci(
            emb_new, probs_new, self.emb_ref, self.probs_ref,
            self.n_classes, batch=batch)
        alert = lo > self.alert_threshold
        rec = {
            "est_drop": est, "ci_lo": lo, "ci_hi": hi,
            "alert": bool(alert),
            "shift_index": self.mapper.shift_index(s),
            "stats": s,
        }
        if meta_new is not None and self.meta_ref is not None:
            rec["metadata"] = metadata_drift(self.meta_ref, meta_new)
        self.history.append(rec)
        return rec

    def false_alarm_rate(self, clean_batches):
        """Fraction of clean batches that (incorrectly) raise an alert.

        clean_batches: list of (emb, probs) drawn from the reference
        distribution — no drift. A good monitor keeps this near zero.
        """
        alarms = 0
        for emb, probs in clean_batches:
            rec = self.update(emb, probs)
            alarms += rec["alert"]
        return alarms / max(len(clean_batches), 1)
