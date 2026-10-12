"""DICOM-aware metadata drift: scanner model, slice thickness, site.

Deployed medical models degrade when the acquisition pipeline changes
(scanner replacement, protocol updates, new site) even when pixels look
fine. driftmed tracks metadata drift alongside image-level shift.
"""
import numpy as np


class DicomMetaSimulator:
    """Simulates DICOM metadata for a multi-site deployment.

    Fields: scanner_model (categorical), slice_thickness_mm (continuous),
    kvp (continuous), site_id (categorical), patient_age (continuous).
    """

    SCANNERS = ["Revolution CT", "Aquilion ONE", "SOMATOM Force",
                "Discovery CT750"]
    SITES = ["site_A", "site_B", "site_C"]

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample(self, n, site_mix=None, scanner_shift=None,
               thickness_shift=0.0):
        """site_mix: dict site->prob (default uniform). scanner_shift:
        dict scanner->prob override. thickness_shift: additive mm."""
        rng = self.rng
        sites = list(self.SITES)
        if site_mix is None:
            site_mix = {s: 1 / len(sites) for s in sites}
        site_ids = rng.choice(sites, n,
                              p=[site_mix[s] for s in sites])
        scanners = list(self.SCANNERS)
        if scanner_shift is None:
            scanner_shift = {s: 1 / len(scanners) for s in scanners}
        scanner = rng.choice(scanners, n,
                             p=[scanner_shift[s] for s in scanners])
        thickness = rng.normal(1.25 + thickness_shift, 0.25, n).clip(0.5, 5.0)
        kvp = rng.choice([100, 120, 140], n, p=[0.2, 0.6, 0.2])
        age = rng.normal(55, 15, n).clip(18, 95)
        return {
            "scanner_model": scanner,
            "slice_thickness_mm": thickness,
            "kvp": kvp,
            "site_id": site_ids,
            "patient_age": age,
        }


def _psi_categorical(ref, new, categories):
    """Population Stability Index for a categorical field."""
    ref = np.asarray(ref)
    new = np.asarray(new)
    psi = 0.0
    for c in categories:
        p = max((ref == c).mean(), 1e-4)
        q = max((new == c).mean(), 1e-4)
        psi += (q - p) * np.log(q / p)
    return float(psi)


def _ks(a, b):
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def metadata_drift(meta_ref, meta_new):
    """Per-field drift + combined score.

    Returns dict: {scanner_psi, site_psi, thickness_ks, kvp_psi, age_ks,
    combined}. Combined = mean of the five normalized-ish scores
    (PSI and KS are both on roughly [0, ~1] for moderate drift).
    """
    out = {}
    out["scanner_psi"] = _psi_categorical(
        meta_ref["scanner_model"], meta_new["scanner_model"],
        DicomMetaSimulator.SCANNERS)
    out["site_psi"] = _psi_categorical(
        meta_ref["site_id"], meta_new["site_id"],
        DicomMetaSimulator.SITES)
    out["thickness_ks"] = _ks(meta_ref["slice_thickness_mm"],
                              meta_new["slice_thickness_mm"])
    out["kvp_psi"] = _psi_categorical(meta_ref["kvp"], meta_new["kvp"],
                                      [100, 120, 140])
    out["age_ks"] = _ks(meta_ref["patient_age"], meta_new["patient_age"])
    out["combined"] = float(np.mean(list(out.values())))
    return out
