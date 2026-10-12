import numpy as np
import pytest

from driftmed import shift as S
from driftmed.mapper import DropMapper, _isotonic_fit, _isotonic_predict, _spearman
from driftmed.metadata import DicomMetaSimulator, metadata_drift
from driftmed.monitor import DriftMonitor
from driftmed.canary import gaussian_noise, blur, brightness

rng = np.random.default_rng(42)


def test_mmd_identical_near_zero():
    X = rng.normal(0, 1, (300, 16))
    Y = rng.normal(0, 1, (300, 16))
    assert S.mmd_rbf(X, Y) < 0.01


def test_mmd_shifted_positive():
    X = rng.normal(0, 1, (300, 16))
    Y = rng.normal(1.5, 1, (300, 16))
    assert S.mmd_rbf(X, Y) > 0.1


def test_coral_identical_near_zero():
    X = rng.normal(0, 1, (400, 8))
    assert S.coral_distance(X, X + 1e-9 * rng.normal(0, 1, X.shape)) < 1e-6


def test_ks_identical_near_zero():
    a = rng.normal(0, 1, 500)
    assert S.ks_distance(a, a) == 0.0


def test_ks_shifted():
    a = rng.normal(0, 1, 500)
    b = rng.normal(2, 1, 500)
    assert S.ks_distance(a, b) > 0.5


def test_confidence_stats_shapes():
    p = rng.dirichlet(np.ones(4), 100)
    c, e = S.confidence_stats(p)
    assert c.shape == (100,) and e.shape == (100,)
    assert (c >= 0.25).all() and (e >= 0).all()


def test_pred_hist_shift_identical():
    p = rng.dirichlet(np.ones(3), 200)
    assert S.prediction_hist_shift(p, p, 3) < 1e-9


def test_isotonic_monotone():
    x = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    y = np.array([0.0, 0.2, 0.1, 0.5, 0.4])
    xs, ys = _isotonic_fit(x, y)
    assert (np.diff(ys) >= -1e-12).all()
    q = _isotonic_predict(xs, ys, np.array([0.0, 0.4, 1.0]))
    assert (np.diff(q) >= -1e-12).all()


def test_spearman_perfect():
    assert abs(_spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(_spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def _fake_mapper():
    m = DropMapper(n_bootstrap=20, seed=1)
    emb = rng.normal(0, 1, (600, 8))
    probs = rng.dirichlet(np.ones(3), 600)
    m.fit_null(emb, probs, 3, n_resamples=10, batch=128)
    rows = [{"acc_drop": 0.01 * i, "mmd": 0.01 * i, "coral": 0.01 * i,
             "conf_ks": 0.01 * i, "entropy_ks": 0.01 * i, "pred_js": 0.001 * i}
            for i in range(1, 11)]
    m.fit(rows)
    return m, emb, probs


def test_mapper_estimate_monotone():
    m, _, _ = _fake_mapper()
    e1 = m.estimate({"mmd": 0.02, "coral": 0.02, "conf_ks": 0.02,
                     "entropy_ks": 0.02, "pred_js": 0.002})
    e2 = m.estimate({"mmd": 0.08, "coral": 0.08, "conf_ks": 0.08,
                     "entropy_ks": 0.08, "pred_js": 0.008})
    assert e2 >= e1


def test_mapper_ci_ordering():
    m, emb, probs = _fake_mapper()
    est, lo, hi = m.estimate_ci(emb[:200], probs[:200], emb, probs, 3,
                                batch=64)
    assert lo <= est <= hi
    assert hi - lo >= 0


def test_mapper_evaluate_spearman():
    m, _, _ = _fake_mapper()
    rows = [{"acc_drop": 0.01 * i, "mmd": 0.01 * i, "coral": 0.01 * i,
             "conf_ks": 0.01 * i, "entropy_ks": 0.01 * i, "pred_js": 0.001 * i}
            for i in range(1, 8)]
    rho, n = m.evaluate(rows)
    assert rho > 0.95 and n == 7


def test_metadata_no_shift_small():
    sim = DicomMetaSimulator(seed=0)
    a = sim.sample(800)
    b = sim.sample(800)
    d = metadata_drift(a, b)
    assert d["combined"] < 0.15


def test_metadata_scanner_shift_detected():
    sim = DicomMetaSimulator(seed=0)
    a = sim.sample(800)
    override = {s: (1.0 if s == "SOMATOM Force" else 0.0)
                for s in sim.SCANNERS}
    b = sim.sample(800, scanner_shift=override)
    d = metadata_drift(a, b)
    assert d["scanner_psi"] > 0.5


def test_metadata_thickness_shift():
    sim = DicomMetaSimulator(seed=0)
    a = sim.sample(800)
    b = sim.sample(800, thickness_shift=1.5)
    d = metadata_drift(a, b)
    assert d["thickness_ks"] > 0.5


def test_corruptions_change_pixels():
    x = rng.uniform(0, 1, (8, 28, 28, 1))
    assert not np.allclose(gaussian_noise(x, 3), x)
    assert not np.allclose(blur(x, 2), x)
    assert not np.allclose(brightness(x, 2), x)


def test_monitor_clean_no_alert():
    m, emb, probs = _fake_mapper()
    mon = DriftMonitor(m, emb, probs, 3, alert_threshold=0.05)
    batches = [(emb[i:i + 100], probs[i:i + 100]) for i in range(0, 300, 100)]
    far = mon.false_alarm_rate(batches)
    assert far == 0.0


def test_monitor_alerts_on_massive_shift():
    m, emb, probs = _fake_mapper()
    mon = DriftMonitor(m, emb, probs, 3, alert_threshold=0.001)
    emb_bad = emb + 10.0
    rec = mon.update(emb_bad[:200], probs[:200])
    assert rec["alert"] is True
    assert rec["ci_lo"] <= rec["est_drop"] <= rec["ci_hi"]
