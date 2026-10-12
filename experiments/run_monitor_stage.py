"""Monitor stage only: false-alarm rate + clean sanity + metadata.
Loads fitted mapper from results/matrix_partial.json, appends 'monitor'.
"""
import json
import os
import numpy as np

from driftmed.mapper import DropMapper
from driftmed import DriftMonitor, DicomMetaSimulator, metadata_drift
from driftmed import shift as shift_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTIAL = os.path.join(ROOT, "results", "matrix_partial.json")


def main():
    st = json.load(open(PARTIAL))
    mapper = DropMapper(n_bootstrap=40, seed=0)
    mapper.null_mean_ = st["null_mean"]
    mapper.null_std_ = st["null_std"]
    mapper.iso_x_ = np.array(st["iso_x"])
    mapper.iso_y_ = np.array(st["iso_y"])
    mapper._canary_idx = np.array([mapper.shift_index(r)
                                   for r in st["canary"]])
    mapper._canary_drop = np.array([r["acc_drop"] for r in st["canary"]])

    ref = np.load(os.path.join(ROOT, "results", "reference.npz"))
    emb_ref, probs_ref = ref["emb"], ref["probs"]
    n_classes = probs_ref.shape[1]
    rng = np.random.default_rng(7)

    mon = DriftMonitor(mapper, emb_ref, probs_ref, n_classes,
                       alert_threshold=0.03)
    clean = []
    for k in range(10):
        bi = rng.choice(len(emb_ref), 512, replace=False)
        clean.append((emb_ref[bi], probs_ref[bi]))
    far = mon.false_alarm_rate(clean)
    print(f"false_alarm_rate: {far}", flush=True)

    bi = rng.choice(len(emb_ref), 512, replace=False)
    stats = shift_mod.shift_statistics(emb_ref, emb_ref[bi], probs_ref,
                                       probs_ref[bi], n_classes)
    clean_est = mapper.estimate(stats)

    sim = DicomMetaSimulator(seed=0)
    m_ref = sim.sample(1000)
    m_same = sim.sample(1000)
    m_shift = sim.sample(1000, scanner_shift={"SOMATOM Force": 1.0,
                                              "Revolution CT": 0.0,
                                              "Aquilion ONE": 0.0,
                                              "Discovery CT750": 0.0})
    d_same = metadata_drift(m_ref, m_same)
    d_shift = metadata_drift(m_ref, m_shift)

    st["monitor"] = {
        "far": far, "clean_est": clean_est,
        "meta_same": d_same["combined"],
        "meta_shift": d_shift["combined"],
        "meta_scanner_psi": d_shift["scanner_psi"],
    }
    json.dump(st, open(PARTIAL, "w"))
    print(f"clean_est={clean_est:.4f} meta_same={d_same['combined']:.4f} "
          f"meta_shift={d_shift['combined']:.4f} "
          f"scanner_psi={d_shift['scanner_psi']:.4f}", flush=True)
    print("MONITOR STAGE DONE", flush=True)


if __name__ == "__main__":
    main()
