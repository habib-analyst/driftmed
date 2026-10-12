"""One FAR batch per invocation; appends to results/far_progress.json."""
import json, os, sys
import numpy as np
from driftmed.mapper import DropMapper
from driftmed import DriftMonitor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = int(sys.argv[1])
st = json.load(open(os.path.join(ROOT, "results", "matrix_partial.json")))
mapper = DropMapper(n_bootstrap=15, seed=0)
mapper.null_mean_ = st["null_mean"]; mapper.null_std_ = st["null_std"]
mapper.iso_x_ = np.array(st["iso_x"]); mapper.iso_y_ = np.array(st["iso_y"])
mapper._canary_idx = np.array([mapper.shift_index(r) for r in st["canary"]])
mapper._canary_drop = np.array([r["acc_drop"] for r in st["canary"]])
ref = np.load(os.path.join(ROOT, "results", "reference.npz"))
emb_ref, probs_ref = ref["emb"], ref["probs"]; n_classes = probs_ref.shape[1]
rng = np.random.default_rng(7)
batches = [rng.choice(len(emb_ref), 512, replace=False) for _ in range(10)]
bi = batches[k]
mon = DriftMonitor(mapper, emb_ref, probs_ref, n_classes, alert_threshold=0.03)
rec = mon.update(emb_ref[bi], probs_ref[bi])
prog_path = os.path.join(ROOT, "results", "far_progress.json")
prog = json.load(open(prog_path)) if os.path.exists(prog_path) else []
prog.append({"k": k, "alert": bool(rec["alert"]), "est": round(float(rec["est_drop"]), 4)})
json.dump(prog, open(prog_path, "w"))
print(f"batch {k}: alert={rec['alert']} est={rec['est_drop']:.4f}", flush=True)
