"""Drift matrix: the kill-gate experiment (RESUMABLE).

Stages, each checkpointed to results/matrix_partial.json:
  1. canary   — build labeled canary table (fast)
  2. fit      — fit null + isotonic mapper (fast)
  3. killgate — Spearman on held-out families (fast)
  4. coverage — per-row 95% CI coverage (SLOW; resumable per-row)
  5. monitor  — false-alarm rate + clean sanity + metadata (fast)

--resume skips completed stages. Writes results/summary.json + MATRIX_DONE.
"""
import argparse
import json
import os
import numpy as np

from driftmed import (CanaryProbe, CANARY_CORRUPTIONS, HELDOUT_CORRUPTIONS,
                      DropMapper, DriftMonitor, DicomMetaSimulator,
                      metadata_drift)
from driftmed import shift as shift_mod
from driftmed.model import ModelFn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTIAL = os.path.join(ROOT, "results", "matrix_partial.json")
DONE = os.path.join(ROOT, "MATRIX_DONE")


def save(state):
    with open(PARTIAL, "w") as f:
        json.dump(state, f)


def load():
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    state = load() if args.resume else {}
    rng = np.random.default_rng(7)

    ref = np.load(os.path.join(ROOT, "results", "reference.npz"))
    emb_ref, probs_ref, y_ref = ref["emb"], ref["probs"], ref["y"]
    x_ref = ref["x"].astype(np.float32)
    n_classes = probs_ref.shape[1]
    model_fn = ModelFn(os.path.join(ROOT, "checkpoints", "model_final.pt"))
    probe = CanaryProbe(model_fn, n_classes)
    idx = rng.choice(len(x_ref), 1500, replace=False)
    xc, yc = x_ref[idx], y_ref[idx]

    mapper = DropMapper(n_bootstrap=40, seed=0)

    if "canary" not in state:
        print("stage: canary", flush=True)
        state["canary"] = probe.build_table(xc, yc, CANARY_CORRUPTIONS)
        state["heldout"] = probe.build_table(xc, yc, HELDOUT_CORRUPTIONS)
        save(state)

    if "fit" not in state:
        print("stage: fit", flush=True)
        mapper.fit_null(emb_ref, probs_ref, n_classes, n_resamples=30,
                        batch=512)
        # persist null stats (plain floats)
        state["null_mean"] = mapper.null_mean_
        state["null_std"] = mapper.null_std_
        mapper.fit(state["canary"])
        state["iso_x"] = mapper.iso_x_.tolist()
        state["iso_y"] = mapper.iso_y_.tolist()
        save(state)
    else:
        mapper.null_mean_ = state["null_mean"]
        mapper.null_std_ = state["null_std"]
        mapper.iso_x_ = np.array(state["iso_x"])
        mapper.iso_y_ = np.array(state["iso_y"])
        mapper._canary_idx = np.array([mapper.shift_index(r)
                                       for r in state["canary"]])
        mapper._canary_drop = np.array([r["acc_drop"]
                                        for r in state["canary"]])

    if "killgate" not in state:
        print("stage: killgate", flush=True)
        rho, n = mapper.evaluate(state["heldout"])
        state["killgate"] = {"spearman": rho, "n": n}
        fams = {}
        for fam in HELDOUT_CORRUPTIONS:
            rows = [r for r in state["heldout"] if r["family"] == fam]
            rrho, _ = mapper.evaluate(rows)
            fams[fam] = {
                "spearman": rrho,
                "true_drop": [round(r["acc_drop"], 4) for r in rows],
                "est_drop": [round(mapper.estimate(r), 4) for r in rows],
            }
        state["per_family"] = fams
        print(f"KILL-GATE: Spearman = {rho:.4f} (n={n})", flush=True)
        save(state)

    if "coverage" not in state:
        state["coverage"] = {"done_rows": [], "covered": 0}
    cov = state["coverage"]
    heldout = state["heldout"]
    if len(cov["done_rows"]) < len(heldout):
        print(f"stage: coverage ({len(cov['done_rows'])}/{len(heldout)})",
              flush=True)
        for i, r in enumerate(heldout):
            if i in cov["done_rows"]:
                continue
            fn, _ = HELDOUT_CORRUPTIONS[r["family"]]
            bi = rng.choice(len(x_ref), 512, replace=False)
            emb_b, pr_b = model_fn(fn(x_ref[bi], r["severity"]))
            est, lo, hi = mapper.estimate_ci(emb_b, pr_b, emb_ref,
                                             probs_ref, n_classes, batch=512)
            if lo <= r["acc_drop"] <= hi:
                cov["covered"] += 1
            cov["done_rows"].append(i)
            save(state)
            print(f"  row {i + 1}/{len(heldout)}: est={est:.4f} "
                  f"[{lo:.4f},{hi:.4f}] true={r['acc_drop']:.4f}", flush=True)

    if "monitor" not in state:
        print("stage: monitor", flush=True)
        mon = DriftMonitor(mapper, emb_ref, probs_ref, n_classes,
                           alert_threshold=0.03)
        clean = []
        for k in range(10):
            bi = rng.choice(len(emb_ref), 512, replace=False)
            clean.append((emb_ref[bi], probs_ref[bi]))
        far = mon.false_alarm_rate(clean)
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
        state["monitor"] = {
            "far": far, "clean_est": clean_est,
            "meta_same": d_same["combined"],
            "meta_shift": d_shift["combined"],
            "meta_scanner_psi": d_shift["scanner_psi"],
        }
        save(state)

    kg = state["killgate"]
    summary = {
        "kill_gate_spearman_heldout": round(kg["spearman"], 4),
        "kill_gate_n": kg["n"],
        "kill_gate_pass": bool(kg["spearman"] >= 0.7),
        "per_family": state["per_family"],
        "ci_coverage_95": round(state["coverage"]["covered"] / len(heldout), 4),
        "false_alarm_rate_clean": round(state["monitor"]["far"], 4),
        "clean_batch_est_drop": round(state["monitor"]["clean_est"], 4),
        "metadata_same_combined": round(state["monitor"]["meta_same"], 4),
        "metadata_shift_combined": round(state["monitor"]["meta_shift"], 4),
        "metadata_shift_scanner_psi": round(
            state["monitor"]["meta_scanner_psi"], 4),
        "canary_rows": len(state["canary"]),
    }
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    open(DONE, "w").write("ok\n")
    print("MATRIX DONE", flush=True)


if __name__ == "__main__":
    main()
