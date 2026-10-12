# driftmed

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/habib-analyst/driftmed)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/driftmed.svg)](https://github.com/habib-analyst/driftmed)

**Your deployed medical model is degrading and nobody told you.** DriftMed estimates *how much accuracy you've lost* — with no labels, from the deployment stream alone — and tells you how sure it is.

Existing drift tools (Evidently, Alibi Detect, NannyML) tell you the *data* changed. They can't tell you the *accuracy* changed. DriftMed's invention: a **canary probe** — synthetic corruptions of your reference data with known labeled degradation — calibrates a mapping from measurable shift statistics to estimated accuracy drop. At deployment, unlabeled shift statistics go in, an estimated drop with a confidence interval comes out.

## Quickstart

```bash
pip install driftmed
```

```python
from driftmed import CanaryProbe, DropMapper, DriftMonitor, CANARY_CORRUPTIONS
from driftmed.model import ModelFn  # or bring your own: images -> (embeddings, probs)

model_fn = ModelFn("checkpoints/model_final.pt")

# 1. Calibrate on your labeled reference set (once)
probe = CanaryProbe(model_fn, n_classes=8)
canary = probe.build_table(x_ref, y_ref, CANARY_CORRUPTIONS)
mapper = DropMapper().fit_null(emb_ref, probs_ref, n_classes).fit(canary)

# 2. Monitor the deployment stream (no labels needed)
monitor = DriftMonitor(mapper, emb_ref, probs_ref, n_classes,
                       alert_threshold=0.03)
for emb_batch, prob_batch in deployment_stream:
    rec = monitor.update(emb_batch, prob_batch)
    if rec["alert"]:
        print(f"DRIFT: est. accuracy drop {rec['est_drop']:.3f} "
              f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]")
```

Real output on the BloodMNIST validation (held-out corruption families):

```
KILL-GATE: Spearman(est, true) on held-out = 0.9878 (n=10)
```

## Measured results

BloodMNIST, 3-block CNN (val acc 0.73), CPU. Calibrated on
{gaussian_noise, blur, brightness}; evaluated on **held-out** families
{contrast, rotation, cutout} — never seen during calibration.

| Metric | Result |
|---|---|
| Kill-gate: Spearman(estimated drop, true drop), held-out | **0.9878** (bar: 0.7) ✅ |
| 95% CI coverage of true drop, held-out | **1/10** — honest negative: CIs undercover cross-family |
| Near-zero-drop batches alerting (threshold 0.03) | **0/3** |
| False-alarm rate on clean batches | unmeasured (compute contention) — predicted ≈ 0, unclaimed |
| DICOM metadata: scanner-replacement PSI | **6.99** (same-site: 0.002) |
| Tests | **18/18** green |

## How it works

1. **Shift statistics** — MMD (RBF, median heuristic) + CORAL on penultimate embeddings; KS distance on confidence/entropy histograms; JS divergence of predicted-class histograms.
2. **Null normalization** — each statistic z-scored against its own reference-vs-reference bootstrap null → one comparable **shift index**.
3. **Canary calibration** — isotonic regression (monotone, non-parametric) from shift index to accuracy drop on the labeled canary table.
4. **Honest uncertainty** — bootstrap over deployment batch *and* canary rows → CI on the estimate. Alerts fire only when the CI's lower bound clears the threshold.
5. **DICOM metadata drift** — PSI/KS per field (scanner model, slice thickness, kVp, site, age) via a built-in multi-site simulator; catches acquisition-pipeline changes invisible to pixels.

## Honest limitations

- The mapping is only as good as the canary table's coverage: a truly novel failure mode (label shift, new pathology) can evade synthetic corruptions — the held-out evaluation measures exactly this risk.
- Isotonic calibration is flat beyond the largest canary shift by construction (no extrapolation).
- Validated at CPU scale (BloodMNIST, one CNN); real multi-site DICOM streams are future work.

## Citation

```bibtex
@software{rehman2026driftmed,
  author = {Rehman, Habib Ur},
  title = {DriftMed: Label-Free Accuracy-Drop Estimation for Deployed Medical Imaging Models},
  year = {2026},
  url = {https://github.com/habib-analyst/driftmed}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
