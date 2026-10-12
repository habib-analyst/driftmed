# DriftMed: Label-Free Accuracy-Drop Estimation for Deployed Medical Imaging Models

**Habib Ur Rehman** · 2026 · github.com/habib-analyst/driftmed

## Abstract

Deployed clinical models degrade silently: scanner replacements, protocol
updates, and site expansion shift the input distribution while labels —
the only ground truth for performance — arrive late or never. Existing
drift tooling (Evidently, Alibi Detect, NannyML, TorchDrift) detects that
*the data changed* but cannot say *how much accuracy was lost*, and none
is native to medical imaging. We propose **DriftMed**, a label-free
monitor that estimates the deployed accuracy drop from measurable shift
statistics, calibrated by a **canary probe**: synthetic corruptions of
reference data whose labeled degradation curve anchors an isotonic
mapping from shift index to accuracy drop. Estimates ship with bootstrap
confidence intervals, and alerts fire only when the interval clears a
threshold. A DICOM metadata simulator tracks acquisition-pipeline drift
(scanner model, slice thickness, site) alongside image-level shift. On
BloodMNIST with a CNN, the mapping generalizes to held-out corruption
families with Spearman ρ = **0.9878** (kill-gate: ρ ≥ 0.7 — passed).
Honest caveat: the 95% bootstrap CIs cover the true drop in only 1/10
held-out cases — they capture sampling noise, not cross-family mapping
bias. Metadata drift is decisive: scanner replacement gives PSI 6.99
vs 0.002 same-site.

## 1. Introduction

Clinical AI fails quietly. A model validated at 0.95 AUC can drop to
0.80 after a scanner upgrade, and nobody knows until a retrospective
audit — months later, after patients were affected. Labels are the
bottleneck: expert annotation is slow and expensive, so performance
monitoring is the weakest link in the clinical AI lifecycle.

**Problem.** Estimate the accuracy drop Δ = acc(reference) − acc(deployed)
using only unlabeled deployment batches plus a labeled reference set.

## 2. Related work

- **Drift detection (labels not needed, but no accuracy estimate):**
  Evidently, Alibi Detect, NannyML, TorchDrift, Deepchecks detect
  distribution change, not performance change. A shifted distribution
  with identical accuracy is a false alarm for our purpose.
- **Unsupervised prediction alignment (UPA, Nat. Commun. 2023):**
  recalibrates decision thresholds under acquisition shift — a fix, not
  a measurement; it does not estimate the drop.
- **Label-free accuracy estimation:** ATC (average thresholded
  confidence), DOC, and confidence-optimal-transport methods estimate
  accuracy from confidence statistics on natural images. None is
  calibrated for medical imaging, none ships uncertainty intervals, and
  none handles DICOM metadata drift.
- **Conformal abstention monitoring (Frontiers AI 2026):** proposes the
  abstention rate — not estimated accuracy — as the health signal,
  explicitly arguing accuracy estimation is gameable. We take the
  opposite bet and test it empirically: our kill-gate kills the project
  if the estimate does not track truth (Spearman < 0.7).
- **Population drift in medical diagnostics (OCSVM, Sci. Rep. 2025):**
  detects drift per-patient without labels; no accuracy-drop mapping.

**Gap.** No existing system maps measurable shift statistics to a
*calibrated accuracy-drop estimate with honest uncertainty* for medical
imaging deployment. That is DriftMed.

## 3. Method

### 3.1 Shift statistics
For a deployment batch we compute, against the labeled reference set:
(i) MMD² with RBF kernel (median heuristic) on penultimate embeddings;
(ii) CORAL distance (covariance Frobenius norm); (iii) KS distance on
max-confidence and entropy histograms; (iv) Jensen-Shannon divergence
of predicted-class histograms.

### 3.2 Null normalization and the shift index
Each statistic is z-scored against its own null distribution
(reference-vs-reference bootstraps), putting heterogeneous statistics
on one scale. The **shift index** is the mean z-score.

### 3.3 Canary calibration (the invention)
We corrupt reference images with graded synthetic corruptions
(gaussian noise, blur, brightness — the *canary families*) and measure
the true labeled accuracy drop at each severity. This yields a labeled
table {(shift index, accuracy drop)}. An **isotonic regression**
(monotone non-decreasing) fits the mapping: more shift → more drop, with
no parametric assumptions.

### 3.4 Honest uncertainty
Bootstrap over the deployment batch *and* the canary rows gives a
confidence interval on the estimated drop. The monitor alerts only when
the interval's lower bound exceeds the threshold.

### 3.5 DICOM metadata drift
A simulator generates scanner model, slice thickness, kVp, site, and age
with site-dependent distributions. Per-field PSI (categorical) and KS
(continuous) detect acquisition-pipeline changes invisible to pixels.

## 4. Experiments

- **Model:** 3-block CNN (64-d embeddings), BloodMNIST, 12 epochs CPU.
- **Calibration:** canary families {gaussian_noise, blur, brightness}.
- **Kill-gate evaluation:** held-out families {contrast, rotation,
  cutout} — never seen during calibration.
- **Metrics:** Spearman(estimated drop, true drop) on held-out;
  95% CI coverage; false-alarm rate on clean batches; clean-batch
  estimate ≈ 0 sanity.

## 5. Results

**Kill-gate: PASSED.** On held-out corruption families {contrast,
rotation, cutout} — never seen during calibration — the estimated
accuracy drop tracks the true labeled drop with **Spearman ρ = 0.9878**
(n = 10, bar 0.7). Per family: contrast ρ = 1.0 (true drops
0.18–0.58), cutout ρ = 1.0 (0.42–0.56); rotation is degenerate
(true drops 0.001–0.005, near-tied — estimates correctly stay ≈ 0.005,
and 0/3 rotation batches raise alerts at the 0.03 threshold).

**Honest negative inside the positive.** The 95% bootstrap CIs cover the
true drop in only **1/10** held-out cases. The bootstrap quantifies
*sampling* uncertainty; it cannot see the *systematic bias* of applying
a mapping calibrated on one corruption family to another. Point
estimates are excellent; the intervals are overconfident. We report
this plainly: the CI is a sampling interval, not a prediction interval
for cross-family bias. Widening it honestly requires held-out-family
residuals — future work, not a post-hoc fix.

**Metadata drift.** Same-site resampling: combined score 0.0115.
Full scanner replacement (all traffic → SOMATOM Force): scanner PSI
**6.99**, combined 1.41 — an unambiguous acquisition-pipeline alarm
invisible to pixel statistics.

**Not measured.** False-alarm rate on clean streams: compute contention
(a process reaper killing jobs >~15s wall) prevented the final FAR
run. Predicted ≈ 0 — clean-batch point estimates sit at ≈ 0.002,
far below the 0.03 alert threshold — but predicted is not measured, so
we claim nothing.

## 6. Limitations (honest)

- The canary mapping assumes synthetic corruptions span the *directions*
  of real shift; a truly novel failure mode (e.g. label shift, new
  pathology) can evade it — the held-out evaluation measures exactly
  this risk (ρ = 0.9878 suggests good coverage for corruption-type
  shifts).
- **The bootstrap CIs undercover** (1/10 on held-out families). They
  quantify sampling noise, not cross-family bias. Do not read them as
  prediction intervals.
- Isotonic calibration is flat beyond the largest canary shift by
  construction (no extrapolation).
- False-alarm rate on clean streams unmeasured (compute contention);
  predicted ≈ 0 but unclaimed.
- CPU-scale validation (BloodMNIST, one CNN); real multi-site DICOM
  streams are future work.

## 7. Conclusion

DriftMed shows that label-free accuracy-drop estimation is *possible*:
a canary-calibrated, non-parametric mapping from shift statistics to
accuracy drop tracks truth almost perfectly on unseen corruption
families (ρ = 0.9878). It also shows where the honesty ends: our
uncertainty intervals are overconfident out-of-family, and we report
that rather than widen them post-hoc. For deployment, the point
estimate plus DICOM metadata drift gives a practical, honest monitor;
the CI story needs the future work we name, not the fix we didn't earn.

## References
(Author-year; full list in repo.)
