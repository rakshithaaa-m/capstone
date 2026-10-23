# 1. Introduction

> DRAFT — structural only. Every number must be re-checked against `results/`
> before circulation. Citation keys in `[CITE: ...]` are placeholders.

## 1.1 Problem

Hard disk drives remain the dominant medium for cold and bulk storage in
hyperscale datacenters, and they fail. Backblaze's own fleet reporting puts the
annualised failure rate at roughly 1.3–1.4% across several hundred thousand
drives, with a lifetime rate near 1.3% [CITE: Backblaze Drive Stats, 2025].
At fleet scale that is thousands of failures per year, each carrying the cost of
a rebuild, degraded redundancy during the rebuild window, and — in replicated or
erasure-coded systems — a measurable increase in the probability of data loss
while the array is degraded.

Predictive maintenance is the obvious response: score every drive every day, and
replace the ones that look like they are about to fail before they do. The
literature contains many such systems, and most report strong results. The
difficulty is that the reported numbers are usually not comparable, and are
frequently not meaningful.

## 1.2 The evaluation gap

Failure within a fixed horizon is a rare-event problem. In the test population
used here, 0.0468% of drive-days are positive (67,689 of 144,739,479). At that
prevalence, a classifier that predicts "never fails" achieves 99.95% accuracy.
Accuracy, and to a lesser extent ROC-AUC, therefore say very little about whether
a system is useful: what matters is how well the top of the ranking is ordered,
which is what precision–recall analysis measures.

This matters for a second reason. Claims that a multi-model or "hybrid" system
improves on a strong single model are typically supported by small differences in
accuracy, F1, or ROC-AUC on downsampled or class-balanced data. Under those
protocols a genuine gain and a seed fluctuation are difficult to distinguish, and
the direction of a small difference can reverse when the evaluation is repeated
with a different row order.

## 1.3 What this paper asks

We take a strong, well-tuned gradient-boosted classifier over daily SMART and
workload features as the reference system, and ask a narrow question:

> **Does a second model add information that the classifier does not already
> have?**

We answer it in three ways: by testing five different fusion strategies, by a
model-free conditional test that asks whether the second model's score ranks
failures better than chance *within* bands of the classifier's own probability,
and by examining what the second model would be used for operationally.

The answer, in this setting, is no. The second model — a recurrent network
regressing remaining useful life — carries no information that the classifier
lacks. We report this as a measured result rather than an implementation failure,
and we show that it is consistent with an earlier observation that a tree model
on a single snapshot can match or exceed a sequence model for RUL estimation
[CITE: Anantharaman et al., 2018].

## 1.4 Contributions

1. **A controlled fusion study at true prevalence.** Five combiners are evaluated
   against the single-model reference on the same split, with a drive-clustered
   bootstrap over the difference rather than a row-level one. All five lose; the
   best loses by a margin whose confidence interval excludes zero in the wrong
   direction.

2. **A conditional test that does not depend on the fusion design.** Within
   deciles of the classifier's predicted probability, the RUL score's AUROC for
   the failure label is 0.4928 (median across deciles). A second model that is
   informative should order failures better than chance inside those bands. It
   does not.

3. **An evaluation protocol appropriate to rare events.** Row-level AUPRC with
   drive-clustered bootstrap intervals, explicit separation of row-level
   (early-warning) from drive-level (last-observation) metrics, and a stated
   prohibition on accuracy.

4. **A portability test with a size-matched control.** Performance does not
   collapse when whole datacenters are held out, and the held-out sites score
   above size-matched control folds, which separates "distribution shift" from
   "less training data."

5. **A pre-registered ablation of infrastructure identifiers.** Site, pod,
   cluster and slot identifiers are removed from the feature set. The protocol,
   including the decision rule and the non-inferiority margin, was fixed before
   the run; the outcome is reported as measured, including if it is null.

Contribution 5 is deliberately framed as an ablation, not a claim. Earlier
informal analysis suggested that removing these identifiers improved AUPRC by
9.7%; that figure compared a single-seed model against a five-seed ensemble and
is withdrawn. The matched comparison is the one reported here.

## 1.5 What this paper does not claim

We do not claim that sequence models are useless for disk failure prediction.
We claim that *this* RUL estimator, fused *these* five ways, adds nothing to
*this* classifier, on *this* fleet, and we state the scope of that claim
explicitly. We do not claim that infrastructure identifiers are never
informative; only that on this dataset, under a matched protocol, their removal
does not degrade ranking quality.

---

## Notes for revision

- `[[VERIFY]]` the Backblaze AFR range and citation year before submission.
- `[[VERIFY]]` the Anantharaman claim: confirm the exact wording of the finding
  (RF on a snapshot comparable to or better than LSTM) against the paper, and
  cite the venue correctly.
- Contribution 5 must be rewritten once the ablation finishes. If the result is
  null, state it as "removal is non-inferior" rather than "removal helps."
- The phrase "hybrid" should not appear in the title or abstract.
