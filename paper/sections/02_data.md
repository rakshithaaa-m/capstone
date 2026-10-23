# 2. Data

> DRAFT. Numbers marked `[[VERIFY]]` are not yet confirmed from the artifacts.

## 2.1 Source

We use the public Backblaze Drive Stats dataset, which publishes daily SMART
attribute readings, drive model and capacity metadata, and failure records for
its production fleet [CITE: Backblaze Drive Stats]. The release is quarterly;
we combine the quarters covering `[[VERIFY: first quarter]]` through
`[[VERIFY: last quarter]]`.

Each record is one drive on one calendar day. A drive is labelled failed on the
day Backblaze removes it from service as a failure; drives removed for other
reasons are excluded from the positive class.

## 2.2 Population and splits

| split | drive-days | note |
|---|---|---|
| training (resampled) | 3,440,506 | class-conditional resample of the training period |
| validation | 52,845,256 | used for threshold selection and Optuna tuning sample |
| **test** | **144,739,479** | held out; all headline metrics |

Test population, one row per physical drive:

| quantity | value |
|---|---|
| drives | 352,383 |
| drives that fail within the horizon | 5,254 |
| drive-level failure rate | 1.4909% |
| **drive-day positive rate** | **0.046766%** (67,689 / 144,739,479) |

The drive-day positive rate is the relevant base rate for the ranking task. It is
179× lower than the drive-level rate, and this difference is the reason a
"drive-level AUPRC" and a "row-level AUPRC" are not interchangeable (Section 3.5).

## 2.3 Label

The target is `failure_within_14d`: 1 if the drive is recorded as failed within
the next 14 days from the record's date, 0 otherwise. Records for a drive after
its failure date do not exist. The 14-day horizon matches the operational
timescale on which a replacement can be scheduled.

## 2.4 Features

Features are daily aggregates of SMART attributes and drive metadata, plus
short-window deltas. `[[VERIFY: exact count by family — raw SMART / normalised
SMART / delta / workload / static]]`.

Two feature sets are compared in the ablation (Section 3.6):

- **212 features** — the shipped set, with five infrastructure identifiers removed.
- **217 features** — the same set with `vault_id`, `pod_id`, `cluster_id`,
  `pod_slot_num` and `is_legacy_format` restored.

The identifiers describe *where* a drive is deployed, not its health. They are
constant or near-constant within a deployment cohort, so a model can use them as
a proxy for installation date and site. They are the subject of the
pre-registered ablation, not an assumption in the main model.

## 2.5 Sites

The dataset exposes five datacenter codes. Three are large enough to hold out
entirely (`phx1`, `sac0`, `sac2`); the remainder are too small for a stable
held-out estimate and are not used in the portability test.

Approximately 32% of training rows belong to drives whose site cannot be
resolved from the published metadata. These are dropped in every fold of the
portability test, which is stated as a limitation: the portability result
describes the attributable 68%, not the whole fleet.

## 2.6 Why the test split is not downsampled

Class-balanced or downsampled evaluation changes the base rate, and therefore
changes what AUPRC means. Because the operational decision is "which of 352,383
drives do we look at this week," the relevant prevalence is the true one. We
therefore evaluate on the full test split and report lift over prevalence
alongside AUPRC, so that the number can be interpreted without knowing the base
rate: AUPRC / prevalence.

## 2.7 Known limitations of the data

- Failure labels come from operational removal, not from a controlled experiment;
  a drive removed for an unrelated reason may be mislabelled.
- SMART attribute availability varies by drive model and firmware; missing values
  are imputed `[[VERIFY: imputation strategy]]`.
- Backblaze's fleet is not representative of all datacenters: it is
  high-capacity, cost-optimised, and concentrated in a small number of sites.
