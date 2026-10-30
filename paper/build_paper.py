"""
Builds paper/DiskGuard_paper_draft.docx from the text below.

Every number here is read off a results file named next to it in the source
comments; when a result changes, edit it here and re-run:

    <python with python-docx> paper/build_paper.py

Inline markup in the text: **bold**, *italic*, and [CITE: ...] / [[...]]
placeholders, which are highlighted yellow so none survive by accident.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(__file__).with_name("DiskGuard_paper_draft.docx")

# Headline row-level AUPRC interval: results/paper_tables/table_row_auprc_ci_full.csv
# (26_row_auprc_ci.py, B = 1000, full drive bootstrap).
ROW_CI = "[[CI pending: 26_row_auprc_ci.py]]"

doc = Document()

# ---------------------------------------------------------------- styles
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.5))
sec.top_margin = sec.bottom_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Title", 18)):
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    # The built-in heading styles take their font from the theme, which
    # overrides font.name; drop the theme references so Times New Roman wins.
    rfonts = st.element.rPr.rFonts
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(attr), None)
doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
doc.styles["Heading 2"].paragraph_format.space_before = Pt(10)

_TOKEN = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+`|\[CITE:[^\]]*\]|\[\[.+?\]\])")


def runs(par, text: str) -> None:
    for part in _TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("[CITE") or part.startswith("[["):
            r = par.add_run(part)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(10)
        elif part.startswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def P(text: str, align=WD_ALIGN_PARAGRAPH.JUSTIFY) -> None:
    p = doc.add_paragraph()
    p.alignment = align
    runs(p, text)


def H1(text: str) -> None:
    doc.add_heading(text, level=1)


def H2(text: str) -> None:
    doc.add_heading(text, level=2)


def bullets(items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        runs(p, it)


def numbered(items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Number")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        runs(p, it)


def _shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(caption: str, header: list[str], rows: list[list[str]], widths_cm: list[float]) -> None:
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(10)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            c.paragraphs[0].text = ""
            runs(c.paragraphs[0], val)
            for run in c.paragraphs[0].runs:
                run.font.size = Pt(9.5)
                if i == 0:
                    run.bold = True
            if i == 0:
                _shade(c, "D9D9D9")
    doc.add_paragraph()


# ===================================================================== text
title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("DiskGuard: Explainable, Agent-Assisted Hard-Drive Failure Management at True Prevalence")
P("[[Author names, affiliation, supervisor]]", align=WD_ALIGN_PARAGRAPH.CENTER)

H1("Abstract")
P("Hard-drive failures in large storage fleets are rare on any given day but frequent in aggregate, "
  "and published prediction results are often reported on downsampled data with metrics that say "
  "little about operational usefulness. We present **DiskGuard**, an end-to-end failure-management "
  "system built on more than 12 years of Backblaze SMART telemetry (681.7 million drive-days). A five-seed "
  "LightGBM ensemble over 212 SMART-derived features predicts failure within 14 days; evaluated on a "
  "15-month held-out period at true prevalence (0.047% positive drive-days, 352,383 drives), it "
  "reaches a row-level AUPRC of **0.0839** (179× the base rate) and a drive-level AUPRC of **0.598** "
  "(95% CI 0.586–0.611). Recall-calibrated risk tiers place 66.9% of failing drives in the HIGH tier "
  "at 56% precision, and 51% of failing drives are already HIGH 14 days before failure. Because no "
  "model we tested produced a trustworthy day count, urgency is expressed as a fleet ranking from an "
  "XGBoost accelerated-failure-time model (C-index 0.749) rather than a countdown. Exact TreeSHAP "
  "attributions are mapped to plain-language root causes with tier-aware actions (96.5% coverage of "
  "at-risk drives), and a tool-using language-model agent drafts maintenance tickets that a human "
  "must approve. The agent's safety properties are enforced in code and verified exhaustively: 13 "
  "invariants hold on all 34,794 evaluated drives, and the server boundary withstands 10 of 10 "
  "adversarial scenarios. More complex alternatives — a recurrent remaining-life model and fusions of "
  "LightGBM with a second model — were evaluated under pre-registered rules and did not improve on "
  "LightGBM by a material margin.")

# ------------------------------------------------------------------ 1
H1("1. Introduction")
H2("1.1 Problem")
P("Hard disk drives remain the dominant medium for bulk storage in large datacenters, and they fail. "
  "Backblaze reports annualised failure rates of roughly 1–2% across a fleet of several hundred "
  "thousand drives [CITE: Backblaze Drive Stats]. At that scale failures arrive daily; each one costs "
  "a rebuild and, while redundancy is degraded, raises the risk of data loss. Predictive maintenance "
  "— scoring every drive every day and replacing those likely to fail — is the natural response, and "
  "SMART (Self-Monitoring, Analysis and Reporting Technology) telemetry is the standard input "
  "[CITE: Botezatu et al., KDD 2016] [CITE: Lu et al., FAST 2020].")
H2("1.2 The evaluation gap")
P("Failure within a short horizon is a rare event. In our test period 0.047% of drive-days are "
  "positive (67,689 of 144,739,479). A model that always predicts \"healthy\" is 99.95% accurate, so "
  "accuracy — and, to a lesser degree, ROC-AUC — reveals little about whether the top of the ranking, "
  "where an operator will actually look, is any good [CITE: Saito & Rehmsmeier, 2015]. Results "
  "reported on class-balanced or downsampled test sets change the base rate and are not comparable "
  "with deployment. A further gap is operational: a probability alone does not tell an operator "
  "*which* drive to replace first, *why*, or *what to do*.")
H2("1.3 This work")
P("We build and evaluate a complete system that answers four operational questions for every drive "
  "in a 352,000-drive fleet: *Will it fail soon?* (a LightGBM ensemble and recall-calibrated risk "
  "tiers), *How urgent is it?* (a survival-model fleet ranking), *Why?* (exact TreeSHAP mapped to "
  "root causes) and *What should be done?* (a tool-using agent whose drafts require human approval). "
  "Every claim is evaluated at true prevalence with drive-clustered uncertainty.")
H2("1.4 Contributions")
numbered([
    "**Failure prediction at true prevalence.** A five-seed LightGBM ensemble evaluated on 144.7 million "
    "held-out drive-days without downsampling, with drive-clustered bootstrap intervals, a lead-time "
    "analysis and a calibration check.",
    "**Recall-calibrated risk tiers** fixed on validation and reported on test at both drive and "
    "drive-day level.",
    "**Urgency as a ranking, not a countdown.** We show that a survival model ranks time-to-failure "
    "well (C-index 0.749) while its absolute day estimates are unreliable, and design the urgency "
    "output around the property that holds.",
    "**Explanations with tier-aware actions.** TreeSHAP drivers are mapped to plain-language root causes; "
    "the risk tier, not the driver, sets the response deadline.",
    "**An approval-gated maintenance agent with verified guard rails.** Safety properties are enforced "
    "server-side and audited exhaustively and adversarially; the audit found, and we fixed, one rule "
    "that had been enforced only by the prompt.",
    "**Robustness evidence.** A pre-registered ablation shows that infrastructure identifiers are not "
    "needed (equivalence), and whole-datacenter hold-outs show no site overfitting.",
])

# ------------------------------------------------------------------ 2
H1("2. Related Work")
P("[[Draft to be completed — suggested structure and sources to verify:]]")
bullets([
    "**SMART-based failure prediction.** Early statistical analyses of large fleets [CITE: Pinheiro et al., FAST 2007]; "
    "learning-based prediction of disk replacement [CITE: Botezatu et al., KDD 2016]; SMART plus "
    "performance and location signals [CITE: Lu et al., FAST 2020]; remaining-life estimation on "
    "Backblaze data [CITE: Anantharaman et al., 2018].",
    "**Gradient boosting for tabular data.** LightGBM [CITE: Ke et al., NeurIPS 2017]; XGBoost "
    "[CITE: Chen & Guestrin, KDD 2016]; Optuna for tuning [CITE: Akiba et al., KDD 2019].",
    "**Survival analysis.** Accelerated-failure-time models in XGBoost [CITE: Barnwal et al., 2022]; "
    "Harrell's concordance index [CITE: Harrell et al., 1982].",
    "**Evaluation under class imbalance.** Precision–recall vs ROC [CITE: Saito & Rehmsmeier, 2015].",
    "**Explainability.** SHAP and exact TreeSHAP [CITE: Lundberg & Lee, NeurIPS 2017] [CITE: Lundberg et al., Nat. Mach. Intell. 2020].",
    "**Tool-using language-model agents** and human-in-the-loop operations [CITE: e.g. Yao et al., ReAct, ICLR 2023].",
])

# ------------------------------------------------------------------ 3
H1("3. Data")
H2("3.1 Source and label")
P("We use the public Backblaze Drive Stats dataset, which publishes one record per drive per day with "
  "SMART attributes, model and capacity, and a failure flag [CITE: Backblaze Drive Stats]. We combine "
  "all quarters from April 2013 to December 2025. The target `failure_within_14d` is 1 when a "
  "drive's recorded failure date lies **1 to 14 days** after the record's date. The failure day "
  "itself is labelled 0, so the model is rewarded only for advance warning, and surviving (censored) "
  "drives are always 0. Fourteen days matches the timescale on which a replacement can be scheduled.")
H2("3.2 Temporal splits")
P("Data are split by time, never at random, so that the model is always evaluated on a future it "
  "has not seen (Table 1). Validation and test keep the real failure rate. Only the training set is "
  "rebalanced: every positive row is kept and negatives are undersampled to ten per positive.")
table("Table 1. Temporal splits.",
      ["Split", "Dates", "Drive-days", "Use"],
      [["Train", "2013-04-10 → 2024-03-31", "484,139,844 (3,440,506 after undersampling; 313,002 positive)", "model fitting"],
       ["Validation", "2024-04-01 → 2024-09-30", "52,845,256", "tuning, threshold and tier cutoffs"],
       ["Test", "2024-10-01 → 2025-12-31", "144,739,479 (352,383 drives)", "all reported results"]],
      [2.2, 4.2, 6.0, 3.6])
P("In the test period 5,254 of 352,383 drives (1.49%) fail within the horizon, and 0.0468% of "
  "drive-days are positive. The drive-day rate is the relevant base rate for daily ranking.")
H2("3.3 Features")
P("The model uses **212 features**: 83 raw and 83 normalised SMART attributes, 36 rolling-window "
  "means and standard deviations (7, 14 and 30 days), 6 day-over-day deltas, and four derived "
  "features — drive age, power-on years, a combined error total and a temperature-stress indicator. "
  "Features are computed on each drive's full history before splitting, so rolling windows at the "
  "start of a split use real earlier readings; all windows look backwards only. Missing values are "
  "imputed with training-period medians. Five infrastructure identifiers (vault, pod, cluster, pod "
  "slot and a legacy-format flag) are excluded; Section 6.3 tests that choice.")

# ------------------------------------------------------------------ 4
H1("4. System Design")
P("DiskGuard separates *prediction* (frozen model outputs) from *action* (decisions an operator "
  "approves). Figure 1 [[insert architecture diagram]] shows the flow: daily SMART records → LightGBM "
  "score → risk tier → fleet urgency and SHAP root cause → agent draft → human approval.")
H2("4.1 Failure prediction: LightGBM ensemble")
P("LightGBM [CITE: Ke et al., NeurIPS 2017] is tuned with Optuna [CITE: Akiba et al., KDD 2019] "
  "(30 trials, TPE sampler, seed 42) to maximise AUPRC on a fixed 5,000,000-row prefix of the "
  "validation split; AUPRC is the objective because the task is ranking. The selected configuration "
  "uses 798 trees, learning rate 0.022, 53 leaves, maximum depth 10, row and column subsampling of "
  "0.70 and 0.90, and L1/L2 penalties of 0.13 and 4.96. The deployed model averages five boosters "
  "trained with seeds 42, 7, 13, 99 and 23. A binary alert threshold of 0.62 minimises expected cost "
  "on validation with a missed failure costed at 100 false alarms.")
H2("4.2 Recall-calibrated risk tiers")
P("Operators work from tiers, not raw scores. Cutoffs are set on validation so that the HIGH tier "
  "captures 50% of failing drive-days and HIGH plus MEDIUM captures 80%, giving HIGH ≥ 0.764 and "
  "MEDIUM ≥ 0.069. The cutoffs are frozen and applied unchanged to test.")
H2("4.3 Fleet urgency from a survival model")
P("Operators also need to know which drive to replace first. We fit an XGBoost accelerated-failure-time "
  "model [CITE: Barnwal et al., 2022] on the same features with interval-censored labels: a failing "
  "drive contributes its observed remaining life, and a surviving drive contributes a lower bound "
  "equal to its remaining observed time in the split. Its absolute day estimates proved unreliable "
  "(Section 5.3), so the system reports a ranking instead: **fleet urgency** is the share of a "
  "reference fleet (each validation drive's latest prediction) that the model expects to outlive the "
  "drive. A drive at 92% is expected to fail sooner than 92% of the fleet. By design, urgency orders "
  "work *within* a tier and never moves a drive to a higher tier.")
H2("4.4 Explanations and tier-aware actions")
P("Exact TreeSHAP [CITE: Lundberg et al., 2020] attributes each at-risk drive's LightGBM score to its "
  "features. The strongest risk-increasing feature (rolling and delta variants mapped to their base "
  "attribute) is looked up in a curated table of root causes and actions. The tier sets the timescale "
  "and the driver sets the kind of action: for a strong indicator such as reallocated sectors, a HIGH "
  "drive is scheduled for replacement within 7 days, while a MEDIUM drive is planned for the next "
  "maintenance window with increased SMART polling. Vendor-specific attributes (for example Toshiba "
  "attribute 226, head load-in time) carry a note that part of the signal may reflect the drive "
  "family rather than wear on the individual unit.")
H2("4.5 Approval-gated maintenance agent")
P("A tool-using language-model agent receives the frozen prediction as immutable evidence and may "
  "call three simulated tools: a spare-parts check, a technician-availability check and a ticket "
  "draft. It never executes an action; every ticket is a draft that requires operator approval. "
  "LOW-tier drives never reach the model. Safety properties are enforced by the server, not only by "
  "the prompt:")
bullets([
    "ticket identity, tier and root cause are attached by the server, whatever the model supplies;",
    "the parts lookup is locked to the drive's own model;",
    "a non-HIGH drive cannot be scheduled as urgent (the request is capped to routine and the model is told);",
    "tool inputs are validated, at most three tools run per turn, and runs are bounded in turns and time.",
])
P("A deterministic rule policy with identical tools serves as a no-network fallback and as the "
  "reference for evaluation. The dashboard (Streamlit, backed by a compact DuckDB serving database) "
  "presents the fleet overview, a per-drive explorer and the agent workspace.")

# ------------------------------------------------------------------ 5
H1("5. Evaluation Protocol")
H2("5.1 Metrics and uncertainty")
P("The primary metric is **row-level (drive-day) AUPRC** on the full test split, reported with lift "
  "over prevalence. Accuracy is not reported. Drive-level AUPRC scores each drive's latest record "
  "against whether it ever fails within the horizon; it measures detection near the end of a drive's "
  "observed life and is reported as secondary. Rows of one drive are not independent, so every "
  "interval resamples **physical drives**, never rows. Row-level intervals use an exact weighted "
  "average-precision bootstrap over all 352,383 drives (B = 1000), in which a drive drawn *m* times "
  "contributes each of its rows with multiplicity *m*.")
H2("5.2 Lead time and calibration")
P("For each failing test drive we check whether it was already in the HIGH tier 30, 14, 7 and 1 days "
  "before failure, among drives observed at that lead time. Calibration is summarised by the Brier "
  "score and the expected calibration error (ECE) over ten equal-width bins.")
H2("5.3 Pre-registered studies")
P("Two studies fixed their decision rules before any result was seen, with deviations logged: an "
  "ablation of infrastructure identifiers, and an evaluation of model fusion (Section 6.5). Both use "
  "a materiality margin of ±0.003 AUPRC and paired drive-clustered bootstraps (B = 1000).")
H2("5.4 Agent evaluation")
P("One set of invariants (Table 6) is applied to three decision-makers: the rule policy, a scripted "
  "adversarial model that replaces the language-model API, and the live model. A regular-expression "
  "detector flags any stated time-to-failure (\"fails in N days\", \"N days left\") while allowing "
  "scheduling language and data-age notes; it was unit-tested on positive and negative phrases.")

# ------------------------------------------------------------------ 6
H1("6. Results")
H2("6.1 Failure prediction")
table("Table 2. Test-set performance (true prevalence).",
      ["Metric", "LightGBM ensemble", "Note"],
      [["Row-level AUPRC", f"**0.0839**  {ROW_CI}", "179× prevalence (0.000468)"],
       ["Row-level ROC-AUC", "0.889", ""],
       ["Drive-level AUPRC", "**0.598** [0.586, 0.611]", "5,254 failing of 352,383 drives"],
       ["Drive-level ROC-AUC", "0.920 [0.915, 0.925]", ""],
       ["Single-seed AUPRC (5 seeds)", "0.0804 – 0.0828", "ensemble adds +0.002 over the seed mean"],
       ["Logistic regression baseline", "AUPRC 0.018, ROC-AUC 0.772", "same split"],
       ["Alert threshold 0.62", "precision 2.7%, recall 59.9%", "cost-optimal at 100:1 on validation"]],
      [4.5, 5.5, 6.0])
P("The ensemble ranks failing drive-days 179 times better than chance and 4.6 times better than "
  "logistic regression. Ensembling five seeds is worth +0.002 AUPRC over the average single seed — "
  "comparable to the effects studied in Section 6.4, which is why every comparison holds the ensemble "
  "fixed. The most influential features (mean |SHAP|) are raw read error rate (`smart_1_raw`), the "
  "combined error total, normalised SMART 18, normalised SMART 1 and raw start/stop count (`smart_4_raw`).")
H2("6.2 Risk tiers and lead time")
table("Table 3. Risk tiers on test (cutoffs fixed on validation).",
      ["Tier", "Drives flagged", "Failing drives caught", "Precision", "Share of failing drive-days"],
      [["HIGH (≥ 0.764)", "6,282", "3,516 (66.9%)", "56.0%", "52.3%"],
       ["MEDIUM (≥ 0.069)", "26,512", "824 (15.7%)", "3.1%", "23.2%"],
       ["HIGH + MEDIUM", "32,794", "4,340 (82.6%)", "13.2%", "75.5%"],
       ["LOW", "319,589", "914 (17.4%)", "0.3%", "24.5%"]],
      [3.2, 2.6, 3.6, 2.2, 4.4])
table("Table 4. Lead time: failing drives already in the HIGH tier before failure.",
      ["Days before failure", "30", "14", "7", "1"],
      [["Share in HIGH", "43.0%", "50.7%", "54.9%", "63.5%"],
       ["Failing drives observed", "4,764", "4,915", "4,939", "5,009"]],
      [4.5, 2.8, 2.8, 2.8, 2.8])
P("The HIGH tier is small (1.8% of drives) and precise enough to act on directly: more than half of "
  "its drives fail within the horizon. Half of all failing drives are flagged HIGH two weeks before "
  "they fail. The tiers were calibrated to capture 50% and 80% of failing drive-days on validation and "
  "capture 52.3% and 75.5% on test, so the calibration transfers to a later period. The 17.4% of "
  "failing drives that remain LOW reflect a known limit of SMART: many failures give no warning.")
H2("6.3 Urgency ranking")
P("On each test drive's first observation (352,197 drives, 5,341 failing), the survival model "
  "achieves a concordance index of **0.749**: for a random comparable pair it orders time-to-failure "
  "correctly three times in four. Its absolute estimates are not usable, however — on failing drives "
  "within 90 days of failure its mean absolute error is about 1,021 days, because roughly 85% of its "
  "training rows are censored and it learns that drives typically survive for years. Expressed as "
  "fleet urgency, the ranking agrees with the classifier's tiers without being redundant: mean "
  "urgency is 99.4% in HIGH, 86.0% in MEDIUM and 53.9% in LOW. Within a tier it orders work; it "
  "never moves a drive between tiers.")
H2("6.4 Robustness")
P("**Infrastructure identifiers (pre-registered).** Restoring the five identifiers changes row-level "
  "AUPRC by −0.00014 (95% CI −0.0025 to +0.0022). The interval lies within the ±0.003 margin, so the "
  "pre-registered verdict is *equivalence*: the model does not depend on where a drive is installed.")
table("Table 5. Whole-datacenter hold-out (3,000,000-row scoring samples; lift = AUPRC / prevalence).",
      ["Fold", "Held-out site", "AUPRC", "Lift", "Size-matched control lift"],
      [["phx1", "held out", "0.0826", "216", "90"],
       ["sac0", "held out", "0.0650", "141", "90"],
       ["sac2", "held out", "0.0807", "126", "80"],
       ["Reference", "all sites", "0.0448", "102", "—"]],
      [2.4, 3.0, 2.4, 2.4, 5.8])
P("**Portability.** Holding out an entire datacenter does not degrade ranking: each held-out site "
  "scores above its size-matched control, which separates distribution shift from simply having less "
  "training data. The test covers the 68% of drives whose site is published.")
P("**Calibration.** Brier score 0.0107; ECE 0.040. Scores are ranking scores and should not be read "
  "as probabilities; decisions use the tiers.")
H2("6.5 Alternatives evaluated")
P("LightGBM was compared, under pre-registered decision rules, with a recurrent (LSTM) "
  "remaining-useful-life model and with combinations of LightGBM and a second model — the LSTM's "
  "estimate, the survival model above, and a separately tuned XGBoost classifier. None improved on "
  "LightGBM by the ±0.003 AUPRC margin: the LSTM fusion and the survival-model fusion were worse "
  "(−0.016 and −0.039 AUPRC on the validation screen), and the XGBoost fusion, confirmed on the full "
  "test set, gained only +0.0004 (95% CI +0.0002 to +0.0006). A model-free check explains why: within "
  "bands of LightGBM's own score the LSTM's estimate ranks failures barely better than chance, and a "
  "second boosted model on the same features largely agrees with LightGBM. The deployed system "
  "therefore uses LightGBM alone for prediction.")
H2("6.6 Agent evaluation")
table("Table 6. Agent invariants on the rule policy (all 32,794 at-risk drives and 2,000 LOW drives).",
      ["Invariant", "Applicable", "Failures"],
      [["Prediction returned unchanged", "34,794", "0"],
       ["LOW drive gets no tools and no ticket", "2,000", "0"],
       ["Ticket identity and tier are server-owned", "32,794", "0"],
       ["Ticket is a draft requiring approval", "32,794", "0"],
       ["Parts lookup locked to the drive's model", "6,282", "0"],
       ["Every at-risk drive gets a ticket", "32,794", "0"],
       ["MEDIUM drive never scheduled as urgent", "26,512", "0"],
       ["MEDIUM action carries no fixed deadline", "26,512", "0"],
       ["No time-to-failure claimed", "34,794", "0"],
       ["Run completes", "34,794", "0"],
       ["HIGH scheduled urgent; parts checked iff HIGH; action equals rule", "6,282 / 32,794 / 32,794", "0"]],
      [9.0, 4.0, 3.0])
P("**Adversarial evaluation.** A scripted model attempted ten attacks: tickets for another drive or "
  "tier, a parts lookup for another model, an unknown tool (`delete_drive_data`), invalid tool input, "
  "escalating a MEDIUM drive, claiming a day count, tool fan-out and loops, an API outage and a "
  "truncated response. The server boundary held in all ten. The first run exposed one gap — "
  "escalation of a MEDIUM drive was prevented only by the prompt in live mode — which was fixed by "
  "capping the request in code; all ten scenarios held afterwards. **Explanation coverage:** 96.5% of "
  "at-risk drives receive a specific root cause; the rest fall back to a generic review message. "
  "[[Live-model evaluation: run 25_agent_eval.py live and report, or state as future work.]]")

# ------------------------------------------------------------------ 7
H1("7. Discussion and Limitations")
P("**A strong single model was the right choice.** On rich SMART features at true prevalence, a tuned "
  "and ensembled LightGBM was not improved by any of the more complex designs we evaluated. Our "
  "contribution is less a new predictor than an honest, complete system around a strong one, with "
  "each component's claims measured.")
P("**Urgency without a countdown.** Operators ask \"how many days are left?\", but neither model we "
  "trained could answer reliably. Presenting a ranking is less familiar but defensible: it uses only "
  "the property that was shown to hold.")
P("**Guard rails belong in code.** Prompt instructions are not a sufficient control: the one gap the "
  "audit found was a rule stated in the prompt but not enforced by the server.")
P("**Limitations.**")
bullets([
    "Scores are ranking scores, not calibrated probabilities (ECE 0.040).",
    "The evaluation is retrospective; some drives' latest assessments are months old.",
    "SMART has a ceiling: 17.4% of failing drives stay in the LOW tier.",
    "Failure labels come from operational removals; a drive removed for another reason may be mislabelled.",
    "The fleet is one operator's — high-capacity, cost-optimised, few sites — and may not represent others.",
    "Compute limits: an undersampled training set and tuning on a 5M-row validation prefix; the agent's tools are simulated.",
])

# ------------------------------------------------------------------ 8
H1("8. Conclusion")
P("DiskGuard shows that useful, explainable and safe failure management can be built around a single "
  "well-evaluated gradient-boosting model. At true prevalence it ranks failing drive-days 179 times "
  "better than chance, flags half of failing drives two weeks ahead in a tier precise enough to act "
  "on, orders urgent work with a survival-model ranking, explains each alert in plain language and "
  "drafts maintenance that a human approves under guard rails verified exhaustively and "
  "adversarially. Future work: a live-model agent study, a fixed-lead-time comparison of model "
  "families, and validation on a second operator's fleet.")

H1("References")
P("[[To be compiled from the [CITE] placeholders; verify every entry against the original source.]]")

H1("Appendix A. Reproducibility")
P("All stages are scripted; the repository README and RUNBOOK give the commands, and "
  "requirements-lock.txt pins the tested environment. Evaluated-but-not-adopted models are archived "
  "with their records. Results files backing each table: Table 2 — results/lgbm_metrics.csv, "
  "results/paper_tables/; Tables 3–4 — results/paper_tables/table_tier_accuracy.csv, "
  "results/tier_accuracy_test.csv, results_ablation/leadtime_curve.csv; Table 5 — "
  "results/datacenter_holdout.csv; Table 6 — results/agent_eval/.", align=WD_ALIGN_PARAGRAPH.LEFT)

doc.save(OUT)
print(f"[SAVE] {OUT}")
