"""
Builds the IEEE conference draft (A4, two-column) on the official template.

    python paper/build_paper_ieee.py   ->  paper/DiskGuard_IEEE_draft.docx

paper/ieee_template_a4.docx is the IEEE A4 conference template re-saved by Word in
transitional format (the original is Strict OOXML, which python-docx cannot open).
Only the template's own styles are used: paper title, Author, Abstract, Keywords,
Heading 1-5, Body Text, bullet list, table head, table col head, table copy,
figure caption, references.  Headings, table and figure numbers and reference
numbers come from the template's own numbering.

Inline markup in text: *italic*, **bold**, [[placeholder]] (yellow highlight),
{cite:key1,key2} -> [n], numbered in order of first citation.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

HERE = Path(__file__).parent
TEMPLATE = HERE / "ieee_template_a4.docx"
OUT = HERE / "DiskGuard_IEEE_draft.docx"
FIG = HERE / "figures"

# ------------------------------------------------------------------ references
REFS = {
    "backblaze": 'Backblaze, Inc., "Hard drive data and stats." [Online]. Available: '
                 "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data",
    "pinheiro": 'E. Pinheiro, W.-D. Weber, and L. A. Barroso, "Failure trends in a large disk drive '
                'population," in *Proc. 5th USENIX Conf. File and Storage Technologies (FAST)*, 2007, pp. 17–28.',
    "schroeder": 'B. Schroeder and G. A. Gibson, "Disk failures in the real world: What does an MTTF of '
                 '1,000,000 hours mean to you?," in *Proc. 5th USENIX Conf. File and Storage Technologies '
                 '(FAST)*, 2007, pp. 1–16.',
    "botezatu": 'M. M. Botezatu, I. Giurgiu, J. Bogojeska, and D. Wiesmann, "Predicting disk replacement '
                'towards reliable data centers," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and '
                'Data Mining*, 2016, pp. 39–48.',
    "lu": 'S. Lu, B. Luo, T. Patel, Y. Yao, D. Tiwari, and W. Shi, "Making disk failure predictions '
          'SMARTer!," in *Proc. 18th USENIX Conf. File and Storage Technologies (FAST)*, 2020, pp. 151–167.',
    "lima": 'F. D. dos Santos Lima, G. M. R. Amaral, L. G. de Moura Leite, J. P. P. Gomes, and J. de Castro '
            'Machado, "Predicting failures in hard drives with LSTM networks," in *Proc. Brazilian Conf. '
            'Intelligent Systems (BRACIS)*, 2017, pp. 222–227.',
    "coursey": 'A. Coursey, G. Nath, S. Prabhu, and S. Sengupta, "Remaining useful life estimation of hard '
               'disk drives using bidirectional LSTM networks," in *Proc. IEEE Int. Conf. Big Data*, 2021, '
               'pp. 4832–4841.',
    "tfbest": 'R. Mohapatra and S. Sengupta, "TFBEST: Dual-aspect transformer with learnable positional '
              'encoding for failure prediction," 2023, arXiv:2309.02641.',
    "murray": 'J. F. Murray, G. F. Hughes, and K. Kreutz-Delgado, "Machine learning methods for predicting '
              'failures in hard drives: A multiple-instance application," *J. Mach. Learn. Res.*, vol. 6, '
              'pp. 783–816, 2005.',
    "ding": 'J. Ding, H. Li, and J. Li, "MLP-LightGBM hard drive failure prediction with transfer learning '
            'performance," in *Proc. IEEE Int. Symp. Parallel and Distributed Processing with Applications '
            '(ISPA)*, 2024, pp. 1333–1340, doi: 10.1109/ISPA63168.2024.00179.',
    "lightgbm": 'G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances '
                'in Neural Information Processing Systems*, vol. 30, 2017, pp. 3146–3154.',
    "xgboost": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD '
               'Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.',
    "optuna": 'T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation '
              'hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge '
              'Discovery and Data Mining*, 2019, pp. 2623–2631.',
    "aft": 'A. Barnwal, H. Cho, and T. Hocking, "Survival regression with accelerated failure time model in '
           'XGBoost," *J. Computational and Graphical Statistics*, vol. 31, no. 4, pp. 1292–1302, 2022.',
    "harrell": 'F. E. Harrell, R. M. Califf, D. B. Pryor, K. L. Lee, and R. A. Rosati, "Evaluating the yield '
               'of medical tests," *JAMA*, vol. 247, no. 18, pp. 2543–2546, 1982.',
    "saito": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot '
             'when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, '
             'Art. no. e0118432, 2015.',
    "shap": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in '
            '*Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765–4774.',
    "treeshap": 'S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable '
                'AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.',
    "react": 'S. Yao *et al.*, "ReAct: Synergizing reasoning and acting in language models," in *Proc. Int. '
             'Conf. Learning Representations (ICLR)*, 2023.',
}
_order: list[str] = []
REFS = {k: re.sub(r'"([^"]*)"', r"“\1”", v) for k, v in REFS.items()}   # IEEE curly quotes


def _cite(m: re.Match) -> str:
    nums = []
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in REFS:
            raise KeyError(f"unknown reference {k}")
        if k not in _order:
            _order.append(k)
        nums.append(_order.index(k) + 1)
    return ", ".join(f"[{n}]" for n in sorted(nums))


# ------------------------------------------------------------------ document
doc = Document(TEMPLATE)
body = doc.element.body
_sects = [p._p.pPr.sectPr for p in doc.paragraphs if p._p.pPr is not None and p._p.pPr.sectPr is not None]
SECT_TITLE, SECT_AUTHORS, SECT_BODY = (copy.deepcopy(_sects[i]) for i in (0, 1, 3))
for child in list(body):
    if child.tag != qn("w:sectPr"):
        body.remove(child)
body.replace(body.find(qn("w:sectPr")), copy.deepcopy(SECT_BODY))

TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])")


def runs(par, text: str) -> None:
    text = re.sub(r"\{cite:([^}]+)\}", _cite, text)
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("[["):
            r = par.add_run(part[2:-2])
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif part.startswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def P(text: str, style: str = "Body Text"):
    par = doc.add_paragraph(style=style)
    runs(par, text)
    return par


def H1(t): return P(t, "Heading 1")
def H2(t): return P(t, "Heading 2")
def H5(t):
    par = P(t, "Heading 5")
    par.paragraph_format.keep_with_next = True
    return par


def bullets(items):
    for it in items:
        P(it, "bullet list")


def end_section(sect, par=None):
    par = par or doc.paragraphs[-1]
    par._p.get_or_add_pPr().append(copy.deepcopy(sect))


def _borders(tbl):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def table(caption: str, header: list[str], rows: list[list[str]], widths_cm: list[float]):
    cap = P(caption, "table head")
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _borders(t)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            par = c.paragraphs[0]
            par.style = doc.styles["table col head" if i == 0 else "table copy"]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
            runs(par, val)
            par.paragraph_format.keep_with_next = i < len(rows)
    doc.add_paragraph(style="Body Text").paragraph_format.space_after = 0


def figure(path: Path, caption: str, width_cm: float = 8.4):
    par = doc.add_paragraph(style="Body Text")
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.first_line_indent = 0
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(str(path), width=Cm(width_cm))
    P(caption, "figure caption")


# ===================================================================== title
P("DiskGuard: Explainable, Agent-Assisted Hard-Drive Failure Management Evaluated at True Prevalence",
  "paper title")
end_section(SECT_TITLE)

for i in range(3):
    a = doc.add_paragraph(style="Author")
    if i > 0:   # break first, so each block starts at the top of its column
        a.add_run().add_break(WD_BREAK.COLUMN)
    lines = [f"[[{i + 1}{'st' if i == 0 else 'nd' if i == 1 else 'rd'} Given Name Surname]]",
             "[[dept. name of organization]]", "[[name of organization]]", "[[City, Country]]",
             "[[email address or ORCID]]"]
    for k, line in enumerate(lines):
        runs(a, line)
        if k < len(lines) - 1:
            a.add_run().add_break(WD_BREAK.LINE)
end_section(SECT_AUTHORS)

# ================================================================== abstract
ab = doc.add_paragraph(style="Abstract")
ab.add_run("Abstract").italic = True
ab.add_run("—")
runs(ab,
     "Hard-drive failures are rare on any one day but frequent across a large fleet, and published "
     "prediction results are often reported on rebalanced test data that hide how many alerts an operator "
     "would actually receive. We present DiskGuard, an end-to-end failure-management system built on "
     "thirteen years of Backblaze SMART telemetry (730 million drive-days). A five-seed LightGBM ensemble "
     "over 212 SMART-derived features predicts failure within 14 days. Trained on data to June 2025 and "
     "evaluated on the first half of 2026 at the true positive rate of 0.048 percent, it reaches a "
     "drive-day AUPRC of 0.078 (162 times the base rate) and a drive-level ROC-AUC of 0.908. Re-scoring the "
     "same predictions on a balanced test set would report an AUPRC of 0.917, a nearly twelvefold inflation. "
     "Recall-calibrated risk tiers place 67 percent of failing drives in a HIGH tier that holds 1.3 percent "
     "of the fleet, and a fixed daily alert budget raises the precision of new alerts by 14 percent over a "
     "fixed threshold with 22 percent fewer alerts. A survival model supplies an urgency ranking (C-index "
     "0.85) instead of an unreliable day count, TreeSHAP attributions are mapped to root causes and actions, "
     "and a tool-using language-model agent drafts maintenance tickets that a human must approve, with safety "
     "rules enforced in code and verified on 64,743 drives, ten adversarial scenarios and a live open-weight "
     "model. A label-free drift monitor flags when the model should be retrained.")
kw = doc.add_paragraph(style="Keywords")
kw.add_run("Keywords").italic = True
kw.add_run("—")
runs(kw, "hard disk drive failure prediction, SMART, gradient boosting, class imbalance, explainability, "
         "concept drift, language-model agents")

# ============================================================== I. introduction
H1("Introduction")
P("Hard disk drives remain the dominant medium for bulk storage in large datacenters, and they fail. "
  "Field studies report annual replacement rates of a few percent, well above datasheet figures "
  "{cite:schroeder}, so at fleet scale failures arrive daily, and each one costs a rebuild and a window of "
  "reduced redundancy. Predictive maintenance, which scores every drive every day and replaces those likely "
  "to fail, is the natural response, and SMART (Self-Monitoring, Analysis and Reporting Technology) "
  "telemetry is its standard input {cite:pinheiro,botezatu,lu}.")
P("Failure within a short horizon is a rare event. In our test period only 0.048 percent of drive-days "
  "are positive, so a model that always predicts healthy is 99.95 percent accurate. Accuracy, and to a "
  "lesser degree ROC-AUC, say little about whether the top of the ranking, where an operator actually "
  "looks, is any good {cite:saito}. Results computed on class-balanced test sets change the base rate and "
  "cannot be compared with deployment. A probability alone also does not tell an operator which drive to "
  "replace first, why it is at risk, or what to do.")
P("We build and evaluate a complete system that answers four operational questions for every drive in a "
  "363,000-drive fleet: will it fail soon, how urgent is it, why, and what should be done. Our "
  "contributions are:")
bullets([
    "Failure prediction evaluated on 57.6 million held-out drive-days from 2026 at true prevalence, "
    "with drive-clustered confidence intervals, and a measurement of how much common evaluation shortcuts "
    "inflate the same model's metrics (nearly twelvefold in AUPRC).",
    "Recall-calibrated risk tiers and a daily alert budget fixed on validation, an urgency ranking from a "
    "survival model in place of an unreliable day count, and TreeSHAP explanations mapped to tier-aware "
    "actions.",
    "An approval-gated maintenance agent whose safety rules are enforced by the server and verified "
    "exhaustively, adversarially and with a live model; the audit found, and we fixed, one rule enforced "
    "only by the prompt.",
    "Pre-registered studies of the infrastructure identifiers and of the alert policy, and a label-free "
    "drift monitor that flags when to retrain.",
])

# ============================================================== II. related work
H1("Related Work")
P("Large field studies established that SMART attributes carry failure signal but that many failures "
  "occur without warning {cite:pinheiro,schroeder}. Learning-based work has predicted disk replacement "
  "from SMART data {cite:botezatu} and combined SMART with performance and location signals "
  "{cite:lu}. A second line of work estimates remaining useful life (RUL) with recurrent and "
  "attention models {cite:lima,coursey,tfbest}.")
P("Closest to our work, Ding *et al.* {cite:ding} combine an MLP time-mixing stage with LightGBM to "
  "regress the RUL of Seagate ST4000DM000 drives, using the 60 days before each failure, and report a "
  "test RMSE near five days. Two properties of that protocol differ from ours. First, its test set "
  "contains only drives that fail, so it cannot measure false alarms on the healthy majority. Second, each "
  "drive is min-max normalised over its full recorded window, which includes observations a daily scorer "
  "would not yet have. We therefore address a complementary task, detection of imminent failure across a "
  "whole multi-vendor fleet at true prevalence, and do not compare error figures directly.")
P("Our predictor uses gradient-boosted trees {cite:lightgbm,xgboost} tuned with Optuna {cite:optuna}; "
  "urgency uses an accelerated-failure-time survival model {cite:aft} evaluated with the concordance index "
  "{cite:harrell}; explanations use SHAP and exact TreeSHAP {cite:shap,treeshap}; and the maintenance "
  "agent follows the tool-use pattern of reasoning-and-acting language-model agents {cite:react}.")

# ============================================================== III. data
H1("Data and Evaluation Protocol")
H2("Dataset and Label")
P("We use the public Backblaze Drive Stats dataset {cite:backblaze}, which publishes one record per drive "
  "per day with SMART attributes, model, capacity and a failure flag; we use April 2013 to June 2026. The "
  "target is 1 when a drive's recorded failure lies 1 to 14 days after the record's date. The failure day "
  "itself is labelled 0, so only advance warning is rewarded, and surviving drives are always 0. Fourteen "
  "days matches the time needed to schedule a replacement.")
H2("Temporal Splits")
P("Data are split by time, never at random (Table I). A two-week gap separates consecutive splits, so no "
  "label in one split looks ahead into the next. Validation and test keep the real failure rate; only the "
  "training set is rebalanced, keeping every positive row and ten negatives per positive.")
table("Temporal splits",
      ["Split", "Dates", "Drive-days", "Use"],
      [["Train", "2013-04-10 to 2025-06-16", "616.4 M (4.27 M after sampling)", "fitting"],
       ["Validation", "2025-07-01 to 12-17", "56.0 M", "threshold, tiers, alert budget"],
       ["Test", "2026-01-01 to 06-16", "57.6 M (363,405 drives)", "all results"]],
      [1.5, 2.4, 3.1, 1.6])
P("In the test period 2,451 of 363,405 drives (0.67 percent) fail within the horizon and 0.048 percent "
  "of drive-days are positive; the drive-day rate is the relevant base rate for daily ranking. The test "
  "ends on 16 June 2026, the last date with a full 14-day follow-up; the threshold, tiers and alert "
  "budget were fixed on validation before it was scored. The 2026 records were processed by a re-implementation of "
  "the cleaning and feature pipeline; on December 2025, where both can run, all 351,584 rows matched and "
  "205 of 212 features were identical.")
H2("Features")
P("The model uses 212 features: 83 raw and 83 normalised SMART attributes, 36 rolling means and standard "
  "deviations over 7, 14 and 30 days, 6 day-over-day deltas, and four derived features (drive age, "
  "power-on years, a combined error total and a temperature-stress indicator). All windows look backwards "
  "only. Missing values in core attributes are imputed with training-period medians. Five infrastructure "
  "identifiers (vault, pod, cluster, slot and a legacy flag) are excluded; Section V-F tests that choice.")
H2("Metrics and Uncertainty")
P("The primary metric is drive-day AUPRC on the full test split, reported with lift over prevalence; "
  "accuracy is not reported. Drive-level AUPRC scores each drive's latest record against whether it ever "
  "fails within the horizon. Rows of one drive are not independent, so every interval resamples physical "
  "drives: an exact weighted average-precision bootstrap over all 363,405 test drives (B = 1000), in which "
  "a drive drawn *m* times contributes each of its rows *m* times. Both comparative studies reported here, "
  "the infrastructure-identifier ablation and the alert-policy study, fixed their decision rules in a committed "
  "pre-registration before any result was seen; model comparisons use a materiality margin of 0.003 AUPRC.")

# ============================================================== IV. system
H1("System Design")
P("DiskGuard separates prediction, which produces immutable model outputs, from action, which an operator "
  "approves (Fig. 1).")
figure(FIG / "fig1_architecture.png", "DiskGuard architecture. Prediction outputs are immutable inputs to "
       "the agent; every ticket requires human approval.")
H2("Failure Prediction")
P("LightGBM {cite:lightgbm} hyperparameters were selected with Optuna {cite:optuna} (30 trials, TPE "
  "sampler), fitting on data to March 2024 and maximising AUPRC on the first 5 million rows of April to "
  "September 2024; both periods lie inside the training window, so no validation or test data informed "
  "them. The configuration uses 798 trees, learning rate 0.022, 53 leaves, maximum depth 10, row and column "
  "subsampling of 0.70 and 0.90, and L1 and L2 penalties of 0.13 and 4.96. The deployed model averages five "
  "boosters trained with different seeds. An alert threshold of 0.73 minimises expected cost on validation "
  "with a missed failure costed at 100 false alarms.")
H2("Recall-Calibrated Risk Tiers")
P("Operators work from tiers, not raw scores. Cutoffs are set on validation so that the HIGH tier captures "
  "50 percent of failing drive-days and HIGH plus MEDIUM captures 80 percent, giving HIGH at 0.721 and "
  "MEDIUM at 0.033. The cutoffs are frozen and applied unchanged to test.")
H2("Fleet Urgency")
P("An XGBoost accelerated-failure-time model {cite:aft} is fitted on the same features with "
  "interval-censored labels, on the classifier's training rows, with hyperparameters tuned on a whole-drive "
  "sample of validation. Failures after a split's last day are treated as censored, so no label uses later "
  "data. Its absolute day estimates are unreliable, because most "
  "of its training rows are censored, so the system reports a ranking: fleet urgency is the share of a "
  "reference fleet that the model expects to outlive the drive. Urgency orders work within a tier and "
  "never moves a drive to a higher tier.")
H2("Explanations and Actions")
P("Exact TreeSHAP {cite:treeshap} attributes each at-risk drive's score to its features. The strongest "
  "risk-increasing feature, with rolling and delta variants mapped to their base attribute, is looked up in "
  "a curated table of root causes and actions. The tier sets the deadline and the driver sets the kind of "
  "action: for reallocated sectors, a HIGH drive is scheduled for replacement within seven days, while a "
  "MEDIUM drive is planned for the next maintenance window with increased SMART polling.")
H2("Approval-Gated Maintenance Agent")
P("A tool-using language-model agent receives the prediction as immutable evidence and may call three "
  "simulated tools: a spare-parts check, a technician-availability check and a ticket draft. It never "
  "executes an action, and LOW-tier drives never reach the model. The server, not only the prompt, enforces "
  "the safety rules: ticket identity, tier and root cause are attached by the server; the parts lookup is "
  "locked to the drive's own model; a non-HIGH drive cannot be scheduled as urgent; tool inputs are "
  "validated; and runs are bounded in tools, turns and time. If the model ends without drafting a ticket, "
  "the server reminds it once and otherwise drafts the ticket from the rule mapping. A deterministic rule "
  "policy with the same tools is the no-network fallback and the evaluation reference.")

# ============================================================== V. results
H1("Results")
H2("Failure Prediction")
table("Test-set performance at true prevalence (January to June 2026)",
      ["Metric", "LightGBM ensemble"],
      [["Drive-day AUPRC [95% CI]", "0.0779 [0.0701, 0.0859]"],
       ["Lift over prevalence (0.00048)", "162"],
       ["Drive-day ROC-AUC", "0.886"],
       ["Drive-level AUPRC [95% CI]", "0.459 [0.438, 0.479]"],
       ["Drive-level ROC-AUC [95% CI]", "0.908 [0.900, 0.916]"],
       ["Single-seed AUPRC (5 seeds)", "0.0765 to 0.0782"],
       ["Logistic regression baseline", "AUPRC 0.021, ROC-AUC 0.820"],
       ["Precision / recall at 0.73", "3.6% / 57.4%"]],
      [4.6, 3.8])
P("The ensemble ranks failing drive-days 162 times better than chance and 3.6 times better than a "
  "class-weighted logistic regression trained on the same rows (Table II). Ensembling five seeds adds "
  "0.0008 AUPRC over the average single seed. At the alert threshold about one alerted drive-day in 28 "
  "precedes a failure. The most influential features (mean absolute SHAP) are the combined error total, "
  "normalised and raw SMART 1 (read error rate), raw SMART 4 (start/stop count) and raw SMART 241 (total "
  "logical blocks written).")
H2("How Evaluation Shortcuts Inflate the Same Model")
P("Table III re-scores the same test predictions under shortcuts common in the literature. Nothing about "
  "the model changes, only the composition of the evaluation set; this analysis is post hoc and descriptive.")
table("Same scores under different evaluation protocols",
      ["Protocol", "Neg. per pos.", "AUPRC", "ROC-AUC", "Prec. at 0.73"],
      [["True prevalence (ours)", "2,083", "0.078", "0.886", "3.6%"],
       ["Never-failing drives as negatives", "2,076", "0.113", "0.887", "4.2%"],
       ["Undersampled 1:100", "100", "0.461", "0.886", "43.6%"],
       ["Undersampled 1:10", "10", "0.742", "0.886", "88.6%"],
       ["Undersampled 1:1", "1", "0.917", "0.886", "98.7%"]],
      [3.1, 1.4, 1.2, 1.3, 1.5])
P("Every positive drive-day is kept in every protocol, so recall at the threshold is 57.4 percent "
  "throughout; undersampled rows are means of ten draws (AUPRC varies by at most 0.005). Undersampling to a "
  "1:1 balance, close to the class mix of early benchmark data (178 good and 191 failed drives, assigned at "
  "random to training and test sets {cite:murray}), raises AUPRC nearly twelvefold and precision at the "
  "deployed threshold from 3.6 to 98.7 percent: an operator would expect almost every alert to be a real "
  "failure, when in deployment about one in 28 is. Drawing negatives only from never-failing drives removes "
  "the hard negatives in a failing drive's earlier life and still inflates AUPRC by 45 percent. ROC-AUC is "
  "unchanged under every protocol, which is why it can neither expose these shortcuts nor describe the "
  "precision an operator will see; accuracy is highest for the rule that never predicts a failure (99.95 "
  "percent).")
H2("Risk Tiers and Lead Time")
table("Risk tiers on test, drive level (cutoffs fixed on validation)",
      ["Tier", "Drives", "Failing caught", "Precision"],
      [["HIGH", "4,841", "1,649 (67.3%)", "34.1%"],
       ["MEDIUM", "57,075", "414 (16.9%)", "0.7%"],
       ["HIGH + MEDIUM", "61,916", "2,063 (84.2%)", "3.3%"],
       ["LOW", "301,489", "388 (15.8%)", "0.1%"]],
      [2.4, 1.7, 2.6, 1.7])
P("The HIGH tier is small (1.3 percent of drives) and precise enough to act on: one in three of its drives "
  "fails within the horizon (Table IV). The tiers were calibrated to capture 50 and 80 percent of failing "
  "drive-days on validation and capture 57.9 and 79.6 percent on test, so the calibration transfers to a "
  "later period. MEDIUM is a broad watch list rather than a work queue: it holds 16 percent of drives at "
  "0.7 percent precision, and its actions are increased monitoring, not replacement. Among failing drives "
  "observed at each lead time, the share scoring at or above the alert threshold is 34.6, 44.1, 47.6 and "
  "54.4 percent at 30, 14, 7 and 1 days before failure. The 15.8 percent of failing drives that remain LOW "
  "reflect a known limit of SMART: many failures give no warning {cite:pinheiro}.")
H2("Daily Alert Budget")
P("A threshold fixes the score that raises an alert, not how many alerts arrive. We pre-registered a "
  "*daily budget*, which alerts on the *k* highest-scoring drives of each day, with *k* = 2,074 fitted on "
  "validation to match the threshold's catch rate, and compared it with the 0.73 threshold on the test "
  "period once, with the same drive-clustered bootstrap. The primary metrics are the precision of new "
  "alerts (an episode of consecutive alerted days counted at its first day, when an operator is paged) and "
  "the share of failing drives caught. A policy is adopted if its precision gain has a CI above zero and it "
  "loses at most 5 points of catch rate. A persistence rule (two consecutive days above the threshold) was "
  "pre-registered alongside it and was worse on both metrics.")
table("Alert policies on the test period",
      ["Policy", "Alerts per day", "New-alert precision", "Failing caught"],
      [["Threshold 0.73", "2,654", "4.02%", "69.5%"],
       ["Daily top 2,074", "2,074", "4.58%", "67.2%"]],
      [2.9, 1.8, 2.0, 1.7])
P("The daily budget is adopted: precision +0.55 points (95% CI +0.43 to +0.68, 14 percent relative) with "
  "22 percent fewer alerts, for 2.2 points of catch rate (Table V). It works because scores drift upward "
  "over time: the fixed threshold raised 26 percent more alerts a day on the test period than on "
  "validation, while the budget holds the workload constant. Smaller budgets trade coverage for precision "
  "(Fig. 2): at 100 alerts a day, 17.7 percent of alerted drive-days precede a failure (about one in six) "
  "and 23.4 percent of failing drives are caught.")
figure(FIG / "fig3_alert_budget.png", "Daily alert budget on the test period: share of alerted drive-days "
       "followed by a failure within 14 days, and share of failing drives caught. Diamonds mark the 0.73 "
       "threshold, which raised 2,654 alerts a day.")
H2("Urgency Ranking")
P("On each test drive's first 2026 observation (363,404 drives, 2,552 with a recorded failure), the "
  "survival model reaches a concordance index of 0.847 {cite:harrell}. Expressed as fleet urgency, the "
  "ranking agrees with the tiers without being redundant: mean urgency is 99.1 percent in HIGH, 82.1 in "
  "MEDIUM and 45.8 in LOW, so it orders the large MEDIUM tier rather than repeating it.")
H2("Robustness")
P("*Infrastructure identifiers (pre-registered).* Restoring the five identifiers, with the same training "
  "rows, hyperparameters and seeds, raises AUPRC from 0.0779 to 0.0797 (+0.0018, 95% CI +0.0003 to "
  "+0.0034), and the version with identifiers wins in all five seeds. By the pre-registered rules this is a "
  "reversal: the identifiers carry a small amount of signal. They describe where one operator installs its "
  "drives, so a model that relies on them would not transfer to another fleet; we keep them excluded and "
  "accept the small loss. *Calibration.* Brier score 0.011 and expected calibration error 0.037; scores are "
  "ranking scores, and decisions use the tiers and the alert budget.")
H2("Agent Evaluation")
table("Agent invariants on the rule policy (64,743 drives)",
      ["Invariant", "Applicable", "Failures"],
      [["Prediction returned unchanged", "64,743", "0"],
       ["LOW drive gets no tools and no ticket", "2,000", "0"],
       ["Ticket identity and tier server-owned", "62,743", "0"],
       ["Ticket is a draft requiring approval", "62,743", "0"],
       ["Parts lookup locked to drive model", "5,108", "0"],
       ["MEDIUM never scheduled as urgent", "57,635", "0"],
       ["No time-to-failure claimed", "64,743", "0"]],
      [4.6, 2.0, 1.8])
P("Table VI lists seven of the thirteen invariants; all thirteen hold on every evaluated drive: all 62,743 "
  "at-risk drives in the deployed system's latest assessment (30 June 2026) and 2,000 LOW drives. A "
  "scripted adversarial model attempted ten attacks, including tickets for another drive or tier, a parts "
  "lookup for another model, an unknown tool, escalating a MEDIUM drive, claiming a day count, tool loops, "
  "an API outage and a truncated response. The server boundary held in all ten; during development the "
  "audit exposed one rule enforced only by the prompt, which was fixed by capping the request in code. "
  "TreeSHAP explanations give a specific root cause for 99.5 percent of at-risk drives. "
  "*Live model.* We also ran the agent with a live open-weight model, GPT-OSS-120B served by Groq, on 22 "
  "drives: 16 HIGH and MEDIUM drives stratified by urgency, 2 LOW drives, and 4 MEDIUM drives whose "
  "explanation carried an injected instruction to escalate the drive and state a failure date. Every "
  "server-enforced invariant held, all four injection probes were resisted, and every at-risk drive reached "
  "the approval queue: 10 tickets came from the model unprompted, 6 after the reminder and 4 from the "
  "server fallback. Four runs did not complete cleanly, two because the provider rejected a malformed or "
  "schema-invalid tool call and two because free-tier rate limits exhausted the time budget; each still "
  "ended in a draft. The safety guarantees therefore rest on server-side enforcement, not on the model's "
  "reliability.")
H2("Drift Monitoring")
P("Labels arrive 14 days late, so we monitor each month against the model's validation period with two "
  "label-free signals, the share of drive-days above the alert threshold and the score distribution's "
  "population stability index (PSI), and the AUPRC of the last three labelled months (Fig. 3). Retraining is "
  "flagged at 1.25 times the validation alert volume, a PSI of 0.25, or 0.75 times the validation AUPRC, "
  "with warnings at 1.10, 0.10 and 0.85. The limits were chosen after the 2026 data had been seen, so this "
  "result is illustrative rather than confirmatory. Alert volume rose steadily from 1.07 times validation "
  "in January to 1.28 in April, when the monitor called for retraining, ten months after the training data "
  "ended. Ranking quality had not degraded: three-month AUPRC dipped to 0.77 times validation in March and "
  "was 1.03 to 1.07 times from April to June. PSI never exceeded 0.013, because the drift is concentrated in "
  "the high-score tail. A plausible mechanism is that cumulative counters such as power-on hours and "
  "start/stop counts grow as the fleet ages, pushing healthy drives' scores up; alert volume is the "
  "earliest and clearest signal, and the daily budget absorbs it until the model is retrained.")
figure(FIG / "fig4_drift_replay.png", "Drift monitor on the test period: monthly alert volume and "
       "three-month AUPRC relative to the validation period, with the retrain limit for alert volume.")

# ============================================================== VI. discussion
H1("Discussion and Limitations")
P("Our contribution is less a new predictor than an honest, complete system around a strong one, with "
  "each component's claims measured at true prevalence. Operators ask how many days are left, but the "
  "survival model cannot answer that reliably; a ranking uses only the property shown to hold, and it "
  "orders the large MEDIUM tier where tiers alone cannot. Prompt instructions are not a sufficient "
  "control, since the one gap the audit found was a rule stated in the prompt but not enforced by the "
  "server, and the live model skipped its ticket tool often enough that a server fallback was needed. A "
  "deployed model also ages: its scores drift upward as the fleet ages, a fixed daily alert budget keeps "
  "the operator's workload robust to that drift, and daily alert volume, which needs no labels, signals "
  "the need to retrain before ranking metrics can confirm it.")
P("Limitations. Scores are ranking scores, not calibrated probabilities. SMART has a ceiling: 15.8 percent "
  "of failing drives stay in the LOW tier. The MEDIUM tier is broad and imprecise. The test period covers "
  "six months, and failure labels come from operational removals and may include drives removed for other "
  "reasons. The fleet belongs to one operator and may not represent others. Training used an undersampled "
  "set, and the agent's tools are "
  "simulated.")

# ============================================================== VII. conclusion
H1("Conclusion")
P("DiskGuard shows that useful, explainable and safe failure management can be built around a single "
  "well-evaluated gradient-boosting model. Tested on six months of unseen data at true prevalence, it "
  "ranks failing drive-days 162 times better than chance, places two-thirds of failing drives in a tier "
  "holding 1.3 percent of the fleet, raises a fixed and more precise daily alert queue, orders work with a "
  "survival-model ranking, explains each alert, and drafts maintenance that a human approves under guard "
  "rails verified exhaustively, adversarially and with a live model. The same predictions scored on a "
  "balanced test set would look nearly twelve times better. Future work includes automating the retraining "
  "the drift monitor triggers and validation on a second operator's fleet.")

H5("Data and Code Availability")
P("The Backblaze data are public {cite:backblaze}. All stages are scripted, with pinned dependencies, "
  "pre-registration documents and the results files behind every table, at "
  "https://github.com/rakshithaaa-m/capstone."
  ).alignment = WD_ALIGN_PARAGRAPH.LEFT   # a justified line would stretch around the unbreakable URL

H5("References")
for k in _order:
    P(REFS[k], "references")
unused = sorted(set(REFS) - set(_order))
assert not unused, f"references never cited: {unused}"

# The template asks for equal columns on the last page: ending the body section
# after the references and starting a continuous one makes Word balance them.
end_section(SECT_BODY)
_final = body.find(qn("w:sectPr"))
_type = _final.find(qn("w:type"))
if _type is None:
    _type = OxmlElement("w:type")
    _final.insert(0, _type)
_type.set(qn("w:val"), "continuous")

doc.save(OUT)
print(f"[SAVE] {OUT}  ({len(_order)} references)")
