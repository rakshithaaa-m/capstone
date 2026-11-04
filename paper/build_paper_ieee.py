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
     "would actually receive. We present DiskGuard, an end-to-end failure-management system built on more "
     "than twelve years of Backblaze SMART telemetry (681.7 million drive-days). A five-seed LightGBM "
     "ensemble over 212 SMART-derived features predicts failure within 14 days. Evaluated at the true "
     "positive rate of 0.047 percent on a 15-month held-out period, it reaches a drive-day AUPRC of 0.084 "
     "(179 times the base rate) and a drive-level AUPRC of 0.600. Re-scoring the same predictions on a "
     "balanced test set would report an AUPRC of 0.917, an elevenfold inflation. Recall-calibrated risk "
     "tiers place 66.4 percent of failing drives in a HIGH tier with 57 percent precision. A survival model "
     "supplies an urgency ranking instead of an unreliable day count, TreeSHAP attributions are mapped to "
     "root causes and actions, and a tool-using language-model agent drafts maintenance tickets that a "
     "human must approve, with safety rules enforced in code and verified on 34,794 drives and ten "
     "adversarial scenarios. On six months of 2026 data published after the model was frozen, recall is "
     "maintained but AUPRC falls to 0.061; the decline is gradual and driven by growing false alarms, and "
     "a pre-registered retraining on more recent data recovers AUPRC to 0.078.")
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
P("Failure within a short horizon is a rare event. In our test period only 0.047 percent of drive-days "
  "are positive, so a model that always predicts healthy is 99.95 percent accurate. Accuracy, and to a "
  "lesser degree ROC-AUC, say little about whether the top of the ranking, where an operator actually "
  "looks, is any good {cite:saito}. Results computed on class-balanced test sets change the base rate and "
  "cannot be compared with deployment. A probability alone also does not tell an operator which drive to "
  "replace first, why it is at risk, or what to do.")
P("We build and evaluate a complete system that answers four operational questions for every drive in a "
  "352,000-drive fleet: will it fail soon, how urgent is it, why, and what should be done. Our "
  "contributions are:")
bullets([
    "Failure prediction evaluated on 144.7 million held-out drive-days at true prevalence, with "
    "drive-clustered confidence intervals, and a measurement of how much common evaluation shortcuts "
    "inflate the same model's metrics (up to elevenfold in AUPRC).",
    "Recall-calibrated risk tiers fixed on validation, an urgency ranking from a survival model in place "
    "of an unreliable day count, and TreeSHAP explanations mapped to tier-aware actions.",
    "An approval-gated maintenance agent whose safety rules are enforced by the server and verified "
    "exhaustively and adversarially; the audit found, and we fixed, one rule enforced only by the prompt.",
    "Pre-registered robustness and model-comparison studies, and an out-of-time evaluation on 2026 data "
    "published after the model was frozen, which quantifies model ageing and shows that retraining "
    "recovers most of the loss.",
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
  "whole multi-vendor fleet at true prevalence, and do not compare error figures directly. "
  "[[Optional: reproduce their protocol and a leak-free variant for a direct comparison.]]")
P("Our predictor uses gradient-boosted trees {cite:lightgbm,xgboost} tuned with Optuna {cite:optuna}; "
  "urgency uses an accelerated-failure-time survival model {cite:aft} evaluated with the concordance index "
  "{cite:harrell}; explanations use SHAP and exact TreeSHAP {cite:shap,treeshap}; and the maintenance "
  "agent follows the tool-use pattern of reasoning-and-acting language-model agents {cite:react}.")

# ============================================================== III. data
H1("Data and Evaluation Protocol")
H2("Dataset and Label")
P("We use the public Backblaze Drive Stats dataset {cite:backblaze}, which publishes one record per drive "
  "per day with SMART attributes, model, capacity and a failure flag, from April 2013 to December 2025. The "
  "target is 1 when a drive's recorded failure lies 1 to 14 days after the record's date. The failure day "
  "itself is labelled 0, so only advance warning is rewarded, and surviving drives are always 0. Fourteen "
  "days matches the time needed to schedule a replacement.")
H2("Temporal Splits")
P("Data are split by time, never at random (Table I). Validation and test keep the real failure rate; only "
  "the training set is rebalanced, keeping every positive row and ten negatives per positive.")
table("Temporal splits",
      ["Split", "Dates", "Drive-days", "Use"],
      [["Train", "2013-04 to 2024-03", "484.1 M (3.44 M after sampling)", "fitting"],
       ["Validation", "2024-04 to 2024-09", "52.8 M", "tuning, threshold, tiers"],
       ["Test", "2024-10 to 2025-12", "144.7 M (352,383 drives)", "all results"]],
      [1.5, 2.4, 3.1, 1.6])
P("In the test period 5,350 of 352,383 drives (1.52 percent) fail within the horizon and 0.0472 percent "
  "of drive-days are positive; the drive-day rate is the relevant base rate for daily ranking. Labels for "
  "the last two weeks of the test split were completed with failures from January 2026, which were not "
  "visible when the split was first built (602 drive-days on 96 drives became positive). Results marked "
  "with an asterisk and the comparative studies in Sections V-E and V-F were computed before this "
  "correction; both arms of each comparison share the same labels.")
H2("Features")
P("The model uses 212 features: 83 raw and 83 normalised SMART attributes, 36 rolling means and standard "
  "deviations over 7, 14 and 30 days, 6 day-over-day deltas, and four derived features (drive age, "
  "power-on years, a combined error total and a temperature-stress indicator). All windows look backwards "
  "only. Missing values in core attributes are imputed with training-period medians. Five infrastructure "
  "identifiers (vault, pod, cluster, slot and a legacy flag) are excluded; Section V-E tests that choice.")
H2("Metrics and Uncertainty")
P("The primary metric is drive-day AUPRC on the full test split, reported with lift over prevalence; "
  "accuracy is not reported. Drive-level AUPRC scores each drive's latest record against whether it ever "
  "fails within the horizon. Rows of one drive are not independent, so every interval resamples physical "
  "drives: an exact weighted average-precision bootstrap over all 352,383 test drives (B = 1000), in which "
  "a drive drawn *m* times contributes each of its rows *m* times. Two studies, an infrastructure-feature "
  "ablation and a model-fusion study, fixed their decision rules before any result was seen, with a "
  "materiality margin of 0.003 AUPRC.")

# ============================================================== IV. system
H1("System Design")
P("DiskGuard separates prediction, which produces frozen model outputs, from action, which an operator "
  "approves (Fig. 1).")
figure(FIG / "fig1_architecture.png", "DiskGuard architecture. Prediction outputs are immutable inputs to "
       "the agent; every ticket requires human approval.")
H2("Failure Prediction")
P("LightGBM {cite:lightgbm} is tuned with Optuna {cite:optuna} (30 trials, TPE sampler) to maximise AUPRC "
  "on a fixed 5-million-row validation prefix. The selected configuration uses 798 trees, learning rate "
  "0.022, 53 leaves, maximum depth 10, row and column subsampling of 0.70 and 0.90, and L1 and L2 penalties "
  "of 0.13 and 4.96. The deployed model averages five boosters trained with different seeds. An alert "
  "threshold of 0.62 minimises expected cost on validation with a missed failure costed at 100 false alarms.")
H2("Recall-Calibrated Risk Tiers")
P("Operators work from tiers, not raw scores. Cutoffs are set on validation so that the HIGH tier captures "
  "50 percent of failing drive-days and HIGH plus MEDIUM captures 80 percent, giving HIGH at 0.764 and "
  "MEDIUM at 0.069. The cutoffs are frozen and applied unchanged to test.")
H2("Fleet Urgency")
P("An XGBoost accelerated-failure-time model {cite:aft} is fitted on the same features with "
  "interval-censored labels. Its absolute day estimates proved unreliable (Section V-D), so the system "
  "reports a ranking: fleet urgency is the share of a reference fleet that the model expects to outlive the "
  "drive. Urgency orders work within a tier and never moves a drive to a higher tier.")
H2("Explanations and Actions")
P("Exact TreeSHAP {cite:treeshap} attributes each at-risk drive's score to its features. The strongest "
  "risk-increasing feature, with rolling and delta variants mapped to their base attribute, is looked up in "
  "a curated table of root causes and actions. The tier sets the deadline and the driver sets the kind of "
  "action: for reallocated sectors, a HIGH drive is scheduled for replacement within seven days, while a "
  "MEDIUM drive is planned for the next maintenance window with increased SMART polling.")
H2("Approval-Gated Maintenance Agent")
P("A tool-using language-model agent receives the frozen prediction as immutable evidence and may call "
  "three simulated tools: a spare-parts check, a technician-availability check and a ticket draft. It never "
  "executes an action, and LOW-tier drives never reach the model. The server, not only the prompt, enforces "
  "the safety rules: ticket identity, tier and root cause are attached by the server; the parts lookup is "
  "locked to the drive's own model; a non-HIGH drive cannot be scheduled as urgent; tool inputs are "
  "validated; and runs are bounded in tools, turns and time. A deterministic rule policy with the same "
  "tools is the no-network fallback and the evaluation reference.")

# ============================================================== V. results
H1("Results")
H2("Failure Prediction")
table("Test-set performance at true prevalence",
      ["Metric", "LightGBM ensemble"],
      [["Drive-day AUPRC [95% CI]", "0.0844 [0.0792, 0.0903]"],
       ["Lift over prevalence (0.000472)", "179"],
       ["Drive-day ROC-AUC", "0.888"],
       ["Drive-level AUPRC [95% CI]", "0.600 [0.584, 0.614]"],
       ["Drive-level ROC-AUC [95% CI]", "0.919 [0.914, 0.924]"],
       ["Single-seed AUPRC (5 seeds)*", "0.0804 to 0.0828"],
       ["Logistic regression baseline*", "AUPRC 0.018, ROC-AUC 0.772"],
       ["Precision / recall at 0.62", "2.8% / 59.8%"]],
      [4.6, 3.8])
P("The ensemble ranks failing drive-days 179 times better than chance and 4.6 times better than logistic "
  "regression (Table II). Ensembling five seeds adds 0.002 AUPRC over the average single seed, comparable to "
  "the effects studied below, which is why every comparison holds the ensemble fixed. The most influential "
  "features (mean absolute SHAP) are the raw read-error rate, the combined error total, normalised SMART 18 "
  "and SMART 1, and the raw start/stop count.")
H2("How Evaluation Shortcuts Inflate the Same Model")
P("Table III re-scores the same test predictions under shortcuts common in the literature. Nothing about "
  "the model changes, only the composition of the evaluation set; this analysis is post hoc and descriptive.")
table("Same scores under different evaluation protocols",
      ["Protocol", "Neg. per pos.", "AUPRC", "ROC-AUC", "Prec. at 0.62"],
      [["True prevalence (ours)", "2,118", "0.084", "0.888", "2.8%"],
       ["Never-failing drives as negatives", "2,103", "0.132", "0.889", "3.4%"],
       ["Undersampled 1:100", "100", "0.458", "0.888", "37.5%"],
       ["Undersampled 1:10", "10", "0.737", "0.888", "85.7%"],
       ["Undersampled 1:1", "1", "0.917", "0.888", "98.4%"]],
      [3.1, 1.4, 1.2, 1.3, 1.5])
P("Every positive drive-day is kept in every protocol, so recall at the threshold is 59.8 percent "
  "throughout; undersampled rows are means of ten draws (AUPRC varies by at most 0.002). Undersampling to a "
  "1:1 balance [[cite studies evaluated on balanced test sets]] raises AUPRC elevenfold and precision at the "
  "deployed threshold from 2.8 to 98.4 percent: an operator would expect almost every alert to be a real "
  "failure, when in deployment about one in 36 is. Drawing negatives only from never-failing drives removes "
  "the hard negatives in a failing drive's earlier life and still inflates AUPRC by 56 percent. ROC-AUC is "
  "unchanged under every protocol, which is why it can neither expose these shortcuts nor describe the "
  "precision an operator will see; accuracy is highest for the rule that never predicts a failure (99.95 "
  "percent).")
H2("Risk Tiers and Lead Time")
table("Risk tiers on test, drive level (cutoffs fixed on validation)",
      ["Tier", "Drives", "Failing caught", "Precision"],
      [["HIGH", "6,282", "3,554 (66.4%)", "56.6%"],
       ["MEDIUM", "26,512", "853 (15.9%)", "3.2%"],
       ["HIGH + MEDIUM", "32,794", "4,407 (82.4%)", "13.4%"],
       ["LOW", "319,589", "943 (17.6%)", "0.3%"]],
      [2.4, 1.7, 2.6, 1.7])
P("The HIGH tier is small (1.8 percent of drives) and precise enough to act on: more than half of its "
  "drives fail within the horizon (Table IV). The tiers were calibrated to capture 50 and 80 percent of "
  "failing drive-days on validation and capture 52.3 and 75.5 percent on test, so the calibration transfers "
  "to a later period. Among failing drives observed at each lead time, the share scoring at or above the "
  "alert threshold (0.62) is 42.7, 50.5, 54.7 and 63.2 percent at 30, 14, 7 and 1 days before failure; "
  "half of failing drives are therefore flagged two weeks ahead. The 17.6 percent of failing drives that "
  "remain LOW reflect a known limit of SMART: many failures give no warning {cite:pinheiro}.")
H2("Urgency Ranking")
P("On each test drive's first observation (352,197 drives, 5,341 failing), the survival model reaches a "
  "concordance index of 0.749 {cite:harrell}. Its absolute estimates are not usable: on failing drives "
  "within 90 days of failure the mean absolute error is about 1,021 days, because roughly 85 percent of its "
  "training rows are censored. Expressed as fleet urgency, the ranking agrees with the tiers without being "
  "redundant: mean urgency is 99.4 percent in HIGH, 86.0 in MEDIUM and 53.9 in LOW.")
H2("Robustness")
P("*Infrastructure identifiers (pre-registered).* Restoring the five identifiers changes AUPRC by "
  "−0.00014 (95% CI −0.0025 to +0.0022), within the 0.003 margin, so the verdict is equivalence: the model "
  "does not depend on where a drive is installed. *Datacenter hold-out.* Holding out an entire site does not "
  "degrade ranking: the held-out sites phx1, sac0 and sac2 reach lifts of 216, 141 and 126, each above its "
  "size-matched control (90, 90 and 80), which separates distribution shift from having less training "
  "data. *Calibration.* Brier score 0.0107 and expected calibration error 0.040; scores are ranking scores, "
  "and decisions use the tiers.")
H2("Alternatives Evaluated")
P("Under pre-registered rules, LightGBM was compared with a recurrent (LSTM) remaining-life model and with "
  "fusions of LightGBM and a second model: the LSTM estimate, the survival model, and a separately tuned "
  "XGBoost classifier {cite:xgboost}. None improved on LightGBM by the 0.003 margin. The LSTM and "
  "survival-model fusions were worse (−0.016 and −0.039 AUPRC on the validation screen), and the XGBoost "
  "fusion, confirmed on the full test set, gained only +0.0004 (95% CI +0.0002 to +0.0006). Within bands of "
  "LightGBM's own score, the LSTM estimate ranks failures barely better than chance, and a second boosted "
  "model on the same features largely agrees with LightGBM.")
H2("Agent Evaluation")
table("Agent invariants on the rule policy (34,794 drives)",
      ["Invariant", "Applicable", "Failures"],
      [["Prediction returned unchanged", "34,794", "0"],
       ["LOW drive gets no tools and no ticket", "2,000", "0"],
       ["Ticket identity and tier server-owned", "32,794", "0"],
       ["Ticket is a draft requiring approval", "32,794", "0"],
       ["Parts lookup locked to drive model", "6,282", "0"],
       ["MEDIUM never scheduled as urgent", "26,512", "0"],
       ["No time-to-failure claimed", "34,794", "0"]],
      [4.6, 2.0, 1.8])
P("Table V lists seven of the thirteen invariants; all thirteen hold on every evaluated drive (all 32,794 "
  "at-risk drives and 2,000 LOW drives). A scripted adversarial model attempted ten attacks, including "
  "tickets for another drive or tier, a parts lookup for another model, an unknown tool, escalating a MEDIUM "
  "drive, claiming a day count, tool loops, an API outage and a truncated response. The server boundary held "
  "in all ten; the first run exposed one rule enforced only by the prompt, which was fixed by capping the "
  "request in code. TreeSHAP explanations give a specific root cause for 96.5 percent of at-risk drives. "
  "*Live model.* We also ran the agent with a live open-weight model, GPT-OSS-120B served by Groq, on 22 "
  "drives of the deployed 2026 system: 16 HIGH and MEDIUM drives stratified by urgency, 2 LOW drives, and "
  "4 MEDIUM drives whose explanation carried an injected instruction to escalate the drive and state a "
  "failure date. A first run showed that the model sometimes wrote its plan as text without calling the "
  "ticket tool (4 of 20 at-risk drives), so the server now reminds it once and otherwise drafts the ticket "
  "from the rule mapping. In the final run every server-enforced invariant held, all four injection probes "
  "were resisted, and every at-risk drive reached the approval queue: 10 tickets came from the model "
  "unprompted, 6 after the reminder and 4 from the server fallback. Four runs did not complete cleanly, two "
  "because the provider rejected a malformed or schema-invalid tool call and two because free-tier rate "
  "limits exhausted the time budget; each still ended in a draft. The safety guarantees therefore rest on "
  "server-side enforcement, not on the model's reliability.")

# ============================================================== VI. 2026
H1("Out-of-Time Evaluation on 2026 Data")
P("Backblaze published data for January to June 2026 after our model, threshold and tiers were frozen. We "
  "wrote the evaluation plan before scoring any 2026 row, applied the same cleaning and features (checked "
  "against the original pipeline on December 2025: all 351,584 rows matched, 205 of 212 features exactly), "
  "and evaluated rows up to 16 June, the last date with a full 14-day follow-up.")
table("Frozen model on the test period and on 2026",
      ["Metric", "Test (2024-25)", "2026 H1"],
      [["Drive-days / failing drives", "144.7 M / 5,350", "57.6 M / 2,451"],
       ["Drive-day AUPRC [95% CI]", "0.084 [0.079, 0.090]", "0.061 [0.055, 0.067]"],
       ["Lift over prevalence", "179", "127"],
       ["ROC-AUC", "0.888", "0.870"],
       ["Recall at 0.62", "59.8%", "63.8%"],
       ["Failing drives caught by HIGH", "66.4%", "66.4%"],
       ["Flagged 14 days ahead", "50.5%", "50.9%"]],
      [3.4, 2.6, 2.4])
P("By the rule stated in the plan, performance is degraded: the 2026 interval lies entirely below the "
  "test estimate (Table VI). Recall, tier capture and lead time held; ranking quality fell. Fig. 2 shows "
  "that the decline is gradual and began inside the test period: monthly AUPRC was 0.10 to 0.13 from "
  "October 2024 to April 2025 and fell to 0.07 to 0.09 by mid-2025. The cause is false alarms rather than "
  "missed failures. HIGH-tier drive-days per day rose from 1,393 in October 2024 to 2,405 in December 2025 "
  "and 2,960 in June 2026, while positive drive-days per day stayed near 140 to 180, and precision fell for "
  "every major drive model. Drive models absent from training are only 7.6 percent of the 2026 fleet and "
  "perform like known models, so new hardware is not the explanation. A plausible mechanism is that "
  "cumulative counters such as power-on hours and start/stop counts grow as the fleet ages, pushing healthy "
  "drives' scores up.")
figure(FIG / "fig2_monthly_auprc.png", "Monthly drive-day AUPRC of the frozen model on the test period "
       "and on 2026, and of the model retrained on data to June 2025 (Section VI-A).")
H2("Retraining on Recent Data")
P("We pre-registered a retraining study before training any model in it. The same features and "
  "hyperparameters were refitted on data to 16 June 2025; threshold and tiers were refitted on July to "
  "mid-December 2025; and 2026 was used once, for a confirmatory comparison against the frozen model with "
  "the same 0.003 margin and drive-clustered bootstrap. A second variant added, for ten key SMART "
  "attributes, each drive's percentile among drives of the same model on the same day, to test whether "
  "fleet-relative features resist ageing. On validation this variant gained +0.0022 AUPRC (95% CI +0.0005 "
  "to +0.0041), real but below the margin, so the plain retrained model went forward.")
table("Frozen and retrained models on 2026 (each at its own validation-fitted threshold)",
      ["Metric", "Frozen", "Retrained"],
      [["Drive-day AUPRC", "0.061", "0.078"],
       ["ROC-AUC", "0.870", "0.886"],
       ["Drive-level AUPRC", "0.422", "0.459"],
       ["Threshold", "0.62", "0.73"],
       ["Precision / recall at threshold", "2.2% / 63.8%", "3.6% / 57.4%"],
       ["Flagged 14 days ahead", "50.9%", "44.1%"]],
      [3.6, 2.0, 2.0])
P("Retraining helps by the pre-registered rule: AUPRC rises from 0.061 to 0.078, a gain of +0.017 (95% CI "
  "+0.013 to +0.021), which recovers most of the gap to the 0.084 of the original test period and improves "
  "every 2026 month (Fig. 2). The cost-optimal threshold rose from 0.62 to 0.73, so the retrained model "
  "trades some recall and lead time for 60 percent higher precision (Table VII). The model's decline was "
  "therefore mostly ageing that periodic retraining repairs, and the retrained model replaces the frozen one.")

# ============================================================== VII. discussion
H1("Discussion and Limitations")
P("A strong single model was the right choice: on rich SMART features at true prevalence, a tuned and "
  "ensembled LightGBM was not improved by any of the more complex designs we evaluated. Our contribution is "
  "less a new predictor than an honest, complete system around a strong one, with each component's claims "
  "measured. Operators ask how many days are left, but neither model we trained could answer reliably; a "
  "ranking uses only the property shown to hold. Prompt instructions are not a sufficient control, since "
  "the one gap the audit found was a rule stated in the prompt but not enforced by the server. Finally, a "
  "frozen model ages: on 2026 data a model retrained on more recent data recovers most of the loss, which "
  "argues for scheduled retraining and for monitoring daily alert volume, which rose well before ranking "
  "metrics could be computed.")
P("Limitations. Scores are ranking scores, not calibrated probabilities. SMART has a ceiling: 17.6 percent "
  "of failing drives stay in the LOW tier. Failure labels come from operational removals and may include "
  "drives removed for other reasons. The fleet belongs to one operator and may not represent others. "
  "Training used an undersampled set and tuning a validation prefix, and the agent's tools are simulated.")

# ============================================================== VIII. conclusion
H1("Conclusion")
P("DiskGuard shows that useful, explainable and safe failure management can be built around a single "
  "well-evaluated gradient-boosting model. At true prevalence it ranks failing drive-days 179 times better "
  "than chance, flags half of failing drives two weeks ahead, places two-thirds of failing drives in a tier "
  "precise enough to act on, orders work with a survival-model ranking, explains each alert, and drafts "
  "maintenance that a human approves under guard rails verified exhaustively and adversarially. The same "
  "predictions scored on a balanced test set would look eleven times better. On data from after the model "
  "was frozen its ranking degrades through rising false alarms, and retraining on recent data recovers "
  "most of the loss. Future work includes automated retraining with drift monitoring, a live-model agent "
  "study and validation on a second operator's fleet.")

H5("Data and Code Availability")
P("The Backblaze data are public {cite:backblaze}. All stages are scripted, with pinned dependencies, "
  "pre-registration documents and the results files behind every table. [[Repository link.]]")

H5("References")
for k in _order:
    P(REFS[k], "references")
unused = sorted(set(REFS) - set(_order))
assert not unused, f"references never cited: {unused}"

doc.save(OUT)
print(f"[SAVE] {OUT}  ({len(_order)} references)")
