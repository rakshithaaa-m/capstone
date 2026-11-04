"""Fig. 3: alert budget vs precision and coverage on 2026 (results_alerts2026).
Fig. 4: drift-monitor replay, monthly alert volume vs validation (results_drift).
Run from the repo root. Same style and colours as make_fig2.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import json
import pandas as pd

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d3"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1b7a50"


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)


# ---------------------------------------------------------------- Fig. 3
g = pd.read_csv("results_alerts2026/topk_grid_test.csv").sort_values("k")
res = json.load(open("results_alerts2026/results.json"))
b = res["test_summary"]["B"]
k_star = res["k_star"]
fig, ax = plt.subplots(figsize=(3.45, 2.1), dpi=300)
ax.plot(g["k"], 100 * g["catch_rate"], color=ORANGE, lw=1.6, marker="s", ms=4, markeredgecolor="white",
        markeredgewidth=0.6, label="Failing drives caught", zorder=3)
ax.plot(g["k"], 100 * g["row_precision"], color=BLUE, lw=1.6, marker="o", ms=4.2, markeredgecolor="white",
        markeredgewidth=0.6, label="Alerts that precede a failure", zorder=3)
# the threshold rule, for reference (hollow markers)
ax.plot([b["alerted_rows_per_day"]] * 2, [100 * b["catch_rate"], 100 * b["row_precision"]], ls="none",
        marker="D", ms=4, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8,
        label="Threshold 0.73", zorder=4)
ax.axvline(k_star, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
ax.text(k_star * 0.93, 49, f"adopted\n{k_star:,}/day", ha="right", va="center", fontsize=6.6, color=MUTED)
ax.set_xscale("log")
ax.set_xticks([50, 100, 200, 500, 1000, 2000])
ax.set_xticklabels(["50", "100", "200", "500", "1,000", "2,000"])
ax.set_xlim(40, 3400); ax.set_ylim(0, 75)
ax.set_xlabel("Daily alert budget (drives)", color=INK)
ax.set_ylabel("Percent", color=INK)
style(ax)
ax.legend(frameon=False, fontsize=6.6, loc="center left", bbox_to_anchor=(0.0, 0.62), handlelength=2.0,
          labelspacing=0.3)
fig.tight_layout(pad=0.3); fig.savefig("paper/figures/fig3_alert_budget.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4
fz = pd.read_csv("results_drift/monthly_frozen.csv")
dp = pd.read_csv("results_drift/monthly_deployed.csv")
lim = json.load(open("results_drift/summary.json"))["frozen"]["limits"]
for df in (fz, dp):
    df["t"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
fig, ax = plt.subplots(figsize=(3.45, 2.1), dpi=300)
ax.plot(fz["t"], fz["alert_ratio"], color=BLUE, lw=1.6, marker="o", ms=3.8, markeredgecolor="white",
        markeredgewidth=0.6, label="Frozen model (trained to Mar 2024)", zorder=3)
ax.plot(dp["t"], dp["alert_ratio"], color=GREEN, lw=1.6, ls="--", marker="^", ms=4.2, markeredgecolor="white",
        markeredgewidth=0.6, label="Retrained model (trained to Jun 2025)", zorder=3)
right = fz["t"].max() + pd.Timedelta(days=12)
for level, ls in (("RETRAIN", (0, (3, 2))), ("WARN", (0, (1, 2)))):
    y = lim[level]["alert_ratio"]
    # the line stops short of its label so the two never overlap
    ax.hlines(y, fz["t"].min() - pd.Timedelta(days=15), right, color=MUTED, lw=0.8, ls=ls, zorder=2)
    ax.text(right + pd.Timedelta(days=4), y, f"{level.lower()} {y:.2f}×", va="center", ha="left",
            fontsize=6.6, color=MUTED)
first = fz.loc[fz["status"] == "RETRAIN", "t"].min()
ax.annotate("retrain flagged\nJul 2025", xy=(first, fz.loc[fz["t"] == first, "alert_ratio"].iloc[0]),
            xytext=(first - pd.Timedelta(days=150), 1.52), fontsize=6.6, color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
ax.set_ylabel("Alert volume (× validation)", color=INK)
ax.set_ylim(0.95, 1.85)
ax.set_xlim(fz["t"].min() - pd.Timedelta(days=15), right + pd.Timedelta(days=135))
# quarterly ticks over the data only; the right margin holds the limit labels
ax.set_xticks(pd.date_range("2024-10-01", "2026-07-01", freq="QS-JAN"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
style(ax)
ax.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=2.2, labelspacing=0.3)
fig.tight_layout(pad=0.3); fig.savefig("paper/figures/fig4_drift_replay.png", dpi=300); plt.close(fig)
print("saved fig3_alert_budget.png and fig4_drift_replay.png")
