"""Fig. 2: monthly AUPRC, test period and 2026 (frozen and retrained). Run from the repo root."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import pandas as pd
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6})
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d3"
m = pd.read_csv("results_fresh2026/monthly_test_vs_2026.csv")
m["t"] = pd.PeriodIndex(m["month"], freq="M").to_timestamp()
fig, ax = plt.subplots(figsize=(3.45, 2.15), dpi=300)
for period, color, marker, label in [("test", "#2a78d6", "o", "Test period (Oct 2024–Dec 2025)"),
                                     ("2026", "#eb6834", "s", "2026, frozen model"),
                                     ("2026_retrained", "#1b7a50", "^", "2026, retrained to Jun 2025")]:
    g = m[m["period"] == period]
    ax.plot(g["t"], g["auprc"], color=color, lw=1.6, marker=marker, ms=4.2, ls="--" if "retrained" in period else "-",
            markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
ax.axhline(0.0844, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2, label="Whole test split (0.084)")
ax.set_ylabel("Monthly AUPRC", color=INK); ax.set_ylim(0, 0.145)
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=7, length=2)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.legend(frameon=False, fontsize=6.6, loc="lower left", ncol=1, handlelength=2.2, labelspacing=0.3)
fig.tight_layout(pad=0.3); fig.savefig("paper/figures/fig2_monthly_auprc.png", dpi=300); plt.close(fig)
