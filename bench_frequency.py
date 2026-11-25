"""Test 4 (revised): sin(kx) regression. Multi-run test MSE vs frequency, baselines, statistics."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

from foldnet_bench import benchmark, standard_builders
import foldnet_viz as V

FREQS = [5, 10, 20, 50, 100]
N_RUNS = 5            # independent runs per (model, frequency); use 10 for a paper
STEPS = 8000
SHOW_FREQ = 50        # frequency shown in the fit panel
D, DEPTH, K = 32, 6, 2
RESULTS_DIR = "results"

BUILDERS, TARGET = standard_builders(1, 1, D, DEPTH, K)

if __name__ == "__main__":
    assert SHOW_FREQ in FREQS
    V.set_style()
    names = list(BUILDERS)
    mse = {n: [] for n in names}          # mse[name][i_freq] -> array of shape (N_RUNS,)
    n_par, fit, base = {}, {}, []
    grid = target_curve = None

    for freq in FREQS:
        res, xs, ys = benchmark(BUILDERS, lambda x, k=freq: torch.sin(k * x),
                                n_runs=N_RUNS, steps=STEPS, label=f"k={freq} ")
        base.append(float(np.var(ys)))    # MSE of the best constant predictor
        for n in names:
            m = np.array([r["mse"] for r in res[n]])
            mse[n].append(m)
            n_par[n] = res[n][0]["n_params"]
            if freq == SHOW_FREQ:         # keep the median run for the fit panel
                fit[n] = res[n][int(np.argsort(m)[len(m) // 2])]["pred"]
                grid, target_curve = xs, ys

    # ---- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    for n in names:
        M = np.stack(mse[n], axis=1)      # (N_RUNS, n_freq)
        V.band(ax, FREQS, M, V.color_for(n), label=f"{n} ({n_par[n]} params)", marker="o")
        V.scatter_runs(ax, FREQS, M, V.color_for(n))
    ax.plot(FREQS, base, "k--", lw=1, label="constant predictor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(FREQS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"Target frequency $k$")
    ax.set_ylabel("Test MSE")
    ax.set_title(r"Regression of $\sin(kx)$ on $[-1, 1]$")
    ax.legend(loc="upper left")
    V.panel_label(ax, "(a)")

    ax = axes[1]
    mask = (grid >= 0.0) & (grid <= 0.4)
    ax.plot(grid[mask], target_curve[mask], color=V.COLORS["Target"], lw=3.0, alpha=0.25,
            label="Target")
    for n in names:
        ax.plot(grid[mask], fit[n][mask], color=V.color_for(n), lw=1.2, label=n)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_title(rf"Median run at $k = {SHOW_FREQ}$")
    ax.legend(loc="upper right", ncol=2, fontsize=7.5)
    V.panel_label(ax, "(b)")

    fig.text(0.5, -0.03, f"{N_RUNS} independent runs per point. Line: median. Band: interquartile "
             "range. Dots: individual runs. All models have a comparable parameter budget.",
             ha="center", fontsize=8, color="0.35")
    plt.tight_layout()
    V.savefig(fig, "test4_sin_kx", RESULTS_DIR)
    plt.show()

    # ---- Table ------------------------------------------------------------
    tab = V.Table(["$k$", "Model", "Params", "Median test MSE", "IQR [Q1, Q3]",
                   "P(FoldNet better)", "p-value"])
    for j, fr in enumerate(FREQS):
        meds = {n: float(np.median(mse[n][j])) for n in names}
        best = min(meds, key=meds.get)
        for n in names:
            s = V.summarize(mse[n][j])
            if n == "FoldNet":
                pb, pv = "-", "-"
            else:
                pb = f"{V.prob_better(mse['FoldNet'][j], mse[n][j]):.2f}"
                pv = V.fmt_p(V.mannwhitney_p(mse["FoldNet"][j], mse[n][j]))
            tab.add(fr if n == names[0] else "", n, n_par[n], V.sci(s["median"]),
                    f"[{V.sci(s['q25'])}, {V.sci(s['q75'])}]", pb, pv,
                    bold={3} if n == best else ())
    tab.show()
    tab.save("test4_sin_kx_table", RESULTS_DIR,
             caption=(rf"Test MSE for $\sin(kx)$ regression ({N_RUNS} runs). "
                      "P(FoldNet better) is the probability that a random FoldNet run beats a "
                      "random run of the baseline; p-values are two-sided Mann-Whitney U tests. "
                      "Bold: best median."),
             label="tab:sinkx")