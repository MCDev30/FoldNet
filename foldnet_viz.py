"""Publication-quality plotting, statistics and table export for the FoldNet benchmarks.
No torch dependency: only numpy and matplotlib (scipy is optional, used for p-values)."""
import csv
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats as _st
except Exception:  # scipy is optional
    _st = None

# Okabe-Ito colour-blind-safe palette
COLORS = {
    "FoldNet": "#0072B2",
    "MLP (6 layers)": "#E69F00",
    "MLP (3 layers)": "#D55E00",
    "SIREN (3 layers)": "#009E73",
    "KAN (RBF)": "#CC79A7",
    "Target": "#222222",
}
_FALLBACK = ["#56B4E9", "#F0E442", "#999999", "#882255"]


def color_for(name):
    if name in COLORS:
        return COLORS[name]
    COLORS[name] = _FALLBACK[len(COLORS) % len(_FALLBACK)]
    return COLORS[name]


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


# def savefig(fig, name, results_dir="results"):
#     """Save a PNG (300 dpi) and a vector PDF."""
#     os.makedirs(results_dir, exist_ok=True)
#     for ext in ("png", "pdf"):
#         path = os.path.join(results_dir, f"{name}.{ext}")
#         fig.savefig(path)
#         print("saved", path)


def savefig(fig, name, results_dir="results"):
    """Save a PNG (300 dpi) and a vector file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.png")
    fig.savefig(path)
    print("saved", path)
    for ext in ("pdf", "svg"):
        path = os.path.join(results_dir, f"{name}.{ext}")
        try:
            fig.savefig(path)
            print("saved", path)
            return
        except Exception as e:
            print(f"could not save {ext.upper()} ({type(e).__name__}); trying next format")
    print("vector export unavailable; PNG saved")


def panel_label(ax, text):
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarize(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(median=np.nan, q25=np.nan, q75=np.nan, mean=np.nan, std=np.nan, n=0)
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return dict(median=med, q25=q25, q75=q75, mean=a.mean(),
                std=a.std(ddof=1) if len(a) > 1 else 0.0, n=len(a))


def band(ax, x, runs, color, label=None, marker=None, ls="-"):
    """Median line with interquartile band. runs has shape (n_runs, len(x))."""
    runs = np.asarray(runs, float)
    med = np.nanmedian(runs, axis=0)
    lo, hi = np.nanpercentile(runs, [25, 75], axis=0)
    ax.plot(x, med, color=color, ls=ls, marker=marker, ms=4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)

def prob_better(a, b):
    """P(a < b) over all pairs of runs (ties count 1/2). Lower is better."""
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float(((a < b) + 0.5 * (a == b)).mean())


def mannwhitney_p(a, b):
    if _st is None or len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(_st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def fmt_p(p):
    if p != p:
        return "n/a"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def sci(x):
    return f"{x:.2e}"


# # ---------------------------------------------------------------------------
# # Plot helpers
# # ---------------------------------------------------------------------------
# def band(ax, x, runs, color, label=None, marker=None, ls="-"):
#     """Median line with interquartile band. runs has shape (n_runs, len(x))."""
#     runs = np.asarray(runs, float)
#     med = np.median(runs, axis=0)
#     lo, hi = np.percentile(runs, [25, 75], axis=0)
#     ax.plot(x, med, color=color, ls=ls, marker=marker, ms=4, label=label)
#     ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def scatter_runs(ax, x, runs, color, size=9, alpha=0.45):
    """Individual runs as dots, with a small multiplicative jitter on x."""
    runs = np.asarray(runs, float)
    rng = np.random.default_rng(0)
    xs = np.asarray(x, float)[None, :] * (1 + rng.uniform(-0.04, 0.04, size=runs.shape))
    ax.scatter(xs.ravel(), runs.ravel(), s=size, color=color, alpha=alpha,
               linewidths=0, zorder=3)


# ---------------------------------------------------------------------------
# Tables (Markdown, LaTeX booktabs, CSV)
# ---------------------------------------------------------------------------
_SCI = re.compile(r"(-?\d+(?:\.\d+)?)e([+-]\d+)")


def _tex(cell):
    cell = _SCI.sub(lambda m: rf"${m.group(1)}\times10^{{{int(m.group(2))}}}$", cell)
    return (cell.replace("<", "$<$").replace("%", r"\%")
                .replace("_", r"\_").replace("&", r"\&"))


class Table:
    def __init__(self, header):
        self.header = [str(h) for h in header]
        self.rows = []

    def add(self, *cells, bold=()):
        self.rows.append(([str(c) for c in cells], set(bold or ())))

    def markdown(self):
        w = len(self.header)
        lines = ["| " + " | ".join(self.header) + " |", "|" + "---|" * w]
        for cells, b in self.rows:
            lines.append("| " + " | ".join(f"**{c}**" if i in b and c else c
                                           for i, c in enumerate(cells)) + " |")
        return "\n".join(lines)

    def latex(self, caption="", label=""):
        lines = [r"% requires \usepackage{booktabs}", r"\begin{table}[t]", r"\centering",
                 r"\small", rf"\begin{{tabular}}{{{'l' * len(self.header)}}}", r"\toprule",
                 " & ".join(_tex(h) for h in self.header) + r" \\", r"\midrule"]
        for k, (cells, b) in enumerate(self.rows):
            if k > 0 and cells[0] != "":
                lines.append(r"\midrule")
            lines.append(" & ".join(rf"\textbf{{{_tex(c)}}}" if i in b and c else _tex(c)
                                    for i, c in enumerate(cells)) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
                  rf"\label{{{label}}}", r"\end{table}"]
        return "\n".join(lines)

    def show(self):
        print(self.markdown())

    def save(self, name, results_dir="results", caption="", label=""):
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(self.markdown() + "\n")
        with open(os.path.join(results_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(self.latex(caption, label) + "\n")
        with open(os.path.join(results_dir, f"{name}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(cells for cells, _ in self.rows)
        print("saved", os.path.join(results_dir, name + ".{md,tex,csv}"))


def med_iqr(a, fmt=sci):
    s = summarize(a)
    return f"{fmt(s['median'])} [{fmt(s['q25'])}, {fmt(s['q75'])}]"


def vs_ref(ref, other):
    """P(ref better than other) and a Mann-Whitney p-value (lower metric = better)."""
    return f"{prob_better(ref, other):.2f}", fmt_p(mannwhitney_p(ref, other))


def log_xticks(ax, ticks):
    from matplotlib.ticker import NullLocator, ScalarFormatter
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())


def strip(ax, names, values, colors, horizontal=False):
    """Categorical dot plot: one dot per run, thick bar = IQR, tick = median."""
    rng = np.random.default_rng(0)
    for i, (n, v) in enumerate(zip(names, values)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        jit = rng.uniform(-0.12, 0.12, size=len(v))
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        c = colors[n]
        if horizontal:
            ax.scatter(v, i + jit, s=16, color=c, alpha=0.6, linewidths=0, zorder=3)
            ax.plot([q25, q75], [i, i], color=c, lw=4, alpha=0.35, solid_capstyle="butt")
            ax.plot([med, med], [i - 0.25, i + 0.25], color=c, lw=2.2)
        else:
            ax.scatter(i + jit, v, s=16, color=c, alpha=0.6, linewidths=0, zorder=3)
            ax.plot([i, i], [q25, q75], color=c, lw=4, alpha=0.35, solid_capstyle="butt")
            ax.plot([i - 0.25, i + 0.25], [med, med], color=c, lw=2.2)
    if horizontal:
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
    else:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")