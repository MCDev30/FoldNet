"""Multi-run benchmarking utilities: paired data, test metrics, fair baselines.
Complements foldnet_lib.py (which is left unchanged)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foldnet_lib import DEV, FoldNet, MLP, n_params, train


class Siren(nn.Module):
    """Sine-activation MLP (Sitzmann et al., 2020): a strong baseline for oscillatory targets."""

    def __init__(self, din, dout, w=32, depth=3, w0=30.0):
        super().__init__()
        self.w0 = w0
        dims = [din] + [w] * depth
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(w, dout)
        with torch.no_grad():
            for i, lin in enumerate(self.hidden):
                bound = 1.0 / dims[i] if i == 0 else math.sqrt(6.0 / dims[i]) / w0
                lin.weight.uniform_(-bound, bound)

    def forward(self, x):
        for lin in self.hidden:
            x = torch.sin(self.w0 * lin(x))
        return self.out(x)


class KANLayer(nn.Module):
    """KAN layer with a Gaussian-RBF basis (FastKAN style): one learnable univariate
    function per edge, plus a SiLU base path."""

    def __init__(self, din, dout, grid=8, lo=-2.0, hi=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(din) if din > 1 else nn.Identity()
        self.register_buffer("centers", torch.linspace(lo, hi, grid))
        self.h = (hi - lo) / (grid - 1)
        self.spline = nn.Linear(din * grid, dout, bias=False)
        self.base = nn.Linear(din, dout)

    def forward(self, x):
        x = self.norm(x)
        phi = torch.exp(-(((x.unsqueeze(-1) - self.centers) / self.h) ** 2))
        return self.spline(phi.flatten(1)) + self.base(F.silu(x))


class KAN(nn.Module):
    def __init__(self, din, dout, w=16, depth=2, grid=8):
        super().__init__()
        dims = [din] + [w] * (depth - 1) + [dout]
        self.layers = nn.ModuleList([KANLayer(dims[i], dims[i + 1], grid)
                                     for i in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def match_budget(make, target, lo=4, hi=512):
    """Return make(w) whose parameter count is closest to target."""
    w = min(range(lo, hi), key=lambda w: abs(n_params(make(w)) - target))
    return make(w)


def standard_builders(din=1, dout=1, d=32, depth=6, k=2):
    """FoldNet and four parameter-matched baselines for 1D regression."""
    target = n_params(FoldNet(din, dout, d, depth, k))
    builders = {
        "FoldNet":          lambda: FoldNet(din, dout, d, depth, k),
        "MLP (6 layers)":   lambda: match_budget(lambda w: MLP(din, dout, w, 6), target),
        "MLP (3 layers)":   lambda: match_budget(lambda w: MLP(din, dout, w, 3), target),
        "SIREN (3 layers)": lambda: match_budget(lambda w: Siren(din, dout, w, 3), target),
        "KAN (RBF)":        lambda: match_budget(lambda w: KAN(din, dout, w, 2), target),
    }
    return builders, target


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_data(f, n=2048, n_test=8192, lo=-1.0, hi=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * (hi - lo) + lo
    xt = torch.linspace(lo, hi, n_test).unsqueeze(1)
    return x, f(x), xt, f(xt)


def fit_and_eval(model, x, y, xt, yt, steps, desc="", keep_model=False):
    out = train(model, x, y, steps=steps, desc=desc)
    m = out["model"]
    m.eval()
    with torch.no_grad():
        pred = m(xt.to(DEV))
        err = pred - yt.to(DEV)
        mse = (err ** 2).mean().item()
        max_err = err.abs().max().item()
        rel_l2 = (err.norm() / yt.to(DEV).norm()).item()
    d = dict(mse=mse, max_err=max_err, rel_l2=rel_l2, curve=out["losses"],
             time=out["time"], pred=pred.cpu().numpy().ravel(), n_params=n_params(m))
    if keep_model:
        d["model"] = m
    return d


def benchmark(builders, f, n_runs=5, steps=8000, n_train=2048, label="", keep_models=False):
    """builders: {name: callable() -> nn.Module}.
    Within each run, all models see the same training set (paired comparison);
    the run index fixes both the data and the initialisation.
    Returns (results, test_grid, target_on_grid); results[name] = list of per-run dicts."""
    res = {name: [] for name in builders}
    xs = ys = None
    for r in range(n_runs):
        x, y, xt, yt = make_data(f, n=n_train, seed=1000 + r)
        xs, ys = xt.numpy().ravel(), yt.numpy().ravel()
        for name, build in builders.items():
            seed_all(r)
            res[name].append(fit_and_eval(build(), x, y, xt, yt, steps,
                                          desc=f"{label}{name} run {r + 1}/{n_runs}",
                                          keep_model=keep_models))
    return res, xs, ys