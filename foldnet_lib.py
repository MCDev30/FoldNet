"""
Librairie partagée pour les 8 benchmarks FoldNet.
Conserve le motif de capture CUDA graph du code original.
"""
import time, math
import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH = DEV == "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Modèles (régression)
# ---------------------------------------------------------------------------
class FoldNet(nn.Module):
    def __init__(self, din, dout, d=32, depth=6, k=2):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.v = nn.Parameter(torch.randn(depth, k, d))
        self.t = nn.Parameter(0.5 * torch.randn(depth, d))
        self.out = nn.Linear(d, dout)
        self.register_buffer("I", torch.eye(d))

    def _compute_Mt(self):
        u = self.v / (self.v.pow(2).sum(-1, keepdim=True) + 1e-8).sqrt()
        M = self.I.expand(self.v.shape[0], -1, -1)
        for i in range(self.v.shape[1]):
            ui = u[:, i]
            M = M - 2 * ui.unsqueeze(-1) @ (ui.unsqueeze(1) @ M)
        return M.transpose(1, 2)

    def forward(self, x):
        Mt = self._compute_Mt()
        x = self.inp(x)
        for l in range(self.v.shape[0]):
            x = (x @ Mt[l] - self.t[l]).abs()
        return self.out(x)

    def forward_with_activations(self, x):
        Mt = self._compute_Mt()
        x = self.inp(x)
        acts = [x]
        for l in range(self.v.shape[0]):
            x = (x @ Mt[l] - self.t[l]).abs()
            acts.append(x)
        return self.out(x), acts


class MLP(nn.Module):
    def __init__(self, din, dout, w=32, depth=6):
        super().__init__()
        L = [nn.Linear(din, w), nn.ReLU()]
        for _ in range(depth - 1):
            L += [nn.Linear(w, w), nn.ReLU()]
        L += [nn.Linear(w, dout)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)

    def forward_with_activations(self, x):
        acts = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h)
        return h, acts


# ---------------------------------------------------------------------------
# Modèles (classification)
# ---------------------------------------------------------------------------
class FoldNetClassifier(nn.Module):
    def __init__(self, din=784, nclass=10, d=64, depth=4, k=2):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.v = nn.Parameter(torch.randn(depth, k, d))
        self.t = nn.Parameter(0.5 * torch.randn(depth, d))
        self.out = nn.Linear(d, nclass)
        self.register_buffer("I", torch.eye(d))

    def forward(self, x):
        u = self.v / (self.v.pow(2).sum(-1, keepdim=True) + 1e-8).sqrt()
        M = self.I.expand(self.v.shape[0], -1, -1)
        for i in range(self.v.shape[1]):
            ui = u[:, i]
            M = M - 2 * ui.unsqueeze(-1) @ (ui.unsqueeze(1) @ M)
        Mt = M.transpose(1, 2)
        x = self.inp(x)
        for l in range(self.v.shape[0]):
            x = (x @ Mt[l] - self.t[l]).abs()
        return self.out(x)


class MLPClassifier(nn.Module):
    def __init__(self, din=784, nclass=10, w=64, depth=4):
        super().__init__()
        L = [nn.Linear(din, w), nn.ReLU()]
        for _ in range(depth - 1):
            L += [nn.Linear(w, w), nn.ReLU()]
        L += [nn.Linear(w, nclass)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def match_mlp(target, din, dout, depth):
    w = min(range(4, 512), key=lambda w: abs(n_params(MLP(din, dout, w, depth)) - target))
    return MLP(din, dout, w, depth)


def match_mlp_classifier(target, din=784, nclass=10, depth=4):
    w = min(range(4, 256),
            key=lambda w: abs(n_params(MLPClassifier(din, nclass, w, depth)) - target))
    return MLPClassifier(din, nclass, w, depth)


# ---------------------------------------------------------------------------
# Entraînement CUDA-graph (données fixes, comme ton code)
# ---------------------------------------------------------------------------
def train(model, x, y, steps=10000, lr=3e-3, desc="", loss_fn=None):
    model = model.to(DEV)
    x, y = x.to(DEV), y.to(DEV)
    if loss_fn is None:
        loss_fn = lambda p, t: ((p - t) ** 2).mean()

    lr_t = torch.tensor(lr, device=DEV) if GRAPH else lr
    opt = torch.optim.Adam(model.parameters(), lr=lr_t, capturable=GRAPH)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        return loss

    if GRAPH:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            static_loss = step()

    losses = torch.empty(steps, device=DEV)
    t0 = time.time()
    pbar = tqdm(range(steps), desc=desc)
    for i in pbar:
        lr_i = 0.5 * lr * (1 + math.cos(math.pi * i / steps))
        if GRAPH:
            opt.param_groups[0]["lr"].fill_(lr_i)
            g.replay()
            losses[i] = static_loss.detach()
        else:
            opt.param_groups[0]["lr"] = lr_i
            losses[i] = step().detach()
        if i % 100 == 0:
            pbar.set_postfix(loss=f"{losses[i].item():.4f}")
    if DEV == "cuda":
        torch.cuda.synchronize()
    return dict(losses=losses.cpu().numpy(), time=time.time() - t0, model=model)


# ---------------------------------------------------------------------------
# Entraînement CUDA-graph avec mini-batches (buffers statiques, MNIST)
# ---------------------------------------------------------------------------
def train_with_loader(model, x_all, y_all, steps=3000, batch=128, lr=1e-3, loss_fn=None, desc=""):
    model = model.to(DEV)
    x_all, y_all = x_all.to(DEV), y_all.to(DEV)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    lr_t = torch.tensor(lr, device=DEV) if GRAPH else lr
    opt = torch.optim.Adam(model.parameters(), lr=lr_t, capturable=GRAPH)

    static_x = torch.empty(batch, *x_all.shape[1:], device=DEV)
    static_y = torch.empty(batch, dtype=y_all.dtype, device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(static_x), static_y)
        loss.backward()
        opt.step()
        return loss

    if GRAPH:
        static_x.copy_(x_all[:batch])
        static_y.copy_(y_all[:batch])
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            static_loss = step()

    n = x_all.shape[0]
    losses = torch.empty(steps, device=DEV)
    t0 = time.time()
    pbar = tqdm(range(steps), desc=desc)
    for i in pbar:
        idx = torch.randint(0, n, (batch,), device=DEV)
        static_x.copy_(x_all[idx])
        static_y.copy_(y_all[idx])
        lr_i = 0.5 * lr * (1 + math.cos(math.pi * i / steps))
        if GRAPH:
            opt.param_groups[0]["lr"].fill_(lr_i)
            g.replay()
            losses[i] = static_loss.detach()
        else:
            opt.param_groups[0]["lr"] = lr_i
            losses[i] = step().detach()
        if i % 100 == 0:
            pbar.set_postfix(loss=f"{losses[i].item():.5f}")
    if DEV == "cuda":
        torch.cuda.synchronize()
    return dict(losses=losses.cpu().numpy(), time=time.time() - t0, model=model)


def smooth(a, k=25):
    return np.convolve(a, np.ones(k) / k, mode="valid")