# FoldNet: Neural Networks Built from Reflections and Folds

**FoldNet** is a fully connected neural network architecture built from Householder reflections and coordinatewise absolute-value operations. It combines orthogonal transformations with gradient-norm-preserving fold layers to obtain a depth-independent Lipschitz bound for the hidden network.

This repository accompanies the research work:

> **FoldNet: Neural Networks Built from Reflections and Folds**  
> Author: Charbel Mamlankou

---

## 1. Overview

A FoldNet hidden layer maps an input vector \(z \in \mathbb{R}^d\) according to

\[
F(z)=|Rz-t|,
\]

where:

- \(R\) is an orthogonal matrix represented as a product of \(k\) Householder reflections;
- \(t\in\mathbb{R}^d\) is a translation vector;
- the absolute value is applied coordinatewise.

A Householder reflection associated with a nonzero vector \(v\) is defined by

\[
H_v=I-2\frac{vv^\top}{\|v\|^2}.
\]

The orthogonal transformation is obtained as

\[
R=H_{v_k}\cdots H_{v_1}.
\]

A complete FoldNet consists of an input affine transformation, a composition of \(L\) fold layers, and an output affine transformation:

\[
f(x)=W_{\mathrm{out}}\,
\Phi(W_{\mathrm{in}}x+b_{\mathrm{in}})
+b_{\mathrm{out}},
\]

with

\[
\Phi=F_L\circ\cdots\circ F_1.
\]

The reflections can be applied successively without explicitly constructing the dense matrix \(R\). This gives a computational cost of \(O(kd)\) per fold layer.

## 2. Main theoretical properties

The architecture is studied through several structural results:

- **Orthogonal parametrization:** for \(k\geq d\), products of \(k\) Householder reflections represent exactly the orthogonal matrices with determinant \((-1)^k\).
- **Non-expansiveness:** every fold layer is 1-Lipschitz.
- **Gradient-norm preservation:** the Jacobian of a fold layer is orthogonal almost everywhere.
- **Depth-independent Lipschitz bound:** the complete network satisfies

  \[
  \operatorname{Lip}(f)
  \leq
  \|W_{\mathrm{out}}\|\,\|W_{\mathrm{in}}\|,
  \]

  independently of the number of hidden fold layers.
- **Piecewise-affine representation:** FoldNet admits an equivalent ReLU representation of width \(2d\).
- **Affine-piece bound:** for scalar input, a FoldNet of width \(d\) and depth \(L\) has at most \((d+1)^L\) affine pieces. The bound is attained when the width is one.

## 3. Numerical experiments

The accompanying study investigates the architecture through controlled numerical experiments.

| Experiment | Objective |
|---|---|
| Gradient flow | Study gradient-norm preservation at depths from 6 to 48 |
| Lipschitz estimation | Compare certified and empirical Lipschitz constants |
| Expressivity | Investigate the number of affine pieces |
| Ablation study | Examine the effects of reflections and the absolute-value operation |
| Frequency sweep | Evaluate regression across different sinusoidal frequencies |
| Discontinuous target | Compare FoldNet with MLP, SIREN and KAN models |
| MNIST classification | Evaluate classification performance under different parameter budgets |
| Robustness | Study certified robustness and performance under PGD attacks |

The experiments use multiple independent runs, with parameter-matched comparisons where specified in the manuscript.

### Selected findings

- FoldNet preserves hidden gradient norms across the tested depths, including depth 48.
- Its certified Lipschitz bound is within a factor of approximately 1.7 of the empirical constant in the reported experiment.
- Without adversarial training, FoldNet certifies approximately 56.2% of MNIST test images at an \(\ell_2\) radius of 0.25.
- At the smallest tested MNIST parameter budget, FoldNet achieves 93.46% test accuracy.
- On a discontinuous regression target, SIREN, KAN and a six-layer MLP outperform FoldNet.

These findings are specific to the experimental settings and baselines described in the accompanying paper.

## 4. Limitations and future directions

The current study has several limitations:

- No direct comparison with GroupSort networks or other Householder-activation architectures.
- Limited experimental coverage, primarily one-dimensional regression and fully connected MNIST models.
- The global Lipschitz certificate remains conservative compared with empirical robustness results.
- The affine-piece theorem concerns scalar inputs.
- No general approximation guarantee is established.
- The current implementation is approximately 2.5 times slower per training step than an MLP in the reported regression experiments.

Potential directions for further investigation include convolutional fold layers, certificate-aware training, approximation theory, and more extensive comparisons with orthogonal and gradient-norm-preserving architectures.

## 5. Repository contents

The repository is intended to provide the implementation and experimental material associated with the paper.

The exact directory structure, installation instructions, and execution commands will be documented here once the corresponding code release is finalized.

## 6. Reproducibility

The experiments described in the paper use:

- Adam optimization;
- cosine learning-rate decay;
- single-precision computation;
- fixed training schedules;
- multiple independent runs;
- parameter-matched baselines where applicable.

MNIST is publicly available. The regression datasets are synthetic and generated programmatically.

**Code availability:** The public repository URL and complete reproduction instructions will be added upon release.

## 7. Citation

If you use FoldNet in your research, please cite the associated paper:

```bibtex
@article{mamlankou_foldnet,
  title   = {FoldNet: Neural Networks Built from Reflections and Folds},
  author  = {Mamlankou, Charbel},
  year    = {2026},
  note    = {Manuscript}
}
```

Please update the citation details if a journal or conference version is published.

## 8. License

The license for this repository will be specified when the code is publicly released.

---

**Author:** Charbel Mamlankou  
**Project:** FoldNet - Neural Networks Built from Reflections and Folds