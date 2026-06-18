# Burgers-Latent-Physics

# 🚀 Burgers Latent Physics  
### *Learning Physical Observables from PDE Dynamics*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Objective](#-objective)
- [Methodology](#-methodology)
  - [1. Supervised baseline](#-1-supervised-baseline)
  - [2. Latent autoencoder](#️-2-latent-autoencoder)
  - [3. Latent dynamics + monotonicity](#-3-latent-dynamics--monotonicity)
- [Pipeline](#️-pipeline)
- [Results](#-results)
  - [Supervised baseline](#-supervised-baseline-1)
  - [Latent autoencoder](#️-latent-autoencoder-1)
  - [Latent dynamics](#-latent-dynamics)
- [Candidate identification of the latent variable](#-candidate-identification-of-the-latent-variable)
- [Interpretation of the learned latent scalar](#-interpretation-of-the-learned-latent-scalar)
- [Key Figures](#-key-figures)
- [Scientific Insights](#-scientific-insights)
- [Conclusion (End of Step 1)](#-conclusion-end-of-step-1)
- [Next Step (Step 2)](#-next-step-step-2)
- [Project Status](#-project-status)
- [Planned Step-2 Experiments](#-planned-step-2-experiments)
- [Author](#-author)
- [Key Insight](#-key-insight)

---

## 📖 Overview

This project investigates whether neural networks can **discover physically meaningful scalar observables** from the dynamics of a nonlinear PDE.

We consider the **viscous 1D Burgers equation**:

$$
\partial_t u + u \partial_x u = \nu \partial_{xx} u
$$

This system exhibits **dissipative behavior**, where the energy

$$
E(t) = \dfrac{1}{2} \int u(x,t)^2 \, dx
$$

decreases over time.

The central scientific question is whether a neural network can learn a **latent scalar variable** that behaves like a physical quantity **without explicit supervision**.

---

## 🎯 Objective

> Can a neural network learn a **latent scalar observable** `z` that behaves like a physically meaningful quantity (e.g. energy or dissipation), **without being explicitly told what that quantity is**?

---

## 🧠 Methodology

We explored three learning settings of increasing structure.

---

### ✅ 1. Supervised baseline

We first train a standard neural network to predict the energy directly from the PDE state:

$$
u \rightarrow E(u)
$$

**Purpose**
- establish a strong baseline,
- confirm that the dataset and learning setup are sound,
- measure how easily a known physical quantity can be learned.

---

### ⚖️ 2. Latent Autoencoder

We then train an autoencoder to compress and reconstruct the state:

$$
u \rightarrow h \rightarrow z \rightarrow \hat{u}
$$

**Purpose**
- test whether a meaningful scalar observable can emerge from reconstruction alone,
- evaluate whether compression is sufficient to recover physics.

**Key idea**
- no explicit physical constraint is imposed on `z`.

---

### 🔁 3. Latent Dynamics + Monotonicity

Finally, we train a latent dynamics model:

$$
h_t \rightarrow h_{t+1}
$$

and extract a scalar variable

$$
z(t)
$$

while imposing a monotonicity constraint:

$$
z(t+1) \leq z(t)
$$

**Purpose**
- test whether a latent scalar can emerge as a **monotone** quantity,
- investigate whether monotonicity is sufficient to recover a physical observable.

**Hypothesis**
- many physical quantities in dissipative systems are monotone in time.

---

## ⚙️ Pipeline

```bash
python main_generate_data.py
python main_train_supervised.py
python main_train_latent.py
python main_evaluate.py
python main_visualize_results.py
python main_visualize_latent_results.py
``
