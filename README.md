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
```
## 📊 Results

### ✅ Supervised Baseline

The supervised model successfully learns the physical energy from the state:

- **RMSE** ≈ 0.013  
- **MAE** ≈ 0.010  
- **Pearson correlation** ≈ 0.99  

**Interpretation**

- Energy is highly learnable from the Burgers state  
- The dataset and numerical pipeline are consistent  
- This provides a strong reference point  

---

### ⚖️ Latent Autoencoder

The autoencoder reconstructs the state but does not uncover any meaningful scalar observable.

- Reconstruction RMSE ≈ 0.09  
- Correlation between `z` and energy ≈ 0  
- Monotonicity score ≈ 0.40  

**Interpretation**

- Good reconstruction does **not** imply physical meaning  
- The latent scalar remains unstructured  
- No evidence of emergent physical observable  

---

### 🔁 Latent Dynamics Model

The latent dynamics model learns a structured scalar variable `z(t)`:

- Monotonicity ≈ 0.99 ✅  
- Latent prediction error → very small ✅  
- Correlation with energy ≈ -0.13 ❌  

**Interpretation**

- The model successfully learns a nearly monotone latent variable  
- However, this variable is **not aligned with energy**  

---

### 🔍 Latent Variable Identification

We compared the latent scalar `z` against a library of physical observables.

#### Main findings

- The **most frequently aligned observable** is:

viscous dissipation: $$ \nu \int |u_x|^2 dx $$

- However, the relationship is **not exact**:
- moderate correlation (~0.4)
- partial fit via affine transformation

### ⚠️ Residual Analysis (Key Insight)

After fitting dissipation, we analyzed the residual:

- Residual vs energy → moderate correlation  
- Residual vs time → weak correlation  
- Residual vs max gradient → **strong correlation (~0.73+)**

#### Interpretation

The latent variable cannot be explained by dissipation alone.

Instead, it retains strong dependence on:

$$ max |u_x|  (shock / gradient intensity) $$
  
---

### 🧪 Final Interpretation

The learned scalar `z` is best understood as a **composite physical indicator** combining:

- global dissipation  
- local shock intensity  

A concise interpretation is:

$$z \sim  f( dissipation, max|u_x| )$$

---

### ✅ Key Takeaways

- ✅ Energy is easy to learn (if supervised)  
- ❌ Reconstruction does not reveal physics  
- ❌ Monotonicity alone is not sufficient  
- ✅ The model discovers a **structured but non-identifiable physical latent variable**  

The latent variable is best described as a **dissipation-dominated but shock-aware scalar**, rather than a single classical observable.
