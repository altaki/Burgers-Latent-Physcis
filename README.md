# Burgers-Latent-Physics

# 🚀 Burgers Latent Physics
### *Learning Physical Observables from PDE Dynamics*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Overview

This project investigates whether neural networks can **discover physically meaningful scalar observables** from the dynamics of a PDE.

We consider the **viscous 1D Burgers equation**:

$$ \partial_t u + u \partial_x u = \nu \partial_{xx} u $$


This system exhibits **dissipative behavior**, where the energy:

$$ E(t) = \dfrac{1}{2} \int u(x,t)^2 dx  $$

is strictly decreasing over time.

---

## 🎯 Objective

> Can a neural network learn a **latent scalar observable `z`** behaving like a physical quantity (e.g. energy), **without supervision**?

---

## 🧠 Methodology

We test three approaches:

---

### ✅ 1. Supervised baseline

Learn directly:

$$ u \rightarrow  E(u) $$

✔ Purpose: reference performance

---

### ⚖️ 2. Latent Autoencoder

Learn compression:

$$u \rightarrow h \to z \rightarrow \hat{u} $$

✔ No physical constraints  
✔ Tests if physics emerges from reconstruction

---

### 🔁 3. Latent Dynamics + Monotonicity

Learn latent dynamics:

$$ h_t → h_{t+1} $$

Scalar extracted: 

$$ z(t)$$


Constraint:


$$z(t+1) ≤ z(t)$$

✔ Hypothesis: physical quantities are monotone

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


---
📊 Results

✅ Supervised baseline
RMSE ≈ 0.013  
MAE  ≈ 0.010  
Pearson ≈ 0.992  
Spearman ≈ 0.976  

✔ Energy is accurately predicted

---


⚖️ Latent Autoencoder
Reconstruction RMSE ≈ 0.09  
Correlation(z, E) ≈ 0  
Monotonicity ≈ 0.40  

❌ Interpretation:

Good reconstruction
BUT no physical meaning in z

---


🔁 Latent Dynamics
Prediction error → very small ✅  
Monotonicity ≈ 0.99 ✅  
Correlation(z, E) ≈ -0.13 ❌  

✔ Learns a monotone variable
❌ But not the energy

---

📸 Key Figures

---

Latent vs Energy (Latent Dynamics)
outputs/figures/latent_dynamics_scatter_z_vs_energy.png
➡ No clear relationship between z and energy

Monotonicity (Latent Dynamics)
outputs/figures/latent_dynamics_monotonicity_histogram.png
➡ Strong monotonic behavior

Monotonicity (Autoencoder)
outputs/figures/latent_autoencoder_monotonicity_histogram.png
➡ No structure without constraint

---


🧠 Scientific Insights

✅ Energy is learnable
Supervised models recover energy accurately.

❌ Reconstruction ≠ Physics
Autoencoders do not extract physical observables.

❌ Monotonicity ≠ Energy
A monotone latent variable is not necessarily physical.


---
🔎 Conclusion

Neither reconstruction nor monotonicity alone is sufficient to recover a physically meaningful observable such as energy. I tried also to check if the Latent scalar variable is related to another physical variables, such as Entropy production, Second Fourrier coefficient, Shock width, Lebesgue norm of u ($L^1$ or $L^\infty$). However, it seems like the latent scalar variable is more strongly related to dissipation- and gradient-based observables than to energy. In particular, the best candidate in the current library is the viscous dissipation $\nu \int ∣u_x∣^2 dx$ , followed by  $\|u_x\|_{L^2}$ ​ and  $\|u_x\|_{L^\infty}$. This suggests that the learned scalar captures the intensity of sharp spatial structures rather than the energy itself.

