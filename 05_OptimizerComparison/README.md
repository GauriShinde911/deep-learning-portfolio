# Practical — Deep Learning Optimizer Comparison

> **Skills demonstrated:** Optimizer design · Adaptive learning rates · Convergence analysis · Regularization · Performance benchmarking  
> **Relevant roles:** ML Engineer · AI Researcher · Deep Learning Practitioner · ML Consultant

---

## 🎯 Objective

Systematically compare five widely-used gradient-based optimization algorithms on real deep learning tasks and study how **optimizer choice** affects:
- Classification accuracy and final performance
- Convergence speed and training stability
- Sensitivity to learning rate
- Behaviour under different architectures (MLP vs CNN)

---

## 📂 Files in this Folder

| File | Description |
|------|-------------|
| `optimizer_comparison.ipynb` | Main notebook — full implementation + benchmarks |
| `README.md` | This file |
| `requirements.txt` | Python dependencies for local setup |

---

## 🗂️ Datasets Used

### MNIST
**Source:** Built into `tensorflow.keras.datasets`  
**Size:** 70,000 grayscale images (60k train + 10k test)  
**Image size:** 28 × 28 = **784 features** per image (after flattening)  
**Classes:** 10 (digits 0–9)

### CIFAR-10
**Source:** Built into `tensorflow.keras.datasets`  
**Size:** 60,000 color images (50k train + 10k test)  
**Image size:** 32 × 32 × 3 = **3,072 features** per image  
**Classes:** 10

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | Airplane | 5 | Dog |
| 1 | Automobile | 6 | Frog |
| 2 | Bird | 7 | Horse |
| 3 | Cat | 8 | Ship |
| 4 | Deer | 9 | Truck |

---

## 🧠 Optimizers Compared (5 Algorithms)

| Optimizer | Type | Adaptive LR | Momentum | Key Hyperparameter |
|-----------|------|:-----------:|:--------:|--------------------|
| SGD | Vanilla + Momentum | ❌ | ✅ | `momentum=0.9` |
| Adagrad | Adaptive per-param | ✅ | ❌ | `lr=0.01` |
| RMSprop | Adaptive EMA | ✅ | ❌ | `rho=0.9` |
| Adam | Adaptive + Momentum | ✅ | ✅ | `β₁=0.9, β₂=0.999` |
| Nadam | Adam + Nesterov | ✅ | ✅ | `β₁=0.9, β₂=0.999` |

---

## ⚙️ Preprocessing Pipeline

```
Raw images  (uint8, 0–255)
        ↓
Flatten  →  (N, 784) for MNIST  |  (N, 3072) for CIFAR-10
        ↓
Normalize  →  / 255.0  →  float32  0.0–1.0
        ↓
[CIFAR-10 only] Per-channel normalization  →  (x - mean) / std
        ↓
One-Hot Encode labels  →  (N, 10)
        ↓
Feed into MLP / CNN
```

---

## 🏗️ Model Architectures

### MLP — MNIST Benchmark
```
Input(784) → Dense(256, ReLU) → BN → Dropout(0.3)
           → Dense(128, ReLU) → BN → Dropout(0.2)
           → Dense(64, ReLU)
           → Dense(10, Softmax)
```

### CNN — CIFAR-10 Benchmark
```
Input(32,32,3) → Conv2D(32) → Conv2D(32) → MaxPool → BN → Dropout(0.25)
               → Conv2D(64) → Conv2D(64) → MaxPool → BN → Dropout(0.25)
               → GlobalAvgPool
               → Dense(128, ReLU) → Dropout(0.4)
               → Dense(10, Softmax)
```

---

## 📊 Results

### MNIST — MLP Benchmark

| Optimizer | Test Accuracy | Best Val Acc | Convergence (90%) | Overfitting? |
|-----------|:-----------:|:----------:|:----------------:|-------------|
| SGD | ~96.5% | ~96.8% | ~18 epochs | Low ✅ |
| Adagrad | ~96.0% | ~96.2% | ~20 epochs | Low ✅ |
| RMSprop | ~97.0% | ~97.2% | ~12 epochs | Moderate |
| Adam | ~97.3% | ~97.5% | ~10 epochs | Low ✅ |
| Nadam | ~97.5% | ~97.7% | ~8 epochs | Low ✅ |

### CIFAR-10 — CNN Benchmark

| Optimizer | Test Accuracy | Test Loss | Convergence | Overfitting? |
|-----------|:-----------:|:---------:|:-----------:|-------------|
| SGD | ~62% | ~1.10 | Slow ⚠️ | Moderate |
| Adagrad | ~58% | ~1.20 | Stalls early ⚠️ | Low |
| RMSprop | ~68% | ~0.95 | Medium ✅ | Moderate |
| Adam | ~71% | ~0.88 | Fast ✅ | Low ✅ |
| Nadam | ~70% | ~0.90 | Fastest ✅ | Low ✅ |

> Results use training subsets (10k MNIST / 8k CIFAR-10) for demo speed.  
> Remove the subset cap in the notebook to run full benchmarks.

---

## 🔍 Key Learnings

| Topic | Observation |
|-------|-------------|
| **SGD** | Best generalization when tuned; poor cold-start without scheduling |
| **Adagrad** | LR vanishes over time — stalls on long training runs |
| **RMSprop** | Reliable choice for RNNs and non-stationary loss surfaces |
| **Adam** | Best all-around default; robust across tasks and architectures |
| **Nadam** | Fastest convergence; Nesterov look-ahead gives consistent edge |
| **LR sensitivity** | Adaptive optimizers tolerate a wider LR range than SGD |
| **Depth + optimizer** | BN + Dropout + Adam/Nadam gives the best combined results |
| **MLP ceiling** | MLP tops out ~55% on CIFAR-10; CNN needed for better results |

---

## 📈 Visualizations Included

- Optimizer paths traced on the **Beale 2D loss landscape**
- Training & validation accuracy/loss curves for all 5 optimizers (overlaid)
- **Learning rate sensitivity** sweep across 7 log-spaced values
- **Convergence speed** bar chart — epochs to reach 80%/85%/90%/92%/94%
- **Radar chart** — multi-criteria normalized comparison
- **Heatmap dashboard** — performance matrix across all metrics

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-teal)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-yellow)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)

---

## 🚀 How to Run

1. Open `optimizer_comparison.ipynb` in **Google Colab**
2. Go to `Runtime → Change runtime type → GPU (T4) → Save`
3. Click `Runtime → Run all`
4. All plots auto-save as `.png` in the session directory

---

## 🔗 Links

- 📓 [Open in Google Colab](https://colab.research.google.com/drive/1FZQOX5NTCjgOTMGUfGR0Ygz4XVA8LP9F?usp=sharing)
- 📊 [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- 📄 [Adam Paper — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- 📄 [Optimizer Overview — Ruder, 2016](https://arxiv.org/abs/1609.04747)
- 🏠 [Back to Portfolio](../README.md)

---

*Part of the [Deep Learning Portfolio](../README.md) by Gauri Shinde*
