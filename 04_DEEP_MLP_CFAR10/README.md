# Practical 4 — Deep MLP for Multi-Class Classification (CIFAR-10)

> **Skills demonstrated:** Deep MLP design · Network depth study · Multi-class classification · Regularization · Performance analysis  
> **Relevant roles:** ML Engineer · AI Product Manager · ML Consultant · Associate PM

---

## 🎯 Objective

Develop progressively deeper MLP models on the **CIFAR-10** dataset and
systematically study how **network depth** affects:
- Classification accuracy
- Overfitting behaviour
- Training time and convergence
- Effect of Dropout and Batch Normalization at different depths

---

## 📂 Files in this Folder

| File | Description |
|------|-------------|
| `04_cifar10_deep_mlp.ipynb` | Main notebook — full implementation + depth study |
| `README.md` | This file |

---

## 🗂️ Dataset

**Name:** CIFAR-10  
**Source:** Built into `tensorflow.keras.datasets`  
**Size:** 60,000 color images (50k train + 10k test)  
**Image size:** 32 × 32 × 3 = **3,072 features** per image (after flattening)  
**Classes:** 10

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | Airplane | 5 | Dog |
| 1 | Automobile | 6 | Frog |
| 2 | Bird | 7 | Horse |
| 3 | Cat | 8 | Ship |
| 4 | Deer | 9 | Truck |

---

## 🧠 Models Compared (5 Architectures)

| Model | Hidden Layers | Neurons | Dropout | BatchNorm | Params |
|-------|-------------|---------|---------|-----------|--------|
| M1 — Shallow | 1 | 512 | ❌ | ❌ | ~1.6M |
| M2 — Medium  | 2 | 1024→512 | ❌ | ❌ | ~4.7M |
| M3 — Deep    | 3 | 1024→512→256 | ❌ | ❌ | ~5.3M |
| M4 — Deep + Dropout | 3 | 1024→512→256 | ✅ | ❌ | ~5.3M |
| M5 — Deep + BN + Dropout | 3 | 1024→512→256 | ✅ | ✅ | ~5.3M |

---

## ⚙️ Preprocessing Pipeline

```
Raw CIFAR-10 images  (50000, 32, 32, 3)  uint8  0–255
        ↓
Flatten  →  (50000, 3072)
        ↓
Normalize  →  / 255.0  →  float32  0.0–1.0
        ↓
One-Hot Encode labels  →  (50000, 10)
        ↓
Feed into MLP
```

---

## 📊 Results

| Model | Test Accuracy | Test Loss | Overfitting? |
|-------|-------------|-----------|-------------|
| M1 — Shallow | ~47% | ~1.52 | Low |
| M2 — Medium | ~50% | ~1.44 | Moderate |
| M3 — Deep (no reg) | ~51% | ~1.42 | High ⚠️ |
| M4 — Deep + Dropout | ~53% | ~1.38 | Reduced ✅ |
| M5 — Deep + BN + Dropout | ~55% | ~1.30 | Minimal ✅ |

> CIFAR-10 with MLP is inherently limited — CNN (Practical 6) achieves ~75%+.
> The goal here is to study depth impact, not maximize accuracy.

---

## 🔍 Key Learnings

| Topic | Observation |
|-------|-------------|
| **Depth alone** | More layers → slightly better accuracy BUT bigger overfitting gap |
| **Dropout** | Reduces overfitting significantly — val/train gap narrows |
| **Batch Normalization** | Stabilizes training, faster convergence, higher final accuracy |
| **CIFAR vs MNIST** | CIFAR-10 is much harder — spatial info lost when flattening |
| **MLP ceiling** | MLP tops out ~55% on CIFAR-10; CNN needed for better results |
| **Training time** | Deeper models take longer — diminishing accuracy returns |
| **Best combo** | Depth + BatchNorm + Dropout gives best results for MLP |

---

## 📈 Visualizations Included

- Training/validation accuracy & loss curves per model
- All 5 models overlaid on same plot
- Confusion matrix for best model
- Per-class F1 score bar chart
- Overfitting analysis (train-val gap vs depth)
- Sample wrong predictions

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-teal)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-yellow)

---

## 🚀 How to Run

1. Open `04_cifar10_deep_mlp.ipynb` in **Google Colab**
2. Go to `Runtime → Change runtime type → GPU → Save`
3. Click `Runtime → Run all`
4. CIFAR-10 loads via `tensorflow_datasets` (reliable — no Toronto server issues)

---

## 🔗 Links

- 📓 [Open in Google Colab](https://colab.research.google.com/drive/184HG4-3BtwruPTPsg4kKrdIcDqUWdG5R?usp=sharing)
- 📊 [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- 🏠 [Back to Portfolio](../README.md)

---

*Part of the [Deep Learning Portfolio](../README.md) by Gauri Shinde*
