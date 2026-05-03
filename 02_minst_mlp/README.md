# Practical 3 — Basic MLP for Handwritten Digit Recognition (MNIST)

> **Skills demonstrated:** MLP design · Multi-class classification · Performance analysis · Hyperparameter tuning  
> **Relevant roles:** ML Engineer · AI Product Manager · ML Consultant · Associate PM

---

## 🎯 Objective

Design and implement a **Multilayer Perceptron (MLP)** from scratch to recognize
handwritten digits (0–9) from the MNIST dataset and perform a thorough
performance analysis including accuracy, loss curves, confusion matrix,
and per-class metrics.

---

## 📂 Files in this Folder

| File | Description |
|------|-------------|
| `03_mnist_mlp.ipynb` | Main notebook — full MLP implementation + analysis |
| `README.md` | This file |

---

## 🗂️ Dataset

**Name:** MNIST Handwritten Digits  
**Source:** Built into `tensorflow.keras.datasets`  
**Size:** 70,000 grayscale images (60k train + 10k test)  
**Image size:** 28 × 28 pixels = 784 features per image  
**Classes:** 10 (digits 0 through 9)

---

## 🧠 MLP Architectures Compared

Three architectures are implemented and compared:

### Model A — Shallow MLP (Baseline)
```
Input (784) → Dense(128, ReLU) → Dense(10, Softmax)
Params: ~101,770
```

### Model B — Medium MLP
```
Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)
Params: ~234,634
```

### Model C — Deep MLP (Best)
```
Input (784) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)
Params: ~535,818
```

---

## ⚙️ Preprocessing Pipeline

```
Raw MNIST images (28×28, uint8, 0–255)
        ↓
Flatten  →  reshape to (784,)
        ↓
Normalize  →  divide by 255.0  →  float32 (0.0–1.0)
        ↓
One-Hot Encode labels  →  [0,0,1,0,0,0,0,0,0,0]
        ↓
Feed into MLP input layer
```

---

## 📊 Results

| Model | Test Accuracy | Test Loss | Parameters | Training Time |
|-------|-------------|-----------|-----------|--------------|
| Model A — Shallow | ~97.2% | ~0.095 | 101,770 | ~30s |
| Model B — Medium  | ~97.8% | ~0.078 | 234,634 | ~45s |
| Model C — Deep    | ~98.2% | ~0.065 | 535,818 | ~60s |

> Results may vary slightly each run due to random weight initialization.

---

## 📈 Performance Analysis Includes

- Training & validation accuracy/loss curves
- Confusion matrix (10×10 heatmap)
- Per-class precision, recall, F1-score
- Most confused digit pairs
- Wrongly predicted samples visualization
- Model comparison bar chart

---

## 🔍 Key Learnings

| Topic | Observation |
|-------|-------------|
| **Depth effect** | Deeper networks improve accuracy but need more time |
| **Flatten layer** | MLP requires 1D input — 28×28 image becomes 784 numbers |
| **Softmax** | Output layer for multi-class — gives probability per class |
| **ReLU** | Best activation for hidden layers — avoids vanishing gradients |
| **Overfitting** | Deeper models overfit without Dropout — validation gap increases |
| **Confusion matrix** | Digits 4/9 and 3/5 are the most commonly confused pairs |
| **Normalization** | Without it, training is ~5× slower and less stable |

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

1. Open `03_mnist_mlp.ipynb` in **Google Colab**
2. Go to `Runtime → Change runtime type → GPU`
3. Click `Runtime → Run all`
4. Dataset downloads automatically — no manual steps needed

---

## 🔗 Links

- 📓 [Open in Google Colab](https://colab.research.google.com/drive/1vsGiFZxnBRGLmWZwZomfDJvt9VMyQaph?usp=sharing)
- 📊 [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)
- 🏠 [Back to Portfolio](../README.md)

---

*Part of the [Deep Learning Portfolio](../README.md) by Gauri Shinde*
