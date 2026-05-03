# Practical 2 — Dataset Handling: Loading, Preprocessing & Visualizing MNIST and CIFAR-10

> **Skills demonstrated:** Data loading · Preprocessing · Normalization · Visualization · EDA  
> **Relevant roles:** ML Engineer · AI Product Manager · ML Consultant · Associate PM

---

## 🎯 Objective

Understand how to handle image datasets end-to-end:
- **Load** MNIST and CIFAR-10 datasets using TensorFlow/Keras
- **Preprocess** images — normalization, reshaping, one-hot encoding
- **Visualize** samples, class distributions, and pixel statistics
- **Compare** both datasets — structure, complexity, and use cases

---

## 📂 Files in this Folder

| File | Description |
|------|-------------|
| `02_dataset_handling.ipynb` | Main notebook — full implementation |
| `README.md` | This file |

---

## 🗂️ Datasets Overview

### MNIST
| Property | Value |
|----------|-------|
| Type | Grayscale handwritten digits |
| Classes | 10 (digits 0–9) |
| Image Size | 28 × 28 pixels |
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Total Size | ~11 MB |

### CIFAR-10
| Property | Value |
|----------|-------|
| Type | Color images (RGB) |
| Classes | 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck) |
| Image Size | 32 × 32 × 3 pixels |
| Training Samples | 50,000 |
| Test Samples | 10,000 |
| Total Size | ~163 MB |

---

## ⚙️ Preprocessing Steps Applied

### MNIST
```
Raw pixel values (0–255)
        ↓
Normalize → divide by 255.0 → values become (0.0 – 1.0)
        ↓
Reshape → (28, 28) → (28, 28, 1) for CNN compatibility
        ↓
One-Hot Encode labels → [0,0,1,0,...] for classification
```

### CIFAR-10
```
Raw pixel values (0–255) with 3 channels (RGB)
        ↓
Normalize → divide by 255.0 → values become (0.0 – 1.0)
        ↓
Shape already (32, 32, 3) — ready for CNN
        ↓
One-Hot Encode labels → [0,0,0,1,0,...] for classification
```

---

## 📊 Results & Visualizations

| Visualization | Description |
|---------------|-------------|
| Sample grid | 5×10 grid showing all 10 classes |
| Class distribution | Bar chart — balanced vs imbalanced check |
| Pixel intensity | Histogram of raw vs normalized values |
| Mean image per class | Average image for each class |
| RGB channel split | CIFAR-10 R, G, B channels separately |

---

## 🔍 Key Learnings

- **MNIST** is simpler — grayscale, centered digits, high contrast → good for beginners
- **CIFAR-10** is harder — color images, varied backgrounds, low resolution → realistic challenge
- **Normalization** is critical — raw pixel values (0–255) cause slow training and poor convergence
- **One-hot encoding** converts class labels into vectors needed by softmax output layers
- **Class balance** — both datasets are perfectly balanced (6000/5000 samples per class)
- **Data shape matters** — CNN expects (height, width, channels); MLP expects flattened (784,) or (3072,)

---

## 📈 Dataset Comparison

| Feature | MNIST | CIFAR-10 |
|---------|-------|----------|
| Color | Grayscale | RGB |
| Size | 28×28 | 32×32 |
| Channels | 1 | 3 |
| Difficulty | Easy | Medium |
| Baseline Accuracy (MLP) | ~98% | ~50% |
| Baseline Accuracy (CNN) | ~99% | ~75% |
| Real-world relevance | Low | Medium |

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-teal)

---

## 🚀 How to Run

1. Open notebook in **Google Colab**
2. Go to `Runtime → Change runtime type → GPU`
3. Run all cells — datasets download automatically (no manual upload needed)
4. All visualizations save as `.png` files

---

## 🔗 Links

- 📓 [Open in Google Colab](https://colab.research.google.com/drive/19_Nh0B1AlEUddPtaDDP5di_LYENV85CI?usp=sharing))
- 📊 [MNIST on Yann LeCun's site](http://yann.lecun.com/exdb/mnist/)
- 📊 [CIFAR-10 on cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html)
- 🏠 [Back to Portfolio](../README.md)

---

*Part of the [Deep Learning Portfolio](../README.md) by Gauri Shinde*
