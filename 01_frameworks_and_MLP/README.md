# Practical 1 — TensorFlow, Keras & PyTorch Frameworks + MLP on Diabetes Dataset

> **Skills demonstrated:** Framework setup · GPU configuration · MLP implementation · Binary classification  
> **Relevant roles:** ML Engineer · AI Product Manager · ML Consultant · Associate PM

---

## 🎯 Objective

Introduce the three major deep learning frameworks — **TensorFlow**, **Keras**, and **PyTorch** —
and understand how to configure GPU support on cloud platforms.  
Apply this knowledge by implementing a **Multilayer Perceptron (MLP)** to predict diabetes
using the Pima Indians Diabetes Dataset.

---

## 📂 Files in this folder

| File | Description |
|------|-------------|
| `01_frameworks_MLP.ipynb` | Main notebook — framework intro + MLP implementation |
| `README.md` | This file |

---

## 🗂️ Dataset

**Name:** Pima Indians Diabetes Dataset  
**Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Size:** 768 samples · 8 input features · 1 binary output  

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **0 = No diabetes · 1 = Diabetes (Target)** |

---

## 🔧 Frameworks Covered

### 1. TensorFlow
- Open-source framework by **Google**
- Best for: Production deployment, scalable systems
- GPU setup on Colab: `Runtime → Change runtime type → GPU`

### 2. Keras
- High-level API that runs **on top of TensorFlow**
- Best for: Beginners, fast prototyping
- Writing models is simpler and more readable

### 3. PyTorch
- Open-source framework by **Meta (Facebook)**
- Best for: Research, custom model building, flexibility
- GPU setup: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

---

## ⚙️ GPU Configuration

### On Google Colab
```python
# Step 1: Runtime → Change runtime type → Select GPU → Save

# Step 2: Verify GPU is available
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# PyTorch check
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
```

### On Kaggle
```
Settings (right panel) → Accelerator → GPU T4 x2 → Save
```

---

## 🧠 MLP Architecture

```
Input Layer     →  8 neurons  (one per feature)
Hidden Layer 1  →  12 neurons  + ReLU activation
Hidden Layer 2  →  8 neurons   + ReLU activation
Output Layer    →  1 neuron    + Sigmoid activation (binary output)
```

---

## 💻 Implementation

### Keras Implementation
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=(8,)),
    layers.Dense(8,  activation='relu'),
    layers.Dense(1,  activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
```

### PyTorch Implementation
```python
import torch
import torch.nn as nn

class DiabetesMLP(nn.Module):
    def __init__(self):
        super(DiabetesMLP, self).__init__()
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model  = DiabetesMLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
```

---

## 📊 Results

## Model Performance Comparison

| Model         | Accuracy | Framework           |
|--------------|---------:|--------------------|
| Keras MLP    | 70.78%   | TensorFlow / Keras |
| PyTorch MLP  | 71.43%   | PyTorch            |

> ✅ Both frameworks achieved similar accuracy — showing that the choice of framework
> affects workflow, not necessarily results.

---

## 🔍 Key Learnings

- **TensorFlow vs PyTorch:** TensorFlow is better for deployment; PyTorch is better for research and flexibility
- **Keras** makes writing neural networks simple — ideal for rapid prototyping
- **GPU vs CPU:** Training was significantly faster with GPU enabled on Colab
- **Sigmoid activation** is used in the output layer for binary classification problems
- **Adam optimizer** converged faster than basic SGD for this dataset

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Keras](https://img.shields.io/badge/Keras-3.x-purple)
![Colab](https://img.shields.io/badge/Google_Colab-GPU-yellow)

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Go to `Runtime → Change runtime type → GPU`
3. Run all cells from top to bottom
4. No local installation needed

---

## 🔗 Links

- 📓 [Open in Google Colab](https://colab.research.google.com/drive/1Qmv_omE8nwXBM9wW-UoYVNNueoThD5nR?usp=sharing)
- 📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 🏠 [Back to Portfolio](../README.md)

---

*Part of the [Deep Learning Portfolio](../README.md) by Gauri Shinde*
