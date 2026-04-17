# 🧠 Diabetic Retinopathy Detection (PyTorch + EfficientNet)

This project builds a deep learning model to detect **diabetic retinopathy severity** from retinal images using **PyTorch** and **EfficientNet**.

---

## 🚀 Features

* 📊 Multi-class classification (5 severity levels)
* ⚖️ Handles class imbalance using **Weighted Random Sampling**
* 🧠 Uses **EfficientNet-B0 (pretrained)** for strong feature extraction
* ⚡ GPU acceleration with CUDA
* 📈 Evaluation with confusion matrix & classification report

---

## 📂 Dataset

Dataset used:
👉 Kaggle Diabetic Retinopathy Detection (resized version)

Structure:

```
archive/
│── trainLabels.csv
│── resized_train/
    └── resized_train/
        ├── 10003_left.jpeg
        ├── 10003_right.jpeg
        └── ...
```

* Images are stored in a single folder
* Labels are provided via CSV (`trainLabels.csv`)

---

## 🏷️ Classes

| Label | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild             |
| 2     | Moderate         |
| 3     | Severe           |
| 4     | Proliferative DR |

---

## ⚙️ Installation

### 1. Create environment

```bash
conda create -n rapids_clean python=3.10 -y
conda activate rapids_clean
```

### 2. Install dependencies

```bash
pip install torch torchvision pandas numpy scikit-learn tqdm pillow
```

---

## 🧠 Model Architecture

* Backbone: **EfficientNet-B0 (pretrained on ImageNet)**
* Final layer modified for 5-class classification

```python
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
```

---

## ⚖️ Handling Class Imbalance

This project uses:

✅ **WeightedRandomSampler**
❌ No weighted loss (avoids over-correction)

---

## 🏃 Training

Run training:

```bash
python train_diabetes.py
```

### Default settings:

* Batch size: 8
* Epochs: 15
* Optimizer: Adam
* Learning rate: 0.0003

---

## 📊 Evaluation

The model outputs:

* Confusion Matrix
* Classification Report (precision, recall, F1-score)
* Accuracy

---

## 🧠 Key Learnings

* Medical datasets are **highly imbalanced**
* Using both weighted loss + sampler can **break training**
* EfficientNet provides strong performance with low VRAM

---

## ⚠️ Hardware Requirements

* GPU recommended (4GB+ VRAM)
* EfficientNet-B0 works on low-memory GPUs
* EfficientNet-B4 requires 6–8GB VRAM

---

## 📈 Future Improvements

* 🔥 Grad-CAM visualization (lesion highlighting)
* 🔥 EfficientNet-B4 / B5 (if higher VRAM available)
* 🔥 Ensemble models
* 🔥 Advanced preprocessing (CLAHE)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

This project is for educational and research purposes.
