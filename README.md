# 🧠 Diabetic Retinopathy Detection using Deep Learning

## 📌 Overview

This project focuses on detecting **Diabetic Retinopathy (DR)** using deep learning models.
We implement and compare multiple architectures and improve performance using **ensemble learning techniques**, including:

* Averaging Ensemble
* **Stacking (Meta-Learning) Ensemble**

---

## 🎯 Objectives

* Train multiple deep learning models for DR classification
* Compare model performance using standard metrics
* Improve predictions using ensemble techniques
* Explore **meta-learning (stacking)** for better generalization

---

## 🏗️ Models Used

* **EfficientNet-B0** – efficient and lightweight CNN
* **ResNet50** – deep residual network for feature extraction

---

## 🔗 Ensemble Methods

### 1. Averaging Ensemble

Simple combination of model predictions:

```python id="avg01"
final_output = (model1_output + model2_output) / 2
```

---

### 2. Stacking (Meta Model) 🧠

A more advanced ensemble technique where:

* Base models (EfficientNet, ResNet) generate predictions
* These predictions are used as **input features**
* A **meta-model (e.g., Logistic Regression / MLP)** learns how to combine them

**Workflow:**

1. Train base models
2. Collect predictions (probabilities/logits)
3. Train meta-model on these predictions
4. Final prediction = meta-model output

👉 This allows the system to learn **which model to trust more in different cases**

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Cohen’s Kappa

---

## 📈 Outputs

* Training & validation loss curves
* Accuracy plots
* CSV logs
* Model comparison results
* Ensemble vs individual model performance

---

## 🧪 Dataset

Retinal fundus images for diabetic retinopathy detection

*(Specify dataset: APTOS / Kaggle / custom dataset)*

---

## ⚙️ Workflow

1. Data preprocessing
2. Train base models (EfficientNet, ResNet)
3. Evaluate individual models
4. Apply averaging ensemble
5. Apply stacking (meta-model)
6. Compare all approaches

---

## 🚀 Future Improvements

* Add more base models (DenseNet, ViT)
* Optimize meta-model architecture
* Cross-validation stacking
* Real-time deployment (web/app)

---

## 🧠 Research Context

Modern healthcare systems increasingly combine **deep learning + IoT + medical imaging** for early disease detection.
For example, machine learning models combined with imaging data can significantly improve diagnostic accuracy and prediction performance .

---

## 📂 Project Structure

```id="tree02"
├── models/
├── data/
├── results/
├── train.py
├── evaluate.py
├── ensemble.py
├── README.md
```

---

## 💡 Key Insights

* Deep CNNs provide strong baseline performance
* Averaging improves stability
* **Stacking provides the best performance by learning model relationships**
