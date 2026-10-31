# DL-Midterm-Project
# Deep Learning Midterm Project

**Name:** Mehnoor | **Roll No:** SP24-BBD-083
**Program:** BS Business Data Analytics, COMSATS University Islamabad

This project builds two models on two different datasets:

1. **DNN (ANN):** predicting bank term deposit subscription (tabular data)
2. **CNN:** detecting cotton leaf diseases (image data)

Both models, including training, evaluation and inference, are in one notebook: `DL_Midterm_Project(SP24-BBD-083).ipynb`.

---

## Part 1: DNN, Bank Marketing Term Deposit Prediction

**Problem:** Banks spend heavily on telemarketing, but only about 12% of customers subscribe to a term deposit. The goal is to predict which customers will subscribe, so calls can be targeted and marketing cost reduced.

**Dataset:** UCI Bank Marketing (`bank-full.csv`), about 45,000 customers, 16 features, target `y` (yes/no).
Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

**Approach:**
- Dropped `duration` (only known after the call, so it causes data leakage)
- One-hot encoded categorical features, StandardScaler fitted on the training set only
- Stratified 70/15/15 train/validation/test split
- Balanced class weights for the imbalanced target

**Architecture:** Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
Adam optimizer, binary cross-entropy loss, EarlyStopping.

**Results (test set, 6,782 customers):**

| Metric | Value |
|---|---|
| Accuracy | 82.88% |
| Precision (Yes) | 0.359 |
| Recall (Yes) | 0.593 |
| F1-score (Yes) | 0.447 |
| ROC-AUC | 0.783 |

The model identified 59.3% of customers who actually subscribed. Accuracy is below the 88% of an "always No" model because class weights favor finding the minority "Yes" class, so recall, F1 and ROC-AUC are the main metrics.

**Inference:** the trained model, scaler, feature list and threshold are saved in `models/dnn_bank_marketing.joblib`. The notebook reloads this file and predicts the subscription probability for new customers using the same preprocessing as in training.

---

## Part 2: CNN, Cotton Leaf Disease Detection

**Problem:** Cotton is a major crop in Pakistan, and diseases cause large yield losses. Manual inspection is slow and needs experts. The goal is to classify a cotton leaf image into one of 4 classes for early detection.

**Dataset:** Kaggle Cotton Leaf Disease dataset, 1,709 images, 4 classes: bacterial blight, curl virus, fusarium wilt and healthy.
Source: https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset

**Approach:**
- The dataset had no predefined split, so a stratified 70/15/15 split was created (1,196 / 256 / 257 images)
- Images resized to 128×128 and rescaled to 0-1
- Data augmentation (flip, rotation, zoom, contrast) on training data only

**Architecture:** Augmentation → Rescaling → 3 × [Conv2D (32/64/128) + MaxPooling] → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(4, Softmax)
Adam optimizer, sparse categorical cross-entropy loss, EarlyStopping and ReduceLROnPlateau.

**Results (test set, 257 images):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Bacterial blight | 0.92 | 0.99 | 0.95 |
| Curl virus | 0.97 | 0.90 | 0.93 |
| Fusarium wilt | 0.94 | 0.97 | 0.95 |
| Healthy | 0.93 | 0.89 | 0.91 |
| **Overall accuracy** | | | **93.77%** |

**Inference:** the trained CNN is saved in `models/cnn_cotton_disease.joblib`. The notebook reloads it and classifies new leaf images (resized to 128×128), showing the predicted class and confidence.

---

## Repository Contents

| File | Description |
|---|---|
| `DL_Midterm_Project(SP24-BBD-083).ipynb` | Full code, plots, results and inference for both models |
| `Project_Report` | Project document (problem, dataset, solution, preprocessing, architecture, results, conclusion) |
| `models/*.joblib` | Saved DNN and CNN models with preprocessing information for inference |
| `images/` | Saved plots (training curves, confusion matrices, ROC curve, inference examples) |
| `requirements.txt` | Python libraries used |



## Libraries

TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, joblib.
