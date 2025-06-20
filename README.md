# Peer Influence Risk Classification Model

This project explores the implementation of machine learning models, including neural networks and classical ML algorithms, to classify the level of peer influence risk among university students based on behavioral and social factors.

 Dataset
- Source: [Factors affecting university student grades dataset]:https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades
- Mission Alignment: The dataset reflects real-world educational and behavioral challenges, supporting our mission to apply AI for improving student wellbeing and academic support.
- Target Label: `Peer_Risk_Level` — derived from peer-related attributes such as bullying, motivation, social media time, and peer group.

---

Here is the link to my youtube video:https://youtu.be/B4HbbOquJF4

Models Implemented

 Neural Network Instances (4 Total)

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers         | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|----------|-----------|-------------|--------|----------------|----------------|----------------|----------|----------|--------|-----------|
| 1        | adam      | None        | 10     | No             | 2              | default        | ✓        | ✓        | ✓      | ✓         |
| 2        | adam      | L2          | 50     | Yes            | 2              | default        | ✓        | ✓        | ✓      | ✓         |
| 3        | rmsprop   | Dropout     | 50     | Yes            | 2 + Dropout    | default        | ✓        | ✓        | ✓      | ✓         |
| 4        | sgd       | Dropout     | 60     | Yes            | 3 + Dropout    | 0.01           | ✓        | ✓        | ✓      | ✓         |


Best Neural Network Instance: Instance 4 — due to a deeper architecture, added dropout regularization, and tuned learning rate, it showed superior generalization.

---

Classical ML Models

| Instance   | Optimizer     | Regularizer | Epochs | Early Stopping | Layers | LR              | Accuracy | F1 Score | Recall | Precision |
|------------|---------------|-------------|--------|----------------|--------|------------------|----------|----------|--------|-----------|
| LogReg     | liblinear     | L2          | n/a    | No             | n/a    | n/a              | ✓        | ✓        | ✓      | ✓         |
| SVM        | n/a           | n/a         | n/a    | No             | n/a    | n/a              | ✓        | ✓        | ✓      | ✓         |
| XGBoost    | tree booster  | L1 & L2     | auto   | No             | n/a    | 0.3 (default)    | ✓        | ✓        | ✓      | ✓         |


Best ML Algorithm**: XGBoost — performed slightly better due to its ability to handle non-linearity and class imbalance.

---
Summary

- Which combination worked better?
  - The neural network model in Instance 4 outperformed others with improved F1 and precision, due to tuning of learning rate, layer depth, and dropout regularization.

- Which implementation performed better overall?
  - While XGBoost was competitive, the optimized Neural Network (Instance 4) achieved the highest performance on unseen data.

---

Repository Structure



PeerRiskModel/
├── notebook.ipynb
├── README.md
├── saved_models/
│ ├── instance1.h5
│ ├── instance2.h5
│ ├── instance3.h5
│ ├── instance4.h5
│ ├── logistic_regression.pkl
│ ├── svm_model.pkl
│ └── xgboost_model.pkl



---

Instructions to Run

1. Clone the repo and open `notebook.ipynb`.
2. Run all cells step by step.
3. Models will automatically save into the `saved_models/` folder.
4. You can re-load the best models using `tf.keras.models.load_model()` or `joblib.load()`.

Example (for neural network):
model = tf.keras.models.load_model("saved_models/instance4.h5")

Example (for classical model):
import joblib
model = joblib.load("saved_models/xgboost_model.pkl")
