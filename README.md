# Academic Assignment 1: Machine Learning Classification & Clinical Risk Assessment Suite

**Course:** Applied Artificial Intelligence & Machine Learning  
**Course Lecturer:** Mr. Harry Attieku-Boateng  
**Academic Institution:** Pentecost University  
**Submission Deadline:** November 30th  

---

## 🌟 Executive Overview
This repository contains the complete implementation for **Assignment 1**, comprising two distinct machine learning classification tasks:
1. **Part 1: Iris Botanical Species Classification (Scikit-Learn)**  
   Multi-class classification on botanical measurements (`sepal length`, `sepal width`, `petal length`, `petal width`) evaluating Random Forest, achieving **92.11% accuracy** and **92.30% F1-score**.
2. **Part 2: Heart Disease Risk Prediction (OneHotEncoder Pipeline)**  
   Binary classification on a 500-patient clinical dataset predicting heart disease likelihood (`0 = No`, `1 = Yes`). Categorical variables (`sex`, `smoker`, `diabetes`, `family_history`) are encoded using `OneHotEncoder(drop='first', sparse_output=False)` within a Scikit-Learn `ColumnTransformer`.

Both tasks are unified into an interactive, cloud-ready **Streamlit Web Application** for real-time model inference and metric inspection.

---

## 📁 Repository Structure
```
├── app.py                     # Streamlit Interactive Web Application
├── train_and_evaluate.py      # Complete modeling & evaluation pipeline script
├── generate_docx_report.py    # Automated Word (.docx) technical report builder
├── requirements.txt           # Python dependencies
├── data/
│   └── heart_disease.csv      # 500-row clinical patient dataset
├── models/
│   ├── iris_model.joblib      # Serialized Scikit-learn Iris pipeline
│   ├── heart_model.joblib     # Serialized Heart Disease classification pipeline
│   └── metrics_summary.joblib # Evaluation metrics dictionary
└── figures/
    ├── iris_correlation_heatmap.png
    ├── iris_feature_distributions.png
    ├── iris_confusion_matrix.png
    ├── heart_class_balance.png
    ├── heart_numerical_distributions.png
    ├── heart_confusion_matrix.png
    └── heart_roc_curve.png
```

---

## 📊 Summary of Evaluation Metrics

### 1. Iris Species Classification (Part 1 - 10 Marks)
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 92.11% |
| **Macro Precision** | 92.46% |
| **Macro Recall** | 92.31% |
| **Macro F1-Score** | 92.30% |

### 2. Heart Disease Prediction (Part 2 - OneHotEncoder Preprocessing)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Best)** | **55.00%** | **55.56%** | **75.47%** | **64.00%** | **0.510** |
| Random Forest | 53.00% | 54.55% | 67.92% | 60.50% | 0.509 |
| Gradient Boosting | 55.00% | 56.25% | 67.92% | 61.54% | 0.586 |

---

## 🚀 Running the Web Application Locally

```bash
# 1. Clone the repository
git clone https://github.com/s47user/PUIT-Assignment1-ML-Classification.git
cd PUIT-Assignment1-ML-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```
