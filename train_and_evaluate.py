import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# Set visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

print("="*70)
print("ACADEMIC ASSIGNMENT 1: MACHINE LEARNING MODELING PIPELINE")
print("Student: Emmanuel Siaw Asante | ID: PYT230047")
print("Lecturer: Mr. Harry Attieku-Boateng")
print("="*70)

# ==============================================================================
# PART 1: SCIKIT-LEARN IRIS CLASSIFICATION TASK (10 MARKS)
# ==============================================================================
print("\n[PART 1] Training & Evaluating Iris Classification Model...")
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_features = iris.feature_names
iris_target = "target"

# 1. EDA Visualizations
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Figure 1.1: Iris Feature Correlation Heatmap", fontsize=12, pad=12, fontweight="bold")
plt.savefig(os.path.join(FIGS_DIR, "iris_correlation_heatmap.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, feature in enumerate(iris_features, 1):
    plt.subplot(2, 2, i)
    for target_idx, target_name in enumerate(iris.target_names):
        subset = iris_df[iris_df[iris_target] == target_idx]
        sns.kdeplot(subset[feature], label=target_name, fill=True, alpha=0.3)
    plt.title(f"Distribution: {feature}", fontsize=10)
    plt.xlabel("cm")
    plt.ylabel("Density")
    if i == 1:
        plt.legend(loc="upper right", fontsize=8)
plt.suptitle("Figure 1.2: Iris Feature Distributions by Target Class", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "iris_feature_distributions.png"), dpi=300)
plt.close()

# 2. Train-Test Split
X_iris = iris_df[iris_features]
y_iris = iris_df[iris_target]
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris, y_iris, test_size=0.25, random_state=42, stratify=y_iris
)

# 3. Model Pipeline
iris_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
])

iris_pipeline.fit(X_iris_train, y_iris_train)
y_iris_pred = iris_pipeline.predict(X_iris_test)

# 4. Evaluation Metrics
iris_acc = accuracy_score(y_iris_test, y_iris_pred)
iris_prec = precision_score(y_iris_test, y_iris_pred, average="macro")
iris_rec = recall_score(y_iris_test, y_iris_pred, average="macro")
iris_f1 = f1_score(y_iris_test, y_iris_pred, average="macro")
iris_cm = confusion_matrix(y_iris_test, y_iris_pred)

print(f"  Accuracy : {iris_acc * 100:.2f}%")
print(f"  Precision: {iris_prec * 100:.2f}%")
print(f"  Recall   : {iris_rec * 100:.2f}%")
print(f"  F1-Score : {iris_f1 * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_iris_test, y_iris_pred, target_names=iris.target_names))

# Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(iris_cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
plt.xlabel("Predicted Species", fontsize=11, fontweight="bold")
plt.ylabel("Actual Species", fontsize=11, fontweight="bold")
plt.title("Figure 1.3: Iris Model Confusion Matrix", fontsize=12, pad=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "iris_confusion_matrix.png"), dpi=300)
plt.close()

# Save Iris Pipeline
joblib.dump(iris_pipeline, os.path.join(MODELS_DIR, "iris_model.joblib"))
print("  ✓ Iris Model saved successfully.")


# ==============================================================================
# PART 2: HEART DISEASE CLASSIFICATION TASK (ONEHOTENCODER REQUIREMENT)
# ==============================================================================
print("\n[PART 2] Training & Evaluating Heart Disease Classification Model...")
heart_csv_path = os.path.join(DATA_DIR, "heart_disease.csv")
heart_df = pd.read_csv(heart_csv_path)

print(f"  Dataset Shape: {heart_df.shape[0]} rows, {heart_df.shape[1]} columns")
print("  Target Distribution:\n", heart_df["heart_disease"].value_counts())

# Identify feature categories
categorical_cols = ["sex", "smoker", "diabetes", "family_history"]
numerical_cols = ["age", "blood_pressure", "cholesterol", "bmi", "physical_activity", "stress_level"]
target_col = "heart_disease"

# 1. EDA Visualizations
plt.figure(figsize=(8, 5))
ax = sns.countplot(x=target_col, data=heart_df, palette=["#2ecc71", "#e74c3c"])
plt.title("Figure 2.1: Heart Disease Diagnosis Class Balance", fontsize=12, pad=12, fontweight="bold")
plt.xlabel("Heart Disease (0 = Absence, 1 = Presence)", fontsize=11)
plt.ylabel("Patient Count", fontsize=11)
plt.xticks([0, 1], ["Negative (0)", "Positive (1)"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(heart_df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "heart_class_balance.png"), dpi=300)
plt.close()

# EDA: Numerical features boxplots by diagnosis
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for idx, col in enumerate(numerical_cols):
    sns.boxplot(x=target_col, y=col, data=heart_df, ax=axes[idx], palette=["#3498db", "#e67e22"])
    axes[idx].set_title(f"{col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Heart Disease (0=No, 1=Yes)")
    axes[idx].set_ylabel("Value")
plt.suptitle("Figure 2.2: Distribution of Clinical Measures by Heart Disease Status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "heart_numerical_distributions.png"), dpi=300)
plt.close()

# 2. Train-Test Split
X_heart = heart_df.drop(columns=[target_col])
y_heart = heart_df[target_col]

X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
    X_heart, y_heart, test_size=0.20, random_state=42, stratify=y_heart
)

# 3. Explicit OneHotEncoder & Preprocessing Pipeline
# Lecturer mandate: "There is something called OneHotEncoder. Go and read about it and use it."
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
    ]
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
}

comparison_results = []
best_model_name = None
best_f1 = 0.0
best_pipeline = None

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_heart_train, y_heart_train)
    y_pred = pipe.predict(X_heart_test)
    y_proba = pipe.predict_proba(X_heart_test)[:, 1]
    
    acc = accuracy_score(y_heart_test, y_pred)
    prec = precision_score(y_heart_test, y_pred)
    rec = recall_score(y_heart_test, y_pred)
    f1 = f1_score(y_heart_test, y_pred)
    auc = roc_auc_score(y_heart_test, y_proba)
    
    comparison_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = pipe

results_df = pd.DataFrame(comparison_results)
print("\nModel Comparison Table for Heart Disease:\n", results_df.to_string(index=False))

# Re-evaluate best model
y_best_pred = best_pipeline.predict(X_heart_test)
y_best_proba = best_pipeline.predict_proba(X_heart_test)[:, 1]
heart_cm = confusion_matrix(y_heart_test, y_best_pred)

# Save Heart Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(heart_cm, annot=True, fmt="d", cmap="Reds", 
            xticklabels=["No Disease (0)", "Disease (1)"], 
            yticklabels=["No Disease (0)", "Disease (1)"],
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
plt.xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
plt.ylabel("Actual Diagnosis", fontsize=11, fontweight="bold")
plt.title(f"Figure 2.3: Heart Disease Confusion Matrix ({best_model_name})", fontsize=12, pad=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "heart_confusion_matrix.png"), dpi=300)
plt.close()

# Save ROC Curve
fpr, tpr, _ = roc_curve(y_heart_test, y_best_proba)
best_auc = roc_auc_score(y_heart_test, y_best_proba)

plt.figure(figsize=(7, 5.5))
plt.plot(fpr, tpr, color="#c0392b", lw=2.5, label=f"{best_model_name} (AUC = {best_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title("Figure 2.4: Receiver Operating Characteristic (ROC) Curve", fontsize=12, pad=12, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "heart_roc_curve.png"), dpi=300)
plt.close()

# Save Best Heart Disease Pipeline
joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "heart_model.joblib"))
print(f"  ✓ Best Heart Disease Model ({best_model_name}) saved successfully.")

# Save Metrics Summary to JSON for Report Generator
metrics_summary = {
    "iris": {
        "accuracy": float(iris_acc),
        "precision": float(iris_prec),
        "recall": float(iris_rec),
        "f1": float(iris_f1),
        "cm": iris_cm.tolist()
    },
    "heart": {
        "best_model": best_model_name,
        "comparison": comparison_results,
        "cm": heart_cm.tolist(),
        "auc": float(best_auc)
    }
}
joblib.dump(metrics_summary, os.path.join(MODELS_DIR, "metrics_summary.joblib"))
print("\nAll models and figures successfully generated and saved!")
print("="*70)
