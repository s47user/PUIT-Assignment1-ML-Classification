import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Suite | Emmanuel Siaw Asante",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")

# Load Models
@st.cache_resource
def load_models():
    iris_path = os.path.join(MODELS_DIR, "iris_model.joblib")
    heart_path = os.path.join(MODELS_DIR, "heart_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.joblib")
    
    iris_model = joblib.load(iris_path) if os.path.exists(iris_path) else None
    heart_model = joblib.load(heart_path) if os.path.exists(heart_path) else None
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None
    return iris_model, heart_model, metrics

iris_model, heart_model, metrics = load_models()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Academic ML Suite")
    st.markdown("**Student:** Emmanuel Siaw Asante  \n**Student ID:** `PYT230047`  \n**Lecturer:** Mr. Harry Attieku-Boateng")
    st.divider()
    
    navigation = st.radio(
        "Navigation",
        [
            "🌸 Task 1: Iris Flower Classifier",
            "❤️ Task 2: Heart Disease Predictor",
            "📊 Model Evaluation & Metrics",
            "📑 Academic Brief & System Info"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("""
    **Mandatory Submission Requirements:**
    * ✅ Scikit-Learn Model Architecture
    * ✅ OneHotEncoder Categorical Processing
    * ✅ Web Application Inference Engine
    * ✅ GitHub Source Repository
    """)

# -----------------------------------------------------------------------------
# TASK 1: IRIS FLOWER CLASSIFIER
# -----------------------------------------------------------------------------
if navigation == "🌸 Task 1: Iris Flower Classifier":
    st.header("🌸 Task 1: Iris Flower Species Classification")
    st.caption("Built using Scikit-Learn `load_iris()` dataset & Random Forest Pipeline (10 Marks).")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Input Morphological Features (cm)")
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
        petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.6, value=1.3, step=0.1)
        
        predict_iris_btn = st.button("Classify Iris Species", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Classification Outcome")
        if predict_iris_btn and iris_model:
            input_df = pd.DataFrame([{
                "sepal length (cm)": sepal_length,
                "sepal width (cm)": sepal_width,
                "petal length (cm)": petal_length,
                "petal width (cm)": petal_width
            }])
            
            prediction = iris_model.predict(input_df)[0]
            probabilities = iris_model.predict_proba(input_df)[0]
            species_names = ["Setosa", "Versicolor", "Virginica"]
            predicted_species = species_names[prediction]
            
            st.success(f"### Predicted Species: **Iris {predicted_species}**")
            st.metric("Model Confidence", f"{probabilities[prediction]*100:.1f}%")
            
            # Probability chart
            prob_df = pd.DataFrame({
                "Species": species_names,
                "Probability (%)": [p * 100 for p in probabilities]
            })
            st.bar_chart(prob_df.set_index("Species"))
        else:
            st.info("Adjust the sliders on the left and click **'Classify Iris Species'** to run live model inference.")
            
    st.divider()
    with st.expander("🔍 View Iris Model Performance Summary"):
        if metrics:
            m = metrics["iris"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{m['accuracy']*100:.2f}%")
            c2.metric("Macro Precision", f"{m['precision']*100:.2f}%")
            c3.metric("Macro Recall", f"{m['recall']*100:.2f}%")
            c4.metric("Macro F1-Score", f"{m['f1']*100:.2f}%")


# -----------------------------------------------------------------------------
# TASK 2: HEART DISEASE RISK PREDICTOR
# -----------------------------------------------------------------------------
elif navigation == "❤️ Task 2: Heart Disease Predictor":
    st.header("❤️ Task 2: Heart Disease Risk Prediction")
    st.caption("Binary Classification on 500 clinical patient observations using an integrated `OneHotEncoder` Pipeline.")
    
    with st.form("heart_form"):
        st.subheader("Patient Clinical & Demographic Indicators")
        
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        with row1_col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=52)
        with row1_col2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
        with row1_col3:
            blood_pressure = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=125)
        with row1_col4:
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=120, max_value=350, value=210)
            
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        with row2_col1:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=27.4, step=0.1)
        with row2_col2:
            smoker = st.selectbox("Smoker Status", ["No", "Yes"])
        with row2_col3:
            diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
        with row2_col4:
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            physical_activity = st.slider("Physical Activity Level (0 = Sedentary, 4 = Very Active)", 0, 4, 2)
        with row3_col2:
            stress_level = st.slider("Reported Stress Level (1 = Low, 10 = Severe)", 1, 10, 5)
            
        submit_btn = st.form_submit_button("Assess Heart Disease Risk", type="primary", use_container_width=True)

    if submit_btn and heart_model:
        patient_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "bmi": bmi,
            "smoker": smoker,
            "diabetes": diabetes,
            "physical_activity": physical_activity,
            "family_history": family_history,
            "stress_level": stress_level
        }])
        
        prediction = heart_model.predict(patient_data)[0]
        probabilities = heart_model.predict_proba(patient_data)[0]
        risk_probability = probabilities[1] * 100
        
        st.subheader("Diagnostic Risk Evaluation")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 1:
                st.error(f"### Result: High Likelihood of Heart Disease (Risk Score: {risk_probability:.1f}%)")
                st.write("Clinical advisory: The model detects elevated cardiovascular risk factors. Diagnostic confirmation and lifestyle review are recommended.")
            else:
                st.success(f"### Result: Low Likelihood of Heart Disease (Risk Score: {risk_probability:.1f}%)")
                st.write("Clinical advisory: The patient's risk profile remains within non-elevated parameters based on the current feature set.")
                
        with res_col2:
            st.metric("Predicted Disease Probability", f"{risk_probability:.1f}%")
            st.progress(risk_probability / 100.0)


# -----------------------------------------------------------------------------
# TASK 3: MODEL EVALUATION & METRICS
# -----------------------------------------------------------------------------
elif navigation == "📊 Model Evaluation & Metrics":
    st.header("📊 Model Evaluation & Validation Metrics")
    st.caption("Comprehensive empirical metrics and confusion matrices as required by Mr. Harry Attieku-Boateng.")
    
    tab1, tab2 = st.tabs(["🌸 Iris Classification Metrics", "❤️ Heart Disease Metrics & ROC Analysis"])
    
    with tab1:
        st.subheader("Iris Model (Random Forest Classifier)")
        if metrics:
            m = metrics["iris"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            c2.metric("Macro Precision", f"{m['precision']*100:.2f}%")
            c3.metric("Macro Recall", f"{m['recall']*100:.2f}%")
            c4.metric("Macro F1-Score", f"{m['f1']*100:.2f}%")
            
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            cm_img_path = os.path.join(FIGS_DIR, "iris_confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Figure 1.3: Iris Model Confusion Matrix", use_container_width=True)
        with fig_col2:
            corr_img_path = os.path.join(FIGS_DIR, "iris_correlation_heatmap.png")
            if os.path.exists(corr_img_path):
                st.image(corr_img_path, caption="Figure 1.1: Iris Feature Correlation Heatmap", use_container_width=True)

    with tab2:
        st.subheader("Heart Disease Benchmark (Logistic Regression, Random Forest, Gradient Boosting)")
        if metrics:
            comp_df = pd.DataFrame(metrics["heart"]["comparison"])
            comp_df["Accuracy"] = comp_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["Precision"] = comp_df["Precision"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["Recall"] = comp_df["Recall"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["F1-Score"] = comp_df["F1-Score"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["ROC-AUC"] = comp_df["ROC-AUC"].apply(lambda x: f"{x:.3f}")
            st.dataframe(comp_df, use_container_width=True)
            
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            h_cm_path = os.path.join(FIGS_DIR, "heart_confusion_matrix.png")
            if os.path.exists(h_cm_path):
                st.image(h_cm_path, caption="Figure 2.3: Heart Disease Confusion Matrix", use_container_width=True)
        with h_col2:
            h_roc_path = os.path.join(FIGS_DIR, "heart_roc_curve.png")
            if os.path.exists(h_roc_path):
                st.image(h_roc_path, caption="Figure 2.4: Receiver Operating Characteristic (ROC) Curve", use_container_width=True)


# -----------------------------------------------------------------------------
# TASK 4: ACADEMIC BRIEF & SYSTEM INFO
# -----------------------------------------------------------------------------
elif navigation == "📑 Academic Brief & System Info":
    st.header("📑 Academic Assessment & Engineering Specifications")
    
    st.markdown(f"""
    ### Assessment Metadata
    * **Student Name:** Emmanuel Siaw Asante
    * **Student Index Number / ID:** `PYT230047`
    * **Course Instructor:** Mr. Harry Attieku-Boateng
    * **Submission Deadline:** November 30th
    
    ### Preprocessing & Architecture Highlights
    1. **Scikit-Learn Iris Task:** Multi-class classification using botanical morphological features, achieving >92% test accuracy.
    2. **OneHotEncoder Mandate:** As strictly instructed during the lecture, categorical variables (`sex`, `smoker`, `diabetes`, `family_history`) were preprocessed using `OneHotEncoder(drop='first', sparse_output=False)` within an automated Scikit-Learn `ColumnTransformer` to prevent dummy variable trap and data leakage.
    3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
    4. **Deployment Architecture:** Lightweight, responsive Streamlit web application with persistent joblib model pipelines ready for production cloud hosting.
    """)
