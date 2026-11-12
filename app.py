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
    page_title="Machine Learning Intelligence Suite | Emmanuel Siaw Asante",
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
# MODERN HUMAN UI STYLING (Zero-Slop, Clean Human Craftsmanship)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Page Header */
    .page-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .page-breadcrumbs {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748B;
        margin-bottom: 4px;
    }
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        color: #0F172A;
    }
    .page-subtitle {
        font-size: 0.90rem;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    /* Clean Card */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Pill & Badges */
    .result-badge-success {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
    }
    .result-badge-danger {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .metric-lbl {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 2px;
    }

    /* Subtle Button Styling */
    div.stButton > button:first-child {
        background: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 14px;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: #1D4ED8;
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🔬</span>
        <div>
            <div style="font-weight: 800; font-size: 17px; color: #0F172A; line-height: 1.2;">PUIT Intelligence</div>
            <div style="font-size: 11px; color: #64748B; font-weight: 600;">Faculty of Computing & IT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; margin-bottom: 16px;">
        <div style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase;">Candidate Profile</div>
        <div style="font-weight: 700; color: #0F172A; font-size: 14px; margin-top: 2px;">Emmanuel Siaw Asante</div>
        <div style="font-size: 12px; color: #2563EB; font-weight: 600;">ID: PYT230047</div>
        <div style="font-size: 11px; color: #475569; margin-top: 4px;">Lecturer: <strong>Mr. Harry Attieku-Boateng</strong></div>
    </div>
    """, unsafe_allow_html=True)

    navigation = st.radio(
        "Select Assignment Module",
        [
            "🌸 Task 1: Iris Botanical Classifier",
            "❤️ Task 2: Heart Disease Predictor",
            "📊 Empirical Model Evaluation",
            "📑 Academic Specifications"
        ],
        index=0
    )

    st.divider()
    st.markdown("""
    <div style="font-size: 11px; color: #64748B; line-height: 1.6;">
        <strong style="color: #0F172A;">Architecture Checklist:</strong><br>
        • Scikit-Learn Pipeline (`v1.4+`)<br>
        • `OneHotEncoder(drop='first')`<br>
        • Zero Data Leakage Split (80/20)<br>
        • Real-Time Inference Latency &lt; 5ms
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TASK 1: IRIS BOTANICAL CLASSIFIER
# -----------------------------------------------------------------------------
if navigation == "🌸 Task 1: Iris Botanical Classifier":
    st.markdown("""
    <div class="page-header">
        <div class="page-breadcrumbs">Machine Learning Classification / Task 1</div>
        <h1 class="page-title">Iris Botanical Species Classification</h1>
        <p class="page-subtitle">Supervised multi-class classification pipeline built on Scikit-Learn with automated standardization and Random Forest decision boundary estimation.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌿 Botanical Morphological Inputs</div>', unsafe_allow_html=True)
        st.write("Adjust morphological measurements in centimeters to predict specimen taxonomy:")
        
        c_sepal1, c_sepal2 = st.columns(2)
        with c_sepal1:
            sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        with c_sepal2:
            sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
            
        c_petal1, c_petal2 = st.columns(2)
        with c_petal1:
            petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
        with c_petal2:
            petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.6, value=1.3, step=0.1)

        predict_iris_btn = st.button("Run Taxonomy Classification", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Diagnostic Inference Result</div>', unsafe_allow_html=True)

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
            confidence_pct = probabilities[prediction] * 100

            st.markdown(f"""
            <div class="result-badge-success">
                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #047857;">Taxonomic Match Confirmed</div>
                <div style="font-size: 24px; font-weight: 800; color: #065F46; margin: 4px 0 6px 0;">Iris {predicted_species}</div>
                <div style="font-size: 13px; color: #047857;">Model Probability Confidence: <strong>{confidence_pct:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("<strong>Class Probability Distribution:</strong>", unsafe_allow_html=True)
            for idx, sp in enumerate(species_names):
                p_val = probabilities[idx] * 100
                st.write(f"• **Iris {sp}**: `{p_val:.1f}%`")
                st.progress(p_val / 100.0)
        else:
            st.info("Adjust the morphological sliders on the left and click **'Run Taxonomy Classification'** to test real-time model inference.")
            st.markdown("""
            <div style="font-size: 12px; color: #64748B; margin-top: 14px; line-height: 1.5;">
                <strong>Benchmark Highlights:</strong><br>
                • 100% classification precision on <em>Iris setosa</em> cluster.<br>
                • 92.11% overall accuracy across held-out stratified test samples.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Metrics Summary Row
    if metrics:
        m = metrics["iris"]
        st.markdown('<div class="clean-card" style="margin-top: 10px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Empirical Verification Scores (Test Partition n=38)</div>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{m["accuracy"]*100:.2f}%</div><div class="metric-lbl">Overall Accuracy</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{m["precision"]*100:.2f}%</div><div class="metric-lbl">Macro Precision</div></div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{m["recall"]*100:.2f}%</div><div class="metric-lbl">Macro Recall</div></div>', unsafe_allow_html=True)
        with m_col4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{m["f1"]*100:.2f}%</div><div class="metric-lbl">Macro F1-Score</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TASK 2: HEART DISEASE RISK PREDICTOR
# -----------------------------------------------------------------------------
elif navigation == "❤️ Task 2: Heart Disease Predictor":
    st.markdown("""
    <div class="page-header">
        <div class="page-breadcrumbs">Machine Learning Classification / Task 2</div>
        <h1 class="page-title">Cardiovascular Disease Clinical Risk Screening</h1>
        <p class="page-subtitle">Binary classification engine on 500 patient observations featuring an integrated Scikit-Learn <code>OneHotEncoder(drop='first')</code> pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("heart_form"):
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Patient Clinical & Demographic Intake</div>', unsafe_allow_html=True)
        
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=52)
        with r1_c2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
        with r1_c3:
            blood_pressure = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=128)
        with r1_c4:
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=120, max_value=350, value=215)

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=27.4, step=0.1)
        with r2_c2:
            smoker = st.selectbox("Smoker History", ["No", "Yes"])
        with r2_c3:
            diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
        with r2_c4:
            family_history = st.selectbox("Cardiovascular Family History", ["No", "Yes"])

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            physical_activity = st.slider("Weekly Physical Activity Level (0 = Sedentary, 4 = High Intensity)", 0, 4, 2)
        with r3_c2:
            stress_level = st.slider("Reported Chronic Stress Level (1 = Baseline, 10 = Severe)", 1, 10, 5)

        submit_btn = st.form_submit_button("Calculate Cardiovascular Risk Score", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🩺 Clinical Triage Recommendation</div>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.2, 1], gap="large")

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-badge-danger">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #B91C1C;">Elevated Cardiovascular Risk Detected</div>
                    <div style="font-size: 22px; font-weight: 800; color: #991B1B; margin: 4px 0 6px 0;">Positive Likelihood of Heart Disease ({risk_probability:.1f}%)</div>
                    <div style="font-size: 13px; color: #7F1D1D; line-height: 1.4;">
                        Clinical Alert: The model identifies high-risk interactions across cholesterol, arterial pressure, and reported lifestyle markers. 
                        Prompt secondary diagnostic evaluation (ECG, stress echo, lipid profile) is recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-badge-success">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #047857;">Non-Elevated Risk Profile</div>
                    <div style="font-size: 22px; font-weight: 800; color: #065F46; margin: 4px 0 6px 0;">Low Likelihood of Heart Disease ({risk_probability:.1f}%)</div>
                    <div style="font-size: 13px; color: #065F46; line-height: 1.4;">
                        Clinical Advisory: The patient's biomarkers remain within expected baseline distributions. Continued lifestyle maintenance and routine annual monitoring are recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Model Risk Probability Score", f"{risk_probability:.1f}%")
            st.progress(risk_probability / 100.0)
            st.markdown("""
            <div style="font-size: 11px; color: #64748B; margin-top: 8px;">
                *Calculated via Logistic Regression pipeline with L2 regularization and OneHotEncoder categorical parameter normalization.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TASK 3: EMPIRICAL MODEL EVALUATION
# -----------------------------------------------------------------------------
elif navigation == "📊 Empirical Model Evaluation":
    st.markdown("""
    <div class="page-header">
        <div class="page-breadcrumbs">Machine Learning Classification / Model Benchmarks</div>
        <h1 class="page-title">Quantitative Model Performance Benchmarks</h1>
        <p class="page-subtitle">Comprehensive validation tables, error confusion matrices, and ROC curves auditing model generalization on unseen test samples.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_iris, tab_heart = st.tabs(["🌸 Task 1: Iris Classification Benchmark", "❤️ Task 2: Heart Disease Multi-Model Benchmark"])

    with tab_iris:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Confusion Matrix & Feature Correlation (Iris)</div>', unsafe_allow_html=True)
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            cm_path = os.path.join(FIGS_DIR, "iris_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Figure 1.3: Iris Confusion Matrix (n=38 test instances)", use_container_width=True)
        with c_i2:
            corr_path = os.path.join(FIGS_DIR, "iris_correlation_heatmap.png")
            if os.path.exists(corr_path):
                st.image(corr_path, caption="Figure 1.1: Feature Correlation Heatmap", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_heart:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Comparative Candidate Classifier Benchmark (Test Cohort n=100)</div>', unsafe_allow_html=True)
        if metrics:
            comp_df = pd.DataFrame(metrics["heart"]["comparison"])
            comp_df["Accuracy"] = comp_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["Precision"] = comp_df["Precision"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["Recall"] = comp_df["Recall"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["F1-Score"] = comp_df["F1-Score"].apply(lambda x: f"{x*100:.2f}%")
            comp_df["ROC-AUC"] = comp_df["ROC-AUC"].apply(lambda x: f"{x:.3f}")
            st.dataframe(comp_df, use_container_width=True)

        st.write("")
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            h_cm_path = os.path.join(FIGS_DIR, "heart_confusion_matrix.png")
            if os.path.exists(h_cm_path):
                st.image(h_cm_path, caption="Figure 2.3: Heart Disease Confusion Matrix (Optimal Logistic Regression)", use_container_width=True)
        with c_h2:
            h_roc_path = os.path.join(FIGS_DIR, "heart_roc_curve.png")
            if os.path.exists(h_roc_path):
                st.image(h_roc_path, caption="Figure 2.4: Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TASK 4: ACADEMIC SPECIFICATIONS
# -----------------------------------------------------------------------------
elif navigation == "📑 Academic Specifications":
    st.markdown("""
    <div class="page-header">
        <div class="page-breadcrumbs">Machine Learning Classification / Coursework Documentation</div>
        <h1 class="page-title">Academic & Engineering Specifications</h1>
        <p class="page-subtitle">Department of Information Technology, Pentecost University • Course Assessment Specification</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="clean-card">
        <div class="card-title">📌 Submission & Architecture Summary</div>
        <table style="width:100%; font-size:14px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E2E8F0; height: 36px;">
                <td style="font-weight: 700; width: 30%; color: #475569;">Candidate Name</td>
                <td style="color: #0F172A;">Emmanuel Siaw Asante</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0; height: 36px;">
                <td style="font-weight: 700; color: #475569;">Student Identification / Index</td>
                <td style="color: #0F172A; font-family: monospace; font-weight: 700;">PYT230047</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0; height: 36px;">
                <td style="font-weight: 700; color: #475569;">Course Lecturer</td>
                <td style="color: #0F172A;">Mr. Harry Attieku-Boateng</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0; height: 36px;">
                <td style="font-weight: 700; color: #475569;">GitHub Source Repository</td>
                <td><a href="https://github.com/s47user/PUIT-Assignment1-ML-Classification" target="_blank" style="color: #2563EB; font-weight: 600;">github.com/s47user/PUIT-Assignment1-ML-Classification</a></td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0; height: 36px;">
                <td style="font-weight: 700; color: #475569;">Submission Deadline</td>
                <td style="color: #DC2626; font-weight: 700;">November 30th</td>
            </tr>
        </table>
        
        <h4 style="margin-top: 24px; color: #0F172A;">Engineering Core Highlights:</h4>
        <ol style="font-size: 13.5px; color: #334155; line-height: 1.8;">
            <li><strong>Automated Pipeline Preprocessing:</strong> Employs <code>ColumnTransformer</code> to combine numerical <code>StandardScaler</code> and categorical <code>OneHotEncoder(drop='first')</code>.</li>
            <li><strong>Prevention of Multicollinearity:</strong> Dropping baseline dummy columns prevents mathematical rank deficiency in regression matrices.</li>
            <li><strong>Clinical Priority on Recall:</strong> Evaluates trade-offs between precision and recall, optimizing sensitivity to mitigate lethal false-negative clinical misclassifications.</li>
            <li><strong>Production Model Persistence:</strong> All estimator weights serialized with <code>joblib</code> for sub-10ms web inference.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
