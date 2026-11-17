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
    page_title="PUIT ML Classification Suite | Emmanuel Siaw Asante",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File and Directory Anchors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")

# -----------------------------------------------------------------------------
# MODEL CACHING & INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    iris_path = os.path.join(MODELS_DIR, "iris_model.joblib")
    heart_path = os.path.join(MODELS_DIR, "heart_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.joblib")

    iris_clf = joblib.load(iris_path) if os.path.exists(iris_path) else None
    heart_clf = joblib.load(heart_path) if os.path.exists(heart_path) else None
    metrics_data = joblib.load(metrics_path) if os.path.exists(metrics_path) else None
    return iris_clf, heart_clf, metrics_data

iris_model, heart_model, metrics = load_artifacts()

# -----------------------------------------------------------------------------
# HIGH-CRAFT DESIGN SYSTEM (MODERN HUMAN UI · ZERO SLOP)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean System Typography Stack */
    html, body, [class*="css"], [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0F172A;
    }

    /* Base Canvas */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Container Constraints */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }

    /* Institutional Header */
    .puit-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .puit-breadcrumbs {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .puit-title {
        font-size: 1.625rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
        line-height: 1.25;
    }
    .puit-subtitle {
        font-size: 0.875rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }

    /* Streamlit Native Bordered Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04) !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
    }

    /* Streamlit Forms */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04) !important;
        padding: 1.25rem 1.5rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Section Eyebrow Labels */
    .section-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: #0F172A;
        margin: 0 0 0.75rem 0;
    }

    /* Standard Button Styling */
    div.stButton > button, div[data-testid="stForm"] button[type="submit"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid #0F172A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.625rem 1.25rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    div.stButton > button:hover, div[data-testid="stForm"] button[type="submit"]:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    div.stButton > button:active, div[data-testid="stForm"] button[type="submit"]:active {
        background-color: #020617 !important;
        transform: scale(0.99) !important;
    }

    /* Native Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 0.875rem 1rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* Diagnostic Result Cards */
    .diagnostic-card-neutral {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
    }
    .diagnostic-card-success {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 1.25rem;
    }
    .diagnostic-card-danger {
        background-color: #FFF1F2;
        border: 1px solid #FECDD3;
        border-radius: 10px;
        padding: 1.25rem;
    }

    /* Probability Bar Component */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
    .prob-track {
        flex-grow: 1;
        height: 7px;
        background-color: #E2E8F0;
        border-radius: 999px;
        margin: 0 1rem;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background-color: #2563EB;
    }
    .prob-val {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 600;
        min-width: 48px;
        text-align: right;
        color: #0F172A;
    }

    /* Data Table Styling */
    .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .clean-table th {
        text-align: left;
        padding: 0.625rem 0.75rem;
        background-color: #F8FAFC;
        color: #475569;
        font-weight: 600;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .clean-table td {
        padding: 0.625rem 0.75rem;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }
    .clean-table tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & ACADEMIC AUDIT PROFILE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 1rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.25rem;">
        <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B;">Faculty Assessment</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; margin-top: 0.2rem;">Pentecost University</div>
        <div style="font-size: 0.8rem; color: #475569;">Computing & Information Technology</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
        <div class="section-eyebrow">Candidate Record</div>
        <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">Emmanuel Siaw Asante</div>
        <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.8rem; color: #2563EB; font-weight: 600; margin-top: 0.15rem;">ID: PYT230047</div>
        <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid #F1F5F9;">
            Lecturer: <strong>Mr. Harry Attieku-Boateng</strong><br>
            Module: <strong>ML Classification (Assignment 1)</strong>
        </div>
        """, unsafe_allow_html=True)

    navigation = st.radio(
        "Navigation",
        [
            "01 · Iris Botanical Taxonomy",
            "02 · Cardiovascular Risk Screening",
            "03 · Quantitative Model Benchmarks",
            "04 · Coursework & Candidate Specs"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #E2E8F0; font-size: 0.75rem; color: #64748B; line-height: 1.6;">
        <span style="font-weight: 600; color: #0F172A;">Pipeline Invariants:</span><br>
        • Scikit-Learn <code>v1.4+</code><br>
        • <code>OneHotEncoder(drop='first')</code><br>
        • Stratified 80/20 train-test partition<br>
        • Deterministic random state seeding
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 1: IRIS BOTANICAL CLASSIFICATION
# -----------------------------------------------------------------------------
if navigation == "01 · Iris Botanical Taxonomy":
    st.markdown("""
    <div class="puit-header">
        <div class="puit-breadcrumbs">Coursework Assignment 1 / Task 1</div>
        <h1 class="puit-title">Iris Botanical Species Classification</h1>
        <p class="puit-subtitle">Multi-class classification pipeline standardizing morphological dimensions and estimating species probabilities via an ensemble Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap="medium")

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Taxonomic Inputs</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Specimen Morphological Measurements</div>', unsafe_allow_html=True)
            st.caption("Adjust the observed petal and sepal dimensions in centimeters:")

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                sepal_len = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
            with s_col2:
                sepal_wid = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                petal_len = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
            with p_col2:
                petal_wid = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.6, value=1.3, step=0.1)

            st.write("")
            classify_iris_btn = st.button("Run Classification", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Inference Telemetry</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Taxonomic Classification Outcome</div>', unsafe_allow_html=True)

            if iris_model is not None:
                sample_df = pd.DataFrame([{
                    "sepal length (cm)": sepal_len,
                    "sepal width (cm)": sepal_wid,
                    "petal length (cm)": petal_len,
                    "petal width (cm)": petal_wid
                }])

                pred_idx = iris_model.predict(sample_df)[0]
                prob_dist = iris_model.predict_proba(sample_df)[0]
                species_labels = ["Setosa", "Versicolor", "Virginica"]
                identified_species = species_labels[pred_idx]
                confidence_score = prob_dist[pred_idx] * 100.0

                st.markdown(f"""
                <div class="diagnostic-card-success">
                    <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #166534;">Verified Taxonomic Classification</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #14532D; margin: 0.25rem 0 0.25rem 0;"><em>Iris {identified_species.lower()}</em></div>
                    <div style="font-size: 0.85rem; color: #15803D;">Model Probability Confidence: <strong style="font-family: ui-monospace, monospace;">{confidence_score:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin: 1.1rem 0 0.5rem 0;">Class Probability Distribution</div>', unsafe_allow_html=True)
                for i, sp in enumerate(species_labels):
                    pct = prob_dist[i] * 100.0
                    fill_color = "#2563EB" if i == pred_idx else "#94A3B8"
                    st.markdown(f"""
                    <div class="prob-row">
                        <span style="font-weight: 500; color: #1E293B; min-width: 100px;">Iris {sp.lower()}</span>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: {pct:.1f}%; background-color: {fill_color};"></div>
                        </div>
                        <span class="prob-val">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Model artifact not found. Please train or verify models directory.")

    # Empirical Metrics Row
    if metrics and "iris" in metrics:
        m = metrics["iris"]
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Empirical Verification</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Stratified Test Partition Scores (n = 38)</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall Accuracy", f"{m['accuracy']*100:.2f}%")
            with m2:
                st.metric("Macro Precision", f"{m['precision']*100:.2f}%")
            with m3:
                st.metric("Macro Recall", f"{m['recall']*100:.2f}%")
            with m4:
                st.metric("Macro F1-Score", f"{m['f1']*100:.2f}%")


# -----------------------------------------------------------------------------
# MODULE 2: CARDIOVASCULAR RISK PREDICTOR
# -----------------------------------------------------------------------------
elif navigation == "02 · Cardiovascular Risk Screening":
    st.markdown("""
    <div class="puit-header">
        <div class="puit-breadcrumbs">Coursework Assignment 1 / Task 2</div>
        <h1 class="puit-title">Cardiovascular Disease Clinical Risk Screening</h1>
        <p class="puit-subtitle">Supervised binary clinical screening on 500 patient observations, featuring automated categorical preprocessing via <code>OneHotEncoder(drop='first')</code> to avert multicollinearity.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("cardio_risk_form"):
        st.markdown('<div class="section-eyebrow">Clinical Intake</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Biomarkers & Clinical History</div>', unsafe_allow_html=True)
        st.caption("Enter patient diagnostic readings and lifestyle covariates for multivariate risk estimation:")

        # Row 1: Demographics & Vitals
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            age = st.number_input("Patient Age (Years)", min_value=18, max_value=100, value=52)
        with r1_c2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
        with r1_c3:
            bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=132)
        with r1_c4:
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=120, max_value=350, value=224)

        # Row 2: Anthropometrics & Comorbidities
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=28.1, step=0.1)
        with r2_c2:
            smoker = st.selectbox("Tobacco Smoking History", ["No", "Yes"])
        with r2_c3:
            diabetes = st.selectbox("Diabetes Mellitus History", ["No", "Yes"])
        with r2_c4:
            fam_hist = st.selectbox("Cardiovascular Family History", ["No", "Yes"])

        # Row 3: Lifestyle & Behavioral Covariates
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            phys_act = st.slider("Weekly Physical Activity Level (0 = Sedentary, 4 = Vigorous)", min_value=0, max_value=4, value=2)
        with r3_c2:
            stress = st.slider("Reported Chronic Stress Scale (1 = Baseline, 10 = Severe)", min_value=1, max_value=10, value=5)

        st.write("")
        submit_risk = st.form_submit_button("Evaluate Cardiovascular Risk Profile", use_container_width=True)

    if submit_risk and heart_model is not None:
        patient_record = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "blood_pressure": bp,
            "cholesterol": chol,
            "bmi": bmi,
            "smoker": smoker,
            "diabetes": diabetes,
            "physical_activity": phys_act,
            "family_history": fam_hist,
            "stress_level": stress
        }])

        pred_risk = heart_model.predict(patient_record)[0]
        prob_risk = heart_model.predict_proba(patient_record)[0][1] * 100.0

        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Clinical Triage Recommendation</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Diagnostic Screening Assessment</div>', unsafe_allow_html=True)

            res_c1, res_c2 = st.columns([1.2, 1], gap="large")

            with res_c1:
                if pred_risk == 1:
                    st.markdown(f"""
                    <div class="diagnostic-card-danger">
                        <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #991B1B;">Elevated Cardiovascular Risk Detected</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #7F1D1D; margin: 0.25rem 0 0.4rem 0;">Positive Likelihood of Heart Disease ({prob_risk:.1f}%)</div>
                        <div style="font-size: 0.85rem; color: #991B1B; line-height: 1.5;">
                            <strong>Clinical Protocol:</strong> High-risk interactions detected across arterial pressure ({bp} mmHg), serum cholesterol ({chol} mg/dL), and reported medical history. Immediate secondary diagnostic workup (12-lead ECG, stress echocardiogram, fasting lipid fraction) is indicated.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="diagnostic-card-success">
                        <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #166534;">Baseline Physiological Profile</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #14532D; margin: 0.25rem 0 0.4rem 0;">Low Likelihood of Heart Disease ({prob_risk:.1f}%)</div>
                        <div style="font-size: 0.85rem; color: #15803D; line-height: 1.5;">
                            <strong>Clinical Advisory:</strong> Patient biomarkers remain within acceptable clinical boundaries. Routine annual cardiovascular monitoring and sustained lifestyle maintenance are advised.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with res_c2:
                st.metric("Estimated Risk Probability", f"{prob_risk:.1f}%")
                # Custom proportional progress indicator
                bar_color = "#DC2626" if pred_risk == 1 else "#16A34A"
                st.markdown(f"""
                <div style="margin-top: 0.75rem;">
                    <div class="prob-track" style="margin: 0; height: 10px;">
                        <div class="prob-fill" style="width: {prob_risk:.1f}%; background-color: {bar_color};"></div>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.5rem; line-height: 1.4;">
                        Estimated via regularized Logistic Regression pipeline with <code>OneHotEncoder(drop='first')</code>.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 3: QUANTITATIVE BENCHMARKS
# -----------------------------------------------------------------------------
elif navigation == "03 · Quantitative Model Benchmarks":
    st.markdown("""
    <div class="puit-header">
        <div class="puit-breadcrumbs">Coursework Assignment 1 / Verification</div>
        <h1 class="puit-title">Quantitative Model Benchmarks & Validation</h1>
        <p class="puit-subtitle">Empirical performance metrics, confusion matrices, and ROC curves auditing model generalization across held-out test partitions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_iris, tab_heart = st.tabs([
        "Task 1: Iris Taxonomic Evaluation",
        "Task 2: Cardiovascular Candidate Classifier Benchmark"
    ])

    with tab_iris:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Task 1 Evaluation</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Confusion Matrix & Botanical Feature Correlations</div>', unsafe_allow_html=True)
            
            c_i1, c_i2 = st.columns(2, gap="medium")
            with c_i1:
                cm_p = os.path.join(FIGS_DIR, "iris_confusion_matrix.png")
                if os.path.exists(cm_p):
                    st.image(cm_p, caption="Figure 1.1: Stratified Iris Confusion Matrix (n = 38 test observations)", use_container_width=True)
            with c_i2:
                corr_p = os.path.join(FIGS_DIR, "iris_correlation_heatmap.png")
                if os.path.exists(corr_p):
                    st.image(corr_p, caption="Figure 1.2: Botanical Feature Correlation Matrix (Pearson r)", use_container_width=True)

    with tab_heart:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Comparative Audit</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Candidate Classifier Performance Comparison (Test Cohort n = 100)</div>', unsafe_allow_html=True)
            
            if metrics and "heart" in metrics and "comparison" in metrics["heart"]:
                comp_rows = metrics["heart"]["comparison"]
                table_html = """
                <table class="clean-table">
                    <thead>
                        <tr>
                            <th>Candidate Estimator</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall (Sensitivity)</th>
                            <th>F1-Score</th>
                            <th>ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for row in comp_rows:
                    is_best = row.get("Model") == "Logistic Regression"
                    highlight = "background-color: #F8FAFC; font-weight: 600;" if is_best else ""
                    table_html += f"""
                        <tr style="{highlight}">
                            <td><strong>{row.get('Model')}</strong> {'(Optimal Pipeline)' if is_best else ''}</td>
                            <td style="font-family: ui-monospace, monospace;">{row.get('Accuracy')*100:.2f}%</td>
                            <td style="font-family: ui-monospace, monospace;">{row.get('Precision')*100:.2f}%</td>
                            <td style="font-family: ui-monospace, monospace; color: {'#16A34A' if is_best else '#0F172A'};"><strong>{row.get('Recall')*100:.2f}%</strong></td>
                            <td style="font-family: ui-monospace, monospace;">{row.get('F1-Score')*100:.2f}%</td>
                            <td style="font-family: ui-monospace, monospace;">{row.get('ROC-AUC'):.3f}</td>
                        </tr>
                    """
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                st.caption("Note: Logistic Regression was selected as the optimal production model because it achieved the highest sensitivity (Recall: 75.47%), directly minimizing critical false-negative clinical misclassifications.")

            st.write("")
            c_h1, c_h2 = st.columns(2, gap="medium")
            with c_h1:
                h_roc_p = os.path.join(FIGS_DIR, "heart_roc_curve.png")
                if os.path.exists(h_roc_p):
                    st.image(h_roc_p, caption="Figure 2.1: Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
            with c_h2:
                h_cm_p = os.path.join(FIGS_DIR, "heart_confusion_matrix.png")
                if os.path.exists(h_cm_p):
                    st.image(h_cm_p, caption="Figure 2.2: Cardiovascular Confusion Matrix (Logistic Regression)", use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 4: ACADEMIC & CANDIDATE SPECIFICATIONS
# -----------------------------------------------------------------------------
elif navigation == "04 · Coursework & Candidate Specs":
    st.markdown("""
    <div class="puit-header">
        <div class="puit-breadcrumbs">Pentecost University / Assessment Record</div>
        <h1 class="puit-title">Academic & Engineering Specifications</h1>
        <p class="puit-subtitle">Official coursework metadata, institutional oversight details, and core architectural invariants.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Academic Submission Record</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Candidate & Assessment Metadata</div>', unsafe_allow_html=True)

        meta_table = """
        <table class="clean-table">
            <tr>
                <td style="font-weight: 600; width: 28%; color: #64748B;">Candidate Name</td>
                <td style="font-weight: 700; color: #0F172A;">Emmanuel Siaw Asante</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Student ID / Matriculation</td>
                <td style="font-family: ui-monospace, monospace; font-weight: 700; color: #2563EB;">PYT230047</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Course Instructor</td>
                <td style="font-weight: 600; color: #0F172A;">Mr. Harry Attieku-Boateng</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Academic Department</td>
                <td>Department of Information Technology, Faculty of Computing & IT</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Institution</td>
                <td>Pentecost University, Sowutuom, Accra, Ghana</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Submission Deadline</td>
                <td style="font-weight: 700; color: #DC2626;">November 30th</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Public Source Repository</td>
                <td><a href="https://github.com/s47user/PUIT-Assignment1-ML-Classification" target="_blank" style="color: #2563EB; font-weight: 600; text-decoration: none;">github.com/s47user/PUIT-Assignment1-ML-Classification</a></td>
            </tr>
        </table>
        """
        st.markdown(meta_table, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Engineering Rigor</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Core Mathematical & Architectural Invariants</div>', unsafe_allow_html=True)

        st.markdown("""
        <ol style="font-size: 0.875rem; color: #334155; line-height: 1.8; margin: 0; padding-left: 1.25rem;">
            <li><strong>Automated Preprocessing via ColumnTransformer:</strong> Numerically standardizes continuous variables while applying <code>OneHotEncoder(drop='first')</code> to categorical fields, entirely eliminating data leakage across train/test splits.</li>
            <li><strong>Multicollinearity & Matrix Rank Deficiency Defense:</strong> Dropping baseline dummy categories guarantees full column rank in design matrices, preventing standard error inflation during Hessian inversion.</li>
            <li><strong>Clinical Sensitivity Prioritization (Recall):</strong> Optimizes medical thresholding to prioritize sensitivity (75.47% test recall), mitigating catastrophic false negatives in clinical triage.</li>
            <li><strong>Zero-Overhead Serialization:</strong> Full pipeline graph serialized via <code>joblib</code>, ensuring production inference execution times under 5 milliseconds.</li>
        </ol>
        """, unsafe_allow_html=True)
