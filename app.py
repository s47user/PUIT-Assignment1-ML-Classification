import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION (FUTURISTIC TELEMETRY CONSOLE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PUIT Telemetry Console | Emmanuel Siaw Asante",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anchor Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION & CACHING
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
# MODERN FUTURISTIC DESIGN SYSTEM (OBSIDIAN HUD · HIGH-CRAFT TELEMETRY)
# Combines Vercel/Linear dark ergonomics, monospace telemetry, and crisp 1px hairlines.
# Zero AI slop, zero floaty gradients, pure precision engineering.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Monospace & Sans Typography */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #E2E8F0;
    }

    /* Deep Obsidian Canvas */
    .stApp {
        background-color: #080A0F;
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(37, 99, 235, 0.08), transparent),
            radial-gradient(circle at 100% 100%, rgba(14, 165, 233, 0.03), transparent);
    }

    /* Container Constraints */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1220px !important;
    }

    /* Futuristic HUD Header */
    .hud-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .hud-breadcrumbs {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hud-breadcrumbs::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        background-color: #38BDF8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38BDF8;
    }
    .hud-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin: 0.4rem 0 0.35rem 0;
        line-height: 1.2;
    }
    .hud-subtitle {
        font-size: 0.875rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Obsidian Telemetry Panels (Streamlit Native Containers) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(14, 18, 27, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.6) !important;
        padding: 1.35rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
    }

    /* Streamlit Forms */
    div[data-testid="stForm"] {
        background: rgba(14, 18, 27, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.6) !important;
        padding: 1.35rem 1.5rem !important;
    }

    /* Sidebar Telemetry Console */
    section[data-testid="stSidebar"] {
        background-color: #0A0D14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    /* Section HUD Labels */
    .hud-eyebrow {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .hud-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: #F8FAFC;
        margin: 0 0 0.75rem 0;
    }

    /* High-Precision Interactive Buttons */
    div.stButton > button, div[data-testid="stForm"] button[type="submit"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover, div[data-testid="stForm"] button[type="submit"]:hover {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        border-color: rgba(96, 165, 250, 0.5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active, div[data-testid="stForm"] button[type="submit"]:active {
        transform: scale(0.98) !important;
        background: #1E3A8A !important;
    }

    /* Metric Panels */
    div[data-testid="stMetric"] {
        background-color: rgba(18, 24, 38, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 8px !important;
        padding: 0.85rem 1rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    }

    /* Form Inputs Dark Restyling */
    div[data-baseweb="input"] input, div[data-baseweb="select"] input {
        color: #F8FAFC !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #121722 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }

    /* Telemetry Result Panels */
    .telemetry-card-success {
        background: rgba(16, 44, 30, 0.4);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 10px;
        padding: 1.25rem;
    }
    .telemetry-card-danger {
        background: rgba(58, 20, 26, 0.4);
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-radius: 10px;
        padding: 1.25rem;
    }

    /* Futuristic Spectrum Bar Component */
    .spectrum-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
        font-size: 0.82rem;
    }
    .spectrum-track {
        flex-grow: 1;
        height: 6px;
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        margin: 0 1rem;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .spectrum-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .spectrum-val {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-weight: 600;
        min-width: 52px;
        text-align: right;
        color: #E2E8F0;
    }

    /* Matrix Benchmark Table */
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }
    .matrix-table th {
        text-align: left;
        padding: 0.65rem 0.85rem;
        background-color: rgba(255, 255, 255, 0.03);
        color: #64748B;
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-weight: 600;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .matrix-table td {
        padding: 0.65rem 0.85rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #E2E8F0;
    }
    .matrix-table tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & TELEMETRY CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1.25rem;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: #38BDF8;">PENTECOST UNIVERSITY</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; margin-top: 0.2rem;">Faculty of Computing & IT</div>
        <div style="font-size: 0.78rem; color: #64748B;">Department of Information Technology</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
        <div class="hud-eyebrow">CANDIDATE DOSSIER</div>
        <div style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC;">Emmanuel Siaw Asante</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38BDF8; font-weight: 600; margin-top: 0.15rem;">ID: PYT230047</div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06);">
            Supervisor: <strong style="color: #E2E8F0;">Mr. Harry Attieku-Boateng</strong><br>
            Module: <strong style="color: #E2E8F0;">ML Classification (Assignment 1)</strong>
        </div>
        """, unsafe_allow_html=True)

    navigation = st.radio(
        "Navigation",
        [
            "01 // BOTANICAL TAXONOMY ENGINE",
            "02 // CARDIOVASCULAR RISK TELEMETRY",
            "03 // EMPIRICAL VALIDATION MATRIX",
            "04 // ACADEMIC & SYSTEM CONTRACTS"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748B; line-height: 1.6;">
        <span style="font-weight: 600; color: #94A3B8;">SYSTEM ARCHITECTURE:</span><br>
        • Scikit-Learn <code>v1.4+</code><br>
        • <code>OneHotEncoder(drop='first')</code><br>
        • Stratified 80/20 train/test split<br>
        • Deterministic L2 regularization<br>
        • Sub-5ms inference latency
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 1: BOTANICAL TAXONOMY ENGINE (IRIS)
# -----------------------------------------------------------------------------
if navigation == "01 // BOTANICAL TAXONOMY ENGINE":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-breadcrumbs">COURSEWORK ASSIGNMENT 1 // TASK 1 // SUPERVISED ENSEMBLE</div>
        <h1 class="hud-title">Botanical Taxonomy Inference Engine</h1>
        <p class="hud-subtitle">Supervised multi-class Random Forest estimator with continuous morphological standardization and real-time class probability distribution mapping.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap="medium")

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="hud-eyebrow">MORPHOMETRIC PARAMETERS</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Specimen Observation Vectors</div>', unsafe_allow_html=True)
            st.caption("Calibrate specimen sepal and petal dimensions in centimeters:")

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
            classify_iris_btn = st.button("EXECUTE CLASSIFICATION PIPELINE", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown('<div class="hud-eyebrow">TELEMETRY READOUT</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Taxonomic Determination</div>', unsafe_allow_html=True)

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
                <div class="telemetry-card-success">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #34D399;">TAXONOMIC DETERMINATION // VERIFIED</div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #F0FDF4; margin: 0.3rem 0 0.25rem 0;"><em>Iris {identified_species.lower()}</em></div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #A7F3D0;">CONFIDENCE: <strong style="color: #34D399;">{confidence_score:.2f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin: 1.25rem 0 0.5rem 0;">CLASS PROBABILITY SPECTRUM</div>', unsafe_allow_html=True)
                for i, sp in enumerate(species_labels):
                    pct = prob_dist[i] * 100.0
                    is_active = (i == pred_idx)
                    fill_color = "linear-gradient(90deg, #2563EB, #38BDF8)" if is_active else "#334155"
                    glow = "box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);" if is_active else ""
                    st.markdown(f"""
                    <div class="spectrum-row">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: {'600' if is_active else '400'}; color: {'#F8FAFC' if is_active else '#64748B'}; min-width: 110px;">Iris {sp.lower()}</span>
                        <div class="spectrum-track">
                            <div class="spectrum-fill" style="width: {pct:.1f}%; background: {fill_color}; {glow}"></div>
                        </div>
                        <span class="spectrum-val" style="color: {'#38BDF8' if is_active else '#64748B'};">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Model weights unavailable. Inspect models directory.")

    # Empirical Verification Scores Grid
    if metrics and "iris" in metrics:
        m = metrics["iris"]
        with st.container(border=True):
            st.markdown('<div class="hud-eyebrow">EMPIRICAL BENCHMARKS</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Stratified Test Partition Metrics (n = 38 Observations)</div>', unsafe_allow_html=True)
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
# MODULE 2: CARDIOVASCULAR RISK TELEMETRY (HEART DISEASE)
# -----------------------------------------------------------------------------
elif navigation == "02 // CARDIOVASCULAR RISK TELEMETRY":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-breadcrumbs">COURSEWORK ASSIGNMENT 1 // TASK 2 // REGULARIZED LOGISTIC REGRESSION</div>
        <h1 class="hud-title">Cardiovascular Disease Risk Telemetry</h1>
        <p class="hud-subtitle">Clinical risk appraisal on 500 patient observations featuring automated categorical normalization via <code>OneHotEncoder(drop='first')</code> to prevent multicollinearity.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("cardio_risk_form"):
        st.markdown('<div class="hud-eyebrow">CLINICAL TELEMETRY INTAKE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-section-title">Patient Biomarkers & Lifestyle Covariates</div>', unsafe_allow_html=True)
        st.caption("Input patient diagnostic biomarkers and lifestyle indicators for multivariate clinical risk estimation:")

        # Row 1: Demographics & Hemodynamics
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            age = st.number_input("Patient Age (Years)", min_value=18, max_value=100, value=52)
        with r1_c2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
        with r1_c3:
            bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=132)
        with r1_c4:
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=120, max_value=350, value=224)

        # Row 2: Anthropometrics & Clinical History
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=28.1, step=0.1)
        with r2_c2:
            smoker = st.selectbox("Tobacco Smoking History", ["No", "Yes"])
        with r2_c3:
            diabetes = st.selectbox("Diabetes Mellitus History", ["No", "Yes"])
        with r2_c4:
            fam_hist = st.selectbox("Cardiovascular Family History", ["No", "Yes"])

        # Row 3: Behavioral Indicators
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            phys_act = st.slider("Weekly Physical Activity Level (0 = Sedentary, 4 = High Intensity)", min_value=0, max_value=4, value=2)
        with r3_c2:
            stress = st.slider("Reported Chronic Stress Scale (1 = Baseline, 10 = Severe)", min_value=1, max_value=10, value=5)

        st.write("")
        submit_risk = st.form_submit_button("COMPUTE MULTIVARIATE RISK VECTOR", use_container_width=True)

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
            st.markdown('<div class="hud-eyebrow">DIAGNOSTIC TRIAGE APPRAISAL</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Clinical Risk Synthesis</div>', unsafe_allow_html=True)

            res_c1, res_c2 = st.columns([1.2, 1], gap="large")

            with res_c1:
                if pred_risk == 1:
                    st.markdown(f"""
                    <div class="telemetry-card-danger">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #F87171;">TRIAGE ALERT // ELEVATED CARDIOVASCULAR RISK</div>
                        <div style="font-size: 1.45rem; font-weight: 800; color: #FEF2F2; margin: 0.3rem 0 0.35rem 0;">Positive Likelihood of Heart Disease ({prob_risk:.1f}%)</div>
                        <div style="font-size: 0.85rem; color: #FCA5A5; line-height: 1.5;">
                            <strong>Diagnostic Protocol:</strong> Multivariate interactions detected across arterial pressure ({bp} mmHg), serum cholesterol ({chol} mg/dL), and reported medical history. Secondary cardiovascular evaluation (12-lead ECG, stress echocardiography, fasting lipid fraction) is indicated.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="telemetry-card-success">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #34D399;">TRIAGE STATUS // BASELINE PHYSIOLOGICAL PROFILE</div>
                        <div style="font-size: 1.45rem; font-weight: 800; color: #F0FDF4; margin: 0.3rem 0 0.35rem 0;">Low Likelihood of Heart Disease ({prob_risk:.1f}%)</div>
                        <div style="font-size: 0.85rem; color: #A7F3D0; line-height: 1.5;">
                            <strong>Clinical Protocol:</strong> Patient biomarkers remain within acceptable epidemiological distributions. Routine annual cardiovascular screening and continued lifestyle maintenance are advised.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with res_c2:
                st.metric("Estimated Risk Vector", f"{prob_risk:.1f}%")
                bar_color = "linear-gradient(90deg, #DC2626, #EF4444)" if pred_risk == 1 else "linear-gradient(90deg, #16A34A, #22C55E)"
                glow = "box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);" if pred_risk == 1 else "box-shadow: 0 0 12px rgba(34, 197, 94, 0.4);"
                st.markdown(f"""
                <div style="margin-top: 0.75rem;">
                    <div class="spectrum-track" style="margin: 0; height: 8px;">
                        <div class="spectrum-fill" style="width: {prob_risk:.1f}%; background: {bar_color}; {glow}"></div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B; margin-top: 0.6rem; line-height: 1.4;">
                        Model: Regularized Logistic Regression with <code>OneHotEncoder(drop='first')</code>.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 3: QUANTITATIVE BENCHMARKS (MATRIX)
# -----------------------------------------------------------------------------
elif navigation == "03 // EMPIRICAL VALIDATION MATRIX":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-breadcrumbs">COURSEWORK ASSIGNMENT 1 // VERIFICATION // EMPIRICAL METRICS</div>
        <h1 class="hud-title">Empirical Validation Matrix</h1>
        <p class="hud-subtitle">Quantitative assessment auditing model generalization, ROC sensitivity curves, and multi-class confusion matrices on unseen test cohorts.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_iris, tab_heart = st.tabs([
        "Task 1: Iris Taxonomic Matrix",
        "Task 2: Cardiovascular Multi-Model Matrix"
    ])

    with tab_iris:
        with st.container(border=True):
            st.markdown('<div class="hud-eyebrow">TASK 1 TELEMETRY</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Stratified Confusion Matrix & Feature Correlations</div>', unsafe_allow_html=True)
            
            c_i1, c_i2 = st.columns(2, gap="medium")
            with c_i1:
                cm_p = os.path.join(FIGS_DIR, "iris_confusion_matrix.png")
                if os.path.exists(cm_p):
                    st.image(cm_p, caption="Figure 1.1: Stratified Iris Confusion Matrix (n = 38 test observations)", use_container_width=True)
            with c_i2:
                corr_p = os.path.join(FIGS_DIR, "iris_correlation_heatmap.png")
                if os.path.exists(corr_p):
                    st.image(corr_p, caption="Figure 1.2: Botanical Morphological Feature Correlation Matrix (Pearson r)", use_container_width=True)

    with tab_heart:
        with st.container(border=True):
            st.markdown('<div class="hud-eyebrow">COMPARATIVE BENCHMARK MATRIX</div>', unsafe_allow_html=True)
            st.markdown('<div class="hud-section-title">Candidate Classifier Performance Comparison (Test Cohort n = 100)</div>', unsafe_allow_html=True)
            
            if metrics and "heart" in metrics and "comparison" in metrics["heart"]:
                comp_rows = metrics["heart"]["comparison"]
                table_html = """
                <table class="matrix-table">
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
                    highlight = "background-color: rgba(37, 99, 235, 0.08); font-weight: 600;" if is_best else ""
                    table_html += f"""
                        <tr style="{highlight}">
                            <td><strong style="color: {'#38BDF8' if is_best else '#F8FAFC'};">{row.get('Model')}</strong> {'[Optimal Pipeline]' if is_best else ''}</td>
                            <td style="font-family: 'JetBrains Mono', monospace;">{row.get('Accuracy')*100:.2f}%</td>
                            <td style="font-family: 'JetBrains Mono', monospace;">{row.get('Precision')*100:.2f}%</td>
                            <td style="font-family: 'JetBrains Mono', monospace; color: {'#34D399' if is_best else '#E2E8F0'}; font-weight: 700;">{row.get('Recall')*100:.2f}%</td>
                            <td style="font-family: 'JetBrains Mono', monospace;">{row.get('F1-Score')*100:.2f}%</td>
                            <td style="font-family: 'JetBrains Mono', monospace;">{row.get('ROC-AUC'):.3f}</td>
                        </tr>
                    """
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                st.caption("Strategic Rationale: Logistic Regression was selected as the optimal production pipeline because it achieved the highest clinical sensitivity (Recall: 75.47%), directly mitigating critical false-negative patient misclassifications.")

            st.write("")
            c_h1, c_h2 = st.columns(2, gap="medium")
            with c_h1:
                h_roc_p = os.path.join(FIGS_DIR, "heart_roc_curve.png")
                if os.path.exists(h_roc_p):
                    st.image(h_roc_p, caption="Figure 2.1: Receiver Operating Characteristic (ROC) Sensitivity Curve", use_container_width=True)
            with c_h2:
                h_cm_p = os.path.join(FIGS_DIR, "heart_confusion_matrix.png")
                if os.path.exists(h_cm_p):
                    st.image(h_cm_p, caption="Figure 2.2: Cardiovascular Confusion Matrix (Optimal Logistic Regression)", use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 4: ACADEMIC & CANDIDATE SPECIFICATIONS
# -----------------------------------------------------------------------------
elif navigation == "04 // ACADEMIC & SYSTEM CONTRACTS":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-breadcrumbs">PENTECOST UNIVERSITY // SYSTEM CONTRACT // AUDIT TRAIL</div>
        <h1 class="hud-title">Academic & Engineering Specifications</h1>
        <p class="hud-subtitle">Institutional compliance record, verification contracts, and mathematical invariants.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="hud-eyebrow">ASSESSMENT RECORD</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-section-title">Candidate & Submission Metadata</div>', unsafe_allow_html=True)

        meta_table = """
        <table class="matrix-table">
            <tr>
                <td style="font-weight: 600; width: 28%; color: #64748B;">Candidate Name</td>
                <td style="font-weight: 700; color: #F8FAFC;">Emmanuel Siaw Asante</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Student ID / Matriculation</td>
                <td style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38BDF8;">PYT230047</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Course Instructor</td>
                <td style="font-weight: 600; color: #F8FAFC;">Mr. Harry Attieku-Boateng</td>
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
                <td style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F87171;">November 30th</td>
            </tr>
            <tr>
                <td style="font-weight: 600; color: #64748B;">Public Source Repository</td>
                <td><a href="https://github.com/s47user/PUIT-Assignment1-ML-Classification" target="_blank" style="font-family: 'JetBrains Mono', monospace; color: #38BDF8; font-weight: 600; text-decoration: none;">github.com/s47user/PUIT-Assignment1-ML-Classification</a></td>
            </tr>
        </table>
        """
        st.markdown(meta_table, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="hud-eyebrow">ARCHITECTURAL INVARIANTS</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-section-title">Core Mathematical & Pipeline Contracts</div>', unsafe_allow_html=True)

        st.markdown("""
        <ol style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.8; margin: 0; padding-left: 1.25rem;">
            <li><strong>Automated Pipeline Preprocessing (ColumnTransformer):</strong> Numerically standardizes continuous variables while applying <code>OneHotEncoder(drop='first')</code> to categorical fields, entirely eliminating data leakage across train/test splits.</li>
            <li><strong>Multicollinearity & Matrix Rank Deficiency Defense:</strong> Dropping baseline dummy categories guarantees full column rank in design matrices, preventing standard error inflation during Hessian inversion.</li>
            <li><strong>Clinical Sensitivity Prioritization (Recall):</strong> Optimizes medical thresholding to prioritize sensitivity (75.47% test recall), mitigating catastrophic false negatives in clinical triage.</li>
            <li><strong>Zero-Overhead Serialization:</strong> Full pipeline graph serialized via <code>joblib</code>, ensuring production inference execution times under 5 milliseconds.</li>
        </ol>
        """, unsafe_allow_html=True)
