import os
import joblib
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DOCX = os.path.join(BASE_DIR, "Emmanuel_Siaw_Asante_PYT230047_Assignment1.docx")

# Load metrics
metrics_path = os.path.join(MODELS_DIR, "metrics_summary.joblib")
metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None

doc = Document()

# Set standard 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Styling Helper Functions
def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header_block():
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_dept = p.add_run("PENTECOST UNIVERSITY\nFACULTY OF COMPUTING & INFORMATION TECHNOLOGY\nDEPARTMENT OF INFORMATION TECHNOLOGY\n")
    run_dept.font.name = "Arial"
    run_dept.font.size = Pt(11)
    run_dept.font.bold = True
    run_dept.font.color.rgb = RGBColor(26, 54, 93)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(18)
    p_title.paragraph_format.space_after = Pt(14)
    run_title = p_title.add_run("ACADEMIC ASSIGNMENT 1: MACHINE LEARNING CLASSIFICATION & CLINICAL RISK ASSESSMENT REPORT")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(16, 42, 77)

    # Metadata Table
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    data = [
        ("Student Name:", "Emmanuel Siaw Asante"),
        ("Student ID / Index Number:", "PYT230047"),
        ("Course Lecturer:", "Mr. Harry Attieku-Boateng"),
        ("GitHub Repository Link:", "https://github.com/wisabuilds/PUIT-Assignment1-ML-Classification"),
        ("Deployed Web Application:", "https://puit-assignment1-ml-classification.streamlit.app/"),
        ("Submission Date / Deadline:", "November 2026 | Deadline: November 30th")
    ]

    for idx, (label, val) in enumerate(data):
        row = table.rows[idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.width = Inches(2.3)
        cell_val.width = Inches(4.2)
        
        set_cell_margins(cell_lbl, 80, 80, 100, 100)
        set_cell_margins(cell_val, 80, 80, 100, 100)
        set_cell_background(cell_lbl, "F0F4F8")
        set_cell_background(cell_val, "FAFAFA")
        
        p1 = cell_lbl.paragraphs[0]
        r1 = p1.add_run(label)
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = RGBColor(26, 54, 93)

        p2 = cell_val.paragraphs[0]
        r2 = p2.add_run(val)
        r2.font.size = Pt(9.5)
        if "http" in val:
            r2.font.color.rgb = RGBColor(11, 84, 164)
            r2.font.underline = True
        else:
            r2.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(18)

def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(26, 54, 93)
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(43, 108, 176)
    return p

def add_body_p(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_bullet_p(bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    r_bold = p.add_run(bold_prefix + " ")
    r_bold.font.name = "Calibri"
    r_bold.font.size = Pt(10.5)
    r_bold.font.bold = True
    r_bold.font.color.rgb = RGBColor(26, 54, 93)
    
    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10.5)
    r_text.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_figure_with_caption(fig_name, caption_text, width=Inches(5.6)):
    fig_path = os.path.join(FIGS_DIR, fig_name)
    if os.path.exists(fig_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.add_run().add_picture(fig_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = "Calibri"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(74, 85, 104)

print("Building Document Structure...")
add_header_block()

# -----------------------------------------------------------------------------
# TABLE OF CONTENTS / TABLE OF FIGURES
# -----------------------------------------------------------------------------
add_heading_1("Table of Contents")
toc_items = [
    ("1. Executive Summary & Assessment Overview", "Page 2"),
    ("2. Task 1: Scikit-Learn Iris Botanical Classification (10 Marks)", "Page 2"),
    ("    2.1 Problem Formulation & Dataset Description", "Page 2"),
    ("    2.2 Exploratory Data Analysis & Feature Interdependence", "Page 2"),
    ("    2.3 Supervised Model Architecture & Training", "Page 3"),
    ("    2.4 Quantitative Evaluation Metrics & Error Matrix Analysis", "Page 3"),
    ("3. Task 2: Heart Disease Clinical Risk Prediction", "Page 4"),
    ("    3.1 Problem Definition & Dataset Structure", "Page 4"),
    ("    3.2 Mandatory OneHotEncoder Preprocessing Pipeline", "Page 4"),
    ("    3.3 Exploratory Analysis & Clinical Feature Trends", "Page 4"),
    ("    3.4 Model Selection, Hyperparameter Tuning & Cross-Comparison", "Page 5"),
    ("    3.5 Confusion Matrix & Receiver Operating Characteristic (ROC) Analysis", "Page 5"),
    ("4. Interactive Web Application Architecture & Deployment", "Page 6"),
    ("5. Source Version Control & GitHub Repository Compliance", "Page 6"),
    ("6. Conclusion & Practical Recommendations", "Page 7"),
    ("7. References (IEEE / APA)", "Page 7")
]
for item, pg in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r1 = p.add_run(item)
    r1.font.name = "Calibri"
    r1.font.size = Pt(9.5)
    r2 = p.add_run(f"  {pg}")
    r2.font.name = "Calibri"
    r2.font.size = Pt(9.5)
    r2.font.italic = True
    r2.font.color.rgb = RGBColor(113, 128, 150)

add_heading_1("Table of Figures")
tof_items = [
    ("Figure 1.1: Iris Feature Correlation Heatmap", "Section 2.2"),
    ("Figure 1.2: Iris Feature Kernel Density Distributions Across Species", "Section 2.2"),
    ("Figure 1.3: Iris Model Confusion Matrix on Test Partition", "Section 2.4"),
    ("Figure 2.1: Heart Disease Diagnosis Class Balance in 500-Patient Cohort", "Section 3.3"),
    ("Figure 2.2: Distribution of Clinical Measures by Heart Disease Status", "Section 3.3"),
    ("Figure 2.3: Heart Disease Confusion Matrix (Logistic Regression Benchmark)", "Section 3.5"),
    ("Figure 2.4: Receiver Operating Characteristic (ROC) Curve & Area Under Curve", "Section 3.5")
]
for fig, sec in tof_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r1 = p.add_run(fig)
    r1.font.name = "Calibri"
    r1.font.size = Pt(9.5)
    r2 = p.add_run(f"  ({sec})")
    r2.font.name = "Calibri"
    r2.font.size = Pt(9.5)
    r2.font.italic = True
    r2.font.color.rgb = RGBColor(113, 128, 150)

doc.add_page_break()

# -----------------------------------------------------------------------------
# 1. EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
add_heading_1("1. Executive Summary & Assessment Overview")
add_body_p(
    "This technical report documents the complete implementation, rigorous evaluation, and web deployment of two distinct "
    "machine learning classification tasks as assigned by Mr. Harry Attieku-Boateng for Assignment 1. The primary objective is "
    "to demonstrate mastery over foundational and modern data mining techniques, encompassing exploratory data analysis (EDA), "
    "feature transformation pipelines, multi-class and binary supervised classification, quantitative metric auditing, and interactive deployment."
)
add_body_p(
    "The assignment comprises two central technical investigations: (1) a multi-class botanical species classification using the classic "
    "Iris dataset loaded directly from Scikit-Learn, and (2) a real-world clinical binary classification task predicting cardiovascular disease "
    "likelihood across 500 patient records using a custom pipeline featuring Scikit-Learn's OneHotEncoder. In strict adherence to academic instructions, "
    "both trained pipelines are deployed via a unified Streamlit web application, with all code tracked under Git version control on GitHub."
)

# -----------------------------------------------------------------------------
# 2. TASK 1: IRIS BOTANICAL SPECIES CLASSIFICATION
# -----------------------------------------------------------------------------
add_heading_1("2. Task 1: Scikit-Learn Iris Botanical Classification (10 Marks)")
add_heading_2("2.1 Problem Formulation & Dataset Description")
add_body_p(
    "The Iris dataset is a classical multi-class benchmark in machine learning originally introduced by British statistician Ronald Fisher. "
    "Loaded directly from Scikit-Learn's dataset repository (sklearn.datasets.load_iris), the dataset comprises 150 balanced botanical "
    "observations divided equally across three distinct species: Iris setosa, Iris versicolor, and Iris virginica (50 observations per class). "
    "Each observation consists of four continuous numerical features measured in centimeters: sepal length, sepal width, petal length, and petal width."
)

add_heading_2("2.2 Exploratory Data Analysis & Feature Interdependence")
add_body_p(
    "Prior to model estimation, an exploratory correlation analysis was conducted to establish inter-variable dependencies and assess class separability. "
    "As visualized in Figure 1.1, petal length and petal width exhibit an exceptionally strong positive linear correlation (r = 0.96), while both petal "
    "dimensions demonstrate substantial positive correlations with sepal length (r = 0.87 and r = 0.82 respectively). Conversely, sepal width shows weak "
    "to moderate negative correlations with all other dimensions."
)

add_figure_with_caption("iris_correlation_heatmap.png", "Figure 1.1: Pearson Correlation Heatmap across Iris Morphological Features.")

add_body_p(
    "Kernel density estimations across the three botanical species (Figure 1.2) reveal that Iris setosa forms a distinctly separable cluster based solely "
    "on petal length (< 2.5 cm) and petal width (< 0.8 cm). In contrast, Iris versicolor and Iris virginica display slight feature overlap along their "
    "distribution tails, providing a meaningful classification challenge requiring non-linear boundary estimation."
)

add_figure_with_caption("iris_feature_distributions.png", "Figure 1.2: Feature Kernel Density Distributions across the Three Iris Species.")

add_heading_2("2.3 Supervised Model Architecture & Training")
add_body_p(
    "The 150 instances were split into training (75%, n=112) and testing (25%, n=38) partitions using stratified sampling to maintain exact class proportions. "
    "A Scikit-Learn Pipeline was engineered incorporating a StandardScaler for zero-mean unit-variance normalization, followed by an ensemble "
    "Random Forest Classifier configured with 100 decision estimators, a maximum depth of 4 to avert overfitting, and a seeded pseudo-random state."
)

add_heading_2("2.4 Quantitative Evaluation Metrics & Error Matrix Analysis")
add_body_p(
    "To satisfy the lecturer's directive to calculate all evaluation metrics, the trained model was subjected to a rigorous evaluation on the unseen test "
    "partition (38 samples). Metrics recorded include Overall Accuracy, Macro-averaged Precision, Macro-averaged Recall, and Macro-averaged F1-Score."
)

# Table for Iris
iris_table = doc.add_table(rows=5, cols=3)
iris_table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Evaluation Metric", "Formula / Definition", "Observed Empirical Score"]
for i, h in enumerate(headers):
    cell = iris_table.rows[0].cells[i]
    set_cell_background(cell, "1A365D")
    set_cell_margins(cell, 100, 100, 120, 120)
    p = cell.paragraphs[0]
    r = p.add_run(h)
    r.font.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(255, 255, 255)

iris_vals = [
    ("Overall Accuracy", "(TP + TN) / (TP + TN + FP + FN)", f"{metrics['iris']['accuracy']*100:.2f}%"),
    ("Macro Precision", "Mean of TP / (TP + FP) across all classes", f"{metrics['iris']['precision']*100:.2f}%"),
    ("Macro Recall", "Mean of TP / (TP + FN) across all classes", f"{metrics['iris']['recall']*100:.2f}%"),
    ("Macro F1-Score", "Harmonic mean of Precision and Recall", f"{metrics['iris']['f1']*100:.2f}%")
]

for row_idx, row_data in enumerate(iris_vals, 1):
    row = iris_table.rows[row_idx]
    for col_idx, text in enumerate(row_data):
        cell = row.cells[col_idx]
        set_cell_background(cell, "F7FAFC" if row_idx % 2 == 1 else "FFFFFF")
        set_cell_margins(cell, 80, 80, 120, 120)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.font.size = Pt(9.5)
        if col_idx == 2:
            r.font.bold = True
            r.font.color.rgb = RGBColor(43, 108, 176)

doc.add_paragraph().paragraph_format.space_after = Pt(8)

add_figure_with_caption("iris_confusion_matrix.png", "Figure 1.3: Confusion Matrix for Iris Random Forest Classifier on Test Partition (n=38).")

add_body_p(
    "Analysis of the Confusion Matrix (Figure 1.3) illustrates exceptional classification fidelity: 100% of Setosa specimens (12/12) were categorized "
    "without error, while 12 out of 13 Versicolor and 11 out of 13 Virginica specimens were correctly classified. The minimal confusion observed between "
    "Versicolor and Virginica aligns perfectly with their morphological boundary overlap identified during preliminary exploratory data analysis."
)

# -----------------------------------------------------------------------------
# 3. TASK 2: HEART DISEASE PREDICTION
# -----------------------------------------------------------------------------
add_heading_1("3. Task 2: Heart Disease Clinical Risk Prediction")
add_heading_2("3.1 Problem Definition & Dataset Structure")
add_body_p(
    "Cardiovascular diseases represent a leading global cause of premature mortality. In this task, a clinical dataset consisting of 500 patient "
    "records was analyzed to predict the presence or absence of heart disease (binary outcome: 0 = Absence, 1 = Presence). As highlighted during the "
    "lecture, the cohort comprises 494 usable patient profiles (here expanded to the full 500-instance dataset provided). The feature space encompasses "
    "six continuous clinical variables (Age, Resting Blood Pressure, Serum Cholesterol, Body Mass Index, Physical Activity score, and Stress Level) "
    "and four categorical demographic/clinical attributes (Sex, Smoker status, Diabetes diagnosis, and Family History of heart disease)."
)

add_heading_2("3.2 Mandatory OneHotEncoder Preprocessing Pipeline")
add_body_p(
    "A pivotal requirement stipulated by Mr. Harry Attieku-Boateng was the explicit, sound utilization of Scikit-Learn's OneHotEncoder for categorical "
    "feature transformation. Categorical attributes such as Sex (Male/Female), Smoker (Yes/No), Diabetes (Yes/No), and Family History (Yes/No) cannot "
    "be natively parsed by numerical optimization algorithms without imposing unintended ordinal hierarchies. Arbitrary integer substitution (e.g., assigning "
    "0 and 1) can introduce artificial geometric assumptions."
)
add_body_p(
    "To resolve this cleanly, a unified Scikit-Learn ColumnTransformer was constructed. Continuous variables were routed through a StandardScaler, "
    "while categorical features were transformed via OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'). The drop='first' "
    "hyperparameter is critical as it drops redundant baseline dummy indicators, directly mitigating collinearity (the dummy variable trap) and "
    "stabilizing model parameter variance."
)

add_heading_2("3.3 Exploratory Analysis & Clinical Feature Trends")
add_body_p(
    "Exploratory evaluation of class proportions (Figure 2.1) confirms a balanced diagnostic representation: 266 patients (53.2%) tested positive for heart "
    "disease, while 234 patients (46.8%) tested negative. This balance eliminates class-imbalance pathologies and ensures reliable sensitivity measures."
)

add_figure_with_caption("heart_class_balance.png", "Figure 2.1: Diagnostic Target Distribution across the 500-Patient Clinical Cohort.")

add_body_p(
    "Clinical distributions partitioned by heart disease diagnosis (Figure 2.2) indicate that patients exhibiting cardiovascular abnormalities "
    "consistently demonstrate higher median serum cholesterol levels, elevated resting blood pressure readings, and higher reported baseline stress levels."
)

add_figure_with_caption("heart_numerical_distributions.png", "Figure 2.2: Boxplot Distributions of Clinical Variables Stratified by Disease Status.")

add_heading_2("3.4 Model Selection, Hyperparameter Tuning & Cross-Comparison")
add_body_p(
    "To establish an objective benchmark, three distinct classification algorithms were embedded into identical preprocessing pipelines and evaluated "
    "across a 20% stratified test set (n=100): Logistic Regression (with L2 regularization), Random Forest Classifier (150 trees, max depth 6), and "
    "Gradient Boosting Classifier (100 estimators, learning rate 0.08)."
)

# Comparison Table
comp_table = doc.add_table(rows=4, cols=6)
comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_comp = ["Candidate Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
for i, h in enumerate(headers_comp):
    cell = comp_table.rows[0].cells[i]
    set_cell_background(cell, "1A365D")
    set_cell_margins(cell, 100, 100, 100, 100)
    p = cell.paragraphs[0]
    r = p.add_run(h)
    r.font.bold = True
    r.font.size = Pt(9.0)
    r.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, res in enumerate(metrics["heart"]["comparison"], 1):
    row = comp_table.rows[r_idx]
    row_vals = [
        res["Model"],
        f"{res['Accuracy']*100:.2f}%",
        f"{res['Precision']*100:.2f}%",
        f"{res['Recall']*100:.2f}%",
        f"{res['F1-Score']*100:.2f}%",
        f"{res['ROC-AUC']:.3f}"
    ]
    for c_idx, val in enumerate(row_vals):
        cell = row.cells[c_idx]
        set_cell_background(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.size = Pt(9.0)
        if res["Model"] == "Logistic Regression" and c_idx > 0:
            r.font.bold = True

doc.add_paragraph().paragraph_format.space_after = Pt(8)

add_body_p(
    "As tabulated above, Logistic Regression emerged as the optimal clinical screening model, achieving an F1-Score of 64.00% and an outstanding "
    "Recall (Sensitivity) of 75.47%. In healthcare diagnostics, prioritizing Recall is crucial, as false negatives (failing to identify an afflicted "
    "patient) carry significantly higher clinical risks than false positives."
)

add_heading_2("3.5 Confusion Matrix & Receiver Operating Characteristic (ROC) Analysis")
add_body_p(
    "Figure 2.3 portrays the Confusion Matrix for the selected Logistic Regression model. Out of 53 actual positive patients in the test sample, "
    "the model successfully detected 40 positive cases (75.5% true positive rate), underscoring its efficacy as a primary clinical triage mechanism."
)

add_figure_with_caption("heart_confusion_matrix.png", "Figure 2.3: Heart Disease Confusion Matrix on 100-Patient Test Partition.")

add_figure_with_caption("heart_roc_curve.png", "Figure 2.4: Receiver Operating Characteristic (ROC) Curve Demonstrating Discrimination Capability.")

# -----------------------------------------------------------------------------
# 4. WEB APPLICATION & DEPLOYMENT
# -----------------------------------------------------------------------------
add_heading_1("4. Interactive Web Application Architecture & Deployment")
add_body_p(
    "To fulfill the directive requiring deployed interactive software with a live URL, a multi-page interactive web application was engineered "
    "using the Streamlit framework. The application decouples interface logic from model estimation by serializing trained Scikit-Learn pipelines "
    "into joblib binaries, ensuring millisecond-level inference latency."
)
add_bullet_p("Iris Flower Classifier Tab:", "Provides interactive real-time sliders for all four morphological dimensions. As adjustments occur, the model renders immediate species classifications accompanied by comparative probability bar charts.")
add_bullet_p("Heart Disease Predictor Tab:", "Features a clinical intake form accommodating demographic parameters, continuous vitals, and categorical toggles. Upon submission, the OneHotEncoder pipeline transforms raw inputs dynamically and computes patient risk tiering.")
add_bullet_p("Evaluation & Transparency Tab:", "Displays full empirical metrics tables, embedded confusion matrices, and ROC curves directly to clinical evaluators, ensuring auditability.")

# -----------------------------------------------------------------------------
# 5. SOURCE CONTROL & GITHUB COMPLIANCE
# -----------------------------------------------------------------------------
add_heading_1("5. Source Version Control & GitHub Repository Compliance")
add_body_p(
    "In accordance with Mr. Harry Attieku-Boateng's explicit grading mandate ('Every work must be committed and pushed to GitHub; if I don't see "
    "your GitHub repo, the work is zero'), the entire codebase has been structured into a modular repository. Version control checkpoints "
    "document data ingestion, exploratory figure generation, model pipeline assembly, Streamlit UI construction, and technical report synthesis. "
    "The repository includes comprehensive setup documentation in README.md and dependency declarations in requirements.txt."
)

# -----------------------------------------------------------------------------
# 6. CONCLUSION & RECOMMENDATIONS
# -----------------------------------------------------------------------------
add_heading_1("6. Conclusion & Practical Recommendations")
add_body_p(
    "This project successfully operationalized end-to-end data mining and machine learning pipelines across both botanical multi-class "
    "and clinical binary classification problems. Key engineering milestones attained include: (1) achieving 92.11% accuracy on the Iris benchmark, "
    "(2) deploying an automated Scikit-Learn OneHotEncoder and StandardScaler ColumnTransformer for real-world patient data, (3) validating "
    "a high-sensitivity (75.47% recall) heart disease diagnostic model, and (4) delivering an accessible, cloud-ready web interface."
)

# -----------------------------------------------------------------------------
# 7. REFERENCES
# -----------------------------------------------------------------------------
add_heading_1("7. References")
refs = [
    "[1] Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.",
    "[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "[3] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "[4] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media.",
    "[5] Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874."
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(ref)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(74, 85, 104)

doc.save(OUTPUT_DOCX)
print(f"✓ Technical Report successfully generated at:\n  {OUTPUT_DOCX}")
