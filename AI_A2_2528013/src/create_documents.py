"""Generate the technical report and evidence PDFs from current project artifacts."""
from pathlib import Path
import json
import csv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image, KeepTogether
)

ROOT = Path(__file__).resolve().parent.parent
CODE = "AI-A2-8013"
REG = "25/28013"
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"
DOCS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)


def load_json(name):
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def p(text, style):
    return Paragraph(text, style)


def make_table(rows, widths=None, font_size=8):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172b4d")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(18 * mm, 10 * mm, f"{CODE} | Synthetic classroom decision-support prototype")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    reg_metrics = load_json("regression_metrics.json")
    cls_metrics = load_json("classification_metrics.json")
    cluster = load_json("clustering_report.json")
    validation = load_json("data_validation.json")
    fingerprint = (ART / "data_fingerprint.txt").read_text(encoding="utf-8") if (ART / "data_fingerprint.txt").exists() else "Not generated yet."
    try:
        with (ROOT / "data" / "AI_A2_2528013.csv").open(encoding="utf-8", newline="") as f:
            raw_rows = sum(1 for _ in csv.DictReader(f))
        with (ROOT / "data" / "AI_A2_2528013_clean.csv").open(encoding="utf-8", newline="") as f:
            clean_rows = sum(1 for _ in csv.DictReader(f))
    except OSError:
        raw_rows, clean_rows = 0, 0

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontSize=15, leading=18, spaceAfter=9, textColor=colors.HexColor("#17365d")))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Heading2"], fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.6, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7.5, leading=9))
    styles.add(ParagraphStyle(name="Center", parent=styles["BodyText"], alignment=TA_CENTER, leading=14))

    out = DOCS / "AI_A2_2528013_REPORT.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=15*mm, bottomMargin=17*mm, title="Musanze Harvest Intelligence Service Technical Report", author="Student author")
    story = []

    # Page 1: identity and decision problem
    story += [Spacer(1, 15*mm), p("Musanze Harvest Intelligence Service", styles["CoverTitle"]), p("SWE 3513 — Individual Assignment 2 | Technical Report", styles["Center"]), Spacer(1, 9*mm)]
    story += [make_table([["Identity field", "Value"], ["Student full name", "Mahgoub Adil Ahmed Alhassan"], ["Registration number", REG], ["Institutional username", "ug2528013@ines.ac.rw"], ["Verification code", CODE]], [55*mm, 115*mm], 9), Spacer(1, 7*mm)]
    story += [p("Decision problem", styles["SubTitle"]), p("A potato cooperative field officer needs a reproducible tool to estimate harvest quantity, flag potentially high-risk fields, and group farms with similar measured conditions. This prototype generates synthetic records, trains three machine-learning tasks, and exposes predictions through a validated API.", styles["BodySmall"]), p("Prediction questions", styles["SubTitle"]), p("1. What harvest yield (tons) does the regression model estimate? 2. Does the classifier label the field as high risk? 3. What is the estimated high-risk probability? 4. Which unsupervised cluster is most similar to this record?", styles["BodySmall"]), p("Ethical boundary", styles["SubTitle"]), p("This is an educational decision-support prototype using synthetic data. It must not replace an agronomist or determine access to finance, insurance, food assistance, or public services. Human review is required for any consequential use.", styles["BodySmall"]), PageBreak()]

    # Page 2: data and reproducibility
    story += [p("2. Data and reproducibility", styles["SectionTitle"]), p("The personal seed is the final four registration digits (8013). NumPy default_rng(8013) generates 240 + (8013 mod 41) = 258 base records. Three exact duplicates are appended, resulting in 261 raw rows. The cleaner removes the three duplicate rows and retains 258 records. Missing values are deliberately inserted into soil_moisture_pct and fertilizer_kg, then imputed only within the training pipelines.", styles["BodySmall"])]
    story += [make_table([["Column", "Meaning / generation rule"], ["rainfall_mm", "Uniform 600–1400 mm"], ["avg_temp_c", "Uniform 12–26 °C"], ["soil_moisture_pct", "Uniform 20–80%; 5% missingness introduced"], ["fertilizer_kg", "Uniform 0–250 kg; 5% missingness introduced"], ["field_size_ha", "Uniform 0.2–4.0 ha"], ["season", "Random category A or B"], ["cooperative_support", "Random binary value 0 or 1"], ["yield_tons", "Specified formula + N(0, 0.80), clipped at 0.50"], ["high_risk", "1 if yield per hectare is below the median, else 0"]], [43*mm, 127*mm], 7.5), Spacer(1, 4*mm)]
    story += [p(f"Raw rows: {raw_rows}. Clean rows: {clean_rows}. Verification code: {CODE}.", styles["BodySmall"]), p("Dataset fingerprint", styles["SubTitle"]), p(fingerprint.replace("\n", "<br/>"), styles["Tiny"]), p("Validation report", styles["SubTitle"]), p(f"{json.dumps(validation, ensure_ascii=False)[:900]}", styles["Tiny"]), PageBreak()]

    # Page 3: pipeline
    story += [p("3. Pipeline design and leakage control", styles["SectionTitle"]), p("run_all.py checks that the supplied seed matches the last four registration digits, then regenerates data, removes duplicates, validates the dataset, trains the supervised models, trains clustering, and rebuilds the required artifacts. The fixed split uses test_size=0.20 and random_state=8013.", styles["BodySmall"]), p("Supervised preprocessing", styles["SubTitle"]), p("The regression and classification models use scikit-learn Pipelines and ColumnTransformer. Numeric missing values are median-imputed and numeric inputs scaled with StandardScaler. Season is imputed using the most frequent category and encoded using OneHotEncoder(handle_unknown='ignore'). These transformations are fitted on the training subset only and saved with each final model.", styles["BodySmall"]), p("Target leakage", styles["SubTitle"]), p("The input features are rainfall, temperature, soil moisture, fertilizer, field size, season, and cooperative support. yield_tons and high_risk are targets, never classifier inputs. Yield per hectare is used only to create the high-risk label and is not a model feature.", styles["BodySmall"]), p("Candidates", styles["SubTitle"]), p("Regression: mean DummyRegressor baseline, LinearRegression, RandomForestRegressor. Classification: most-frequent DummyClassifier baseline, LogisticRegression, RandomForestClassifier. Clustering: median imputation, StandardScaler, and KMeans for K=2 through K=6 evaluated using silhouette score.", styles["BodySmall"]), p("Determinism", styles["SubTitle"]), p("The generator and models use the personal seed 8013. Re-run the full command and compare artifacts/data_fingerprint.txt to confirm reproducibility in the submission environment.", styles["BodySmall"]), PageBreak()]

    # Page 4: metrics and plots
    story += [p("4. Regression and classification", styles["SectionTitle"]), p("Regression metrics on the held-out test set", styles["SubTitle"])]
    rmodels = reg_metrics.get("models", {})
    rrows = [["Model", "MAE", "RMSE", "R²"]]
    for name, m in rmodels.items():
        rrows.append([name, f"{m.get('MAE',0):.3f}", f"{m.get('RMSE',0):.3f}", f"{m.get('R2',0):.3f}"])
    story += [make_table(rrows, [45*mm, 35*mm, 35*mm, 35*mm], 8), p(f"Selected regression model: {reg_metrics.get('selected_model','linear_regression')}. {reg_metrics.get('selection_reason','')}", styles["BodySmall"]), p("Classification metrics", styles["SubTitle"])]
    cmodels = cls_metrics.get("models", {})
    crows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for name, m in cmodels.items():
        crows.append([name, f"{m.get('accuracy',0):.3f}", f"{m.get('precision',0):.3f}", f"{m.get('recall',0):.3f}", f"{m.get('f1',0):.3f}", f"{m.get('roc_auc',0):.3f}"])
    story += [make_table(crows, [37*mm, 25*mm, 25*mm, 25*mm, 20*mm, 25*mm], 7), p(f"Selected classifier: {cls_metrics.get('selected_model','logistic_regression')}. {cls_metrics.get('selection_reason','')}", styles["BodySmall"])]
    residual = ART / "plots" / "residuals.png"
    confusion = ART / "plots" / "confusion_matrix.png"
    if residual.exists(): story.append(Image(str(residual), width=78*mm, height=55*mm))
    if confusion.exists(): story.append(Image(str(confusion), width=70*mm, height=55*mm))
    story += [p("Student-authored error interpretation required: choose one row in artifacts/test_predictions.csv and explain its actual yield, prediction, absolute error, and a plausible reason for the difference. Identify one false positive or false negative from the confusion matrix and explain its practical significance.", styles["Tiny"]), PageBreak()]

    # Page 5: clustering
    story += [p("5. Clustering and operational meaning", styles["SectionTitle"]), p(f"KMeans was evaluated for K=2…6. The selected K is {cluster.get('selected_k', 'not available')}; silhouette scores: {json.dumps(cluster.get('silhouette_scores', {}))}.", styles["BodySmall"]), p("The silhouette score is modest, so groups overlap and should not be interpreted as sharply separated types. Cluster profiles are averages of synthetic input features, not a farm-quality or risk score.", styles["BodySmall"])]
    profiles = cluster.get("cluster_profiles", {})
    if profiles:
        keys = sorted({k for v in profiles.values() for k in v.keys()})
        rows = [["Cluster"] + keys]
        for cid, values in profiles.items():
            rows.append([str(cid)] + [f"{values.get(k, 0):.2f}" if isinstance(values.get(k), (int,float)) else str(values.get(k)) for k in keys])
        story.append(make_table(rows, None, 6.7))
    clplot = ART / "plots" / "clusters.png"
    if clplot.exists(): story.append(Image(str(clplot), width=135*mm, height=78*mm))
    story += [p("Student-authored cluster interpretation required: choose one cluster, describe at least two feature means, compare it with another cluster, and explain why these synthetic profiles must not be used to rank real farms.", styles["BodySmall"]), PageBreak()]

    # Page 6: API and tests
    story += [p("6. API and testing", styles["SectionTitle"]), make_table([["Method / endpoint", "Purpose"], ["GET /health", "Returns service health and verification code"], ["GET /model-info", "Returns task, model, and synthetic-data information"], ["POST /predict", "Accepts validated farm features and returns yield, risk, probability, cluster, verification code"]], [45*mm, 125*mm], 8), Spacer(1, 4*mm), p("Input contract", styles["SubTitle"]), p("rainfall_mm: 600–1400; avg_temp_c: 12–26; soil_moisture_pct: 20–80; fertilizer_kg: 0–250; field_size_ha: 0.2–4.0; season: A or B; cooperative_support: 0 or 1. Missing fields, invalid types, out-of-range values, and extra fields are rejected by the API with HTTP 422.", styles["BodySmall"]), p("Valid input example", styles["SubTitle"]), p('{"rainfall_mm":1000,"avg_temp_c":20,"soil_moisture_pct":50,"fertilizer_kg":100,"field_size_ha":2,"season":"A","cooperative_support":1}', styles["Tiny"]), p("Invalid input example", styles["SubTitle"]), p('{"rainfall_mm":2000,"avg_temp_c":20,"soil_moisture_pct":50,"fertilizer_kg":100,"field_size_ha":2,"season":"A","cooperative_support":1}', styles["Tiny"]), p("Automated tests", styles["SubTitle"]), p("Nine tests cover health, model information, valid prediction, missing field, wrong type, range violation, repeatability, extra-field rejection, and synthetic-data disclosure. The prepared environment run completed with 9 passed. Re-run pytest -q on the machine used for submission and retain the actual output.", styles["BodySmall"]), p("Commands: python run_all.py --student-id 25/28013 --seed 8013 ; python predict.py --json samples/valid.json ; python predict.py --json samples/invalid.json ; pytest -q ; uvicorn app:app --host 127.0.0.1 --port 8000", styles["Tiny"]), PageBreak()]

    # Page 7: model card and reflection
    story += [p("7. Model card and reflection", styles["SectionTitle"]), p("Intended use", styles["SubTitle"]), p("Classroom demonstration of reproducible data generation, supervised learning, clustering, and validated API inference.", styles["BodySmall"]), p("Out-of-scope use", styles["SubTitle"]), p("Do not use these predictions to make or automate lending, insurance, public-service eligibility, or high-stakes farm allocation decisions. Do not treat clusters as quality labels or the predicted probability as certainty.", styles["BodySmall"]), p("Data and performance limitations", styles["SubTitle"]), p("All rows are synthetic and generated from a fixed mathematical formula. Artificial missingness does not represent real data collection. The held-out scores measure performance on this synthetic sample only and cannot demonstrate real-world agricultural accuracy. Real deployment would require representative field data, agronomist review, external validation, monitoring, and documented governance.", styles["BodySmall"]), p("Student reflection — complete in your own words before submission", styles["SubTitle"]), p("1. What did you learn from comparing the baseline with the selected models?", styles["BodySmall"]), Spacer(1, 7*mm), p("2. Which model error concerned you most, and why?", styles["BodySmall"]), Spacer(1, 7*mm), p("3. What would you change or test next, and why?", styles["BodySmall"]), Spacer(1, 7*mm), p("4. Which code component can you explain and modify independently?", styles["BodySmall"]), Spacer(1, 7*mm), p("Required pre-submission completion", styles["SubTitle"]), p("Replace identity placeholders, write your own error and cluster interpretations, review the AI-use disclosure, add a real repository/commit hash if required, and record the required continuous unedited demo video in your actual environment.", styles["BodySmall"])]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def build_test_log():
    out = EVIDENCE / "AI_A2_2528013_TEST_LOG.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    story = [Paragraph("Automated Test Log", styles["Title"]), Paragraph(f"Project: Musanze Harvest Intelligence Service | {REG} | {CODE}", styles["BodyText"]), Spacer(1, 6*mm), Paragraph("Test cases", styles["Heading2"])]
    rows = [["Test", "Purpose"], ["test_health", "GET /health returns healthy"], ["test_model_info", "Model information includes verification code"], ["test_valid_prediction", "Valid record returns prediction"], ["test_missing_field", "Missing field returns HTTP 422"], ["test_wrong_type", "Invalid type returns HTTP 422"], ["test_out_of_range", "Out-of-range value returns HTTP 422"], ["test_reproducibility", "Repeated input returns same result"], ["test_extra_field_rejected", "Extra input fields return HTTP 422"], ["test_model_info_discloses_synthetic_data", "Model info discloses synthetic dataset"]]
    story.append(make_table(rows, [55*mm, 110*mm], 8))
    story += [Spacer(1, 6*mm), Paragraph("Preparation environment result: 9 passed. Re-run <font name='Courier'>python -m pytest -q</font> on the submission computer and replace this evidence if the result differs. This document records the preparation run; it is not a substitute for showing the actual command in the demonstration video.", styles["BodyText"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def build_declaration():
    out = EVIDENCE / "AI_A2_2528013_DECLARATION.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=22*mm, leftMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)
    story = [Paragraph("Student Declaration — Draft for Student Review", styles["Title"]), Spacer(1, 5*mm), Paragraph("INES SWE 3513 Individual Assignment 2 — Musanze Harvest Intelligence Service", styles["BodyText"]), Spacer(1, 5*mm), Paragraph(f"Full name: Mahgoub Adil Ahmed Alhassan<br/>Registration number: {REG}<br/>Institutional username: ug2528013@ines.ac.rw<br/>Verification code: {CODE}", styles["BodyText"]), Spacer(1, 8*mm), Paragraph("Declaration wording must be checked against the institution's official requirement before signing. Do not sign this draft until you have reviewed the final code, data, model results, report, and AI-use disclosure and can explain the work. Accurately disclose assistance received and distinguish generated assistance from your own interpretations and decisions.", styles["BodyText"]), Spacer(1, 15*mm), Paragraph("Student declaration (complete in accordance with the official wording):", styles["Heading2"]), Spacer(1, 15*mm), Paragraph("Signature: ______________________________________________", styles["BodyText"]), Spacer(1, 8*mm), Paragraph("Date: __________________________________________________", styles["BodyText"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == "__main__":
    print(f"Created report: {build_report()}")
    print(f"Created test log: {build_test_log()}")
    print(f"Created declaration draft: {build_declaration()}")
