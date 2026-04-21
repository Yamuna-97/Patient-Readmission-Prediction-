import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import io
import re
import base64
from datetime import datetime
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

st.set_page_config(page_title="Diabetes Readmission Predictor", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0f1e; color: #e8eaf0; }
    h1 {
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 2.2rem; color: #7eb8f7; letter-spacing: -0.5px;
    }
    .section-title {
        font-family: 'Space Mono', monospace; font-size: 0.72rem;
        letter-spacing: 3px; text-transform: uppercase; color: #4a7fa5;
        margin: 1.6rem 0 0.6rem; border-left: 3px solid #2a5f8a; padding-left: 10px;
    }
    label, .stSelectbox label, .stNumberInput label {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important; color: #8faecb !important; letter-spacing: 1px;
    }
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: #111827 !important; border: 1px solid #1e3a5f !important;
        border-radius: 6px !important; color: #dce8f5 !important;
        font-family: 'Space Mono', monospace !important;
    }
    .stTextInput > div > div > input {
        background: #111827 !important; border: 1px solid #1e3a5f !important;
        border-radius: 6px !important; color: #dce8f5 !important;
        font-family: 'Space Mono', monospace !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a4a7a, #2a7abf);
        color: white; font-family: 'Space Mono', monospace;
        font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase;
        border: none; border-radius: 6px; padding: 0.6rem 2.4rem;
        margin-top: 1.2rem; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5f9a, #3a8acf); transform: translateY(-1px);
    }
    .result-box {
        margin-top: 1.8rem; padding: 1.6rem 2rem; border-radius: 10px;
        font-family: 'Space Mono', monospace; font-size: 0.95rem; letter-spacing: 1px;
    }
    .result-no   { background: #0d2a1a; border-left: 5px solid #2ecc71; color: #7ef5a8; }
    .result-lt30 { background: #2a1010; border-left: 5px solid #e74c3c; color: #f5817e; }
    .result-gt30 { background: #1a1a0a; border-left: 5px solid #f39c12; color: #f5c97e; }
    .sub-text { font-family: 'Syne', sans-serif; font-size: 0.82rem; color: #6a8fab; margin-top: 0.4rem; }
    .attr-box {
        background: #0d1a2e; border: 1px solid #1e3a5f; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 1rem; font-family: 'Space Mono', monospace;
        font-size: 0.75rem; color: #8faecb;
    }
    .attr-box span { color: #dce8f5; font-weight: bold; }
    .parse-ok {
        background: #0d2a1a; border: 1px solid #2ecc71; border-radius: 6px;
        padding: 0.7rem 1rem; font-family: 'Space Mono', monospace;
        font-size: 0.72rem; color: #7ef5a8; margin-top: 0.5rem;
    }
    .upload-zone {
        background: #0d1a2e; border: 1.5px dashed #2a5f8a; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
        font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #6a8fab;
    }
    hr { border-color: #1e3a5f; margin: 1.5rem 0; }
    .record-tag {
        display: inline-block; background: #1e3a5f; color: #7eb8f7;
        font-family: 'Space Mono', monospace; font-size: 0.68rem;
        letter-spacing: 2px; padding: 3px 10px; border-radius: 4px; margin-bottom: 0.5rem;
    }
    .pdf-badge {
        display: inline-block; background: #0d2a1a; color: #2ecc71; border: 1px solid #2ecc71;
        font-family: 'Space Mono', monospace; font-size: 0.6rem;
        letter-spacing: 1px; padding: 1px 6px; border-radius: 3px; margin-left: 6px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("diabetes_model.h5")
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, encoders, scaler

model, encoders, scaler = load_assets()

CATEGORICAL_COLS = [
    "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed"
]

PAYER_CODES = ["?","BC","MD","HM","UN","SP","MC","OG","OT","CH","PO","DM","CM","WC","SI","FR","CP","MP","BP"]
MED_SPECIALTIES = [
    "?","InternalMedicine","Emergency/Trauma","Family/GeneralPractice",
    "Cardiology","Surgery-General","Orthopedics","Nephrology","Orthopedics-Reconstructive",
    "Pulmonology","Psychiatry","Urology","ObstetricsandGynecology","Gastroenterology",
    "Radiology","Neurology","Anesthesiology","Oncology","Surgery-Cardiovascular",
    "Hematology","Endocrinology","Surgery-Neuro","Pediatrics","Other"
]

MED_KEYS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "examide","citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone",
]

# canonical lookup: normalised label → dict key
MED_LABEL_MAP = {k.replace("-","").lower(): k for k in MED_KEYS}
MED_LABEL_MAP.update({
    "glyburidemetformin":           "glyburide-metformin",
    "glipizidemetformin":           "glipizide-metformin",
    "glimepiridepioglitazone":      "glimepiride-pioglitazone",
    "metforminrosiglitazone":       "metformin-rosiglitazone",
    "metforminpioglitazone":        "metformin-pioglitazone",
})


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf_fields(pdf_bytes: bytes) -> dict:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    def find_after(pattern, cast=str, default=None):
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                same = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()
                for candidate in [same] + ([lines[i+1].strip()] if i+1 < len(lines) else []):
                    if candidate:
                        try:
                            return cast(candidate)
                        except Exception:
                            pass
        return default

    def find_int(pattern, default, lo, hi):
        val = find_after(pattern, cast=lambda x: int(x.split()[0]), default=default)
        if val is None:
            return default
        return max(lo, min(hi, val))

    def find_choice(pattern, choices, default):
        val = find_after(pattern, cast=str, default=None)
        if val is None:
            return default
        if val in choices:
            return val
        for c in choices:
            if c.lower() == val.lower():
                return c
        return default

    parsed = {}

    parsed["admission_type_id"]        = find_int(r"admission\s+type\s+id",        1,  1,  8)
    parsed["discharge_disposition_id"] = find_int(r"discharge\s+disposition\s+id", 1,  1, 29)
    parsed["admission_source_id"]      = find_int(r"admission\s+source\s+id",      1,  1, 25)
    parsed["payer_code"]               = find_choice(r"payer\s+code",        PAYER_CODES,    "?")
    parsed["medical_specialty"]        = find_choice(r"medical\s+specialty",  MED_SPECIALTIES, "?")

    parsed["time_in_hospital"]   = find_int(r"time\s+in\s+hospital",       4,  1,  14)
    parsed["num_lab_procedures"] = find_int(r"lab\s+procedures?",          43, 1, 132)
    parsed["num_procedures"]     = find_int(r"(?<!\w)procedures?(?!\s+id)", 1,  0,   6)
    parsed["num_medications"]    = find_int(r"medications?(?!\s+detail)",  16,  1,  81)
    parsed["number_outpatient"]  = find_int(r"outpatient\s+visits?",        0,  0,  42)
    parsed["number_emergency"]   = find_int(r"emergency\s+visits?",         0,  0,  76)
    parsed["number_inpatient"]   = find_int(r"inpatient\s+visits?",         0,  0,  21)
    parsed["number_diagnoses"]   = find_int(r"number\s+of\s+diagnoses",     8,  1,  16)

    parsed["diag_1"] = find_after(r"primary\s+diagnosis",    cast=str, default="250.01") or "250.01"
    parsed["diag_2"] = find_after(r"secondary\s+diagnosis",  cast=str, default="276")    or "276"
    parsed["diag_3"] = find_after(r"additional\s+diagnosis", cast=str, default="250.01") or "250.01"

    parsed["max_glu_serum"] = find_choice(r"max\s+glu(?:cose)?\s+serum", ["None",">200",">300","Norm"], "None")
    parsed["A1Cresult"]     = find_choice(r"a1c\s*result",               ["None",">7",">8","Norm"],     "None")

    med_val_options = {"no","steady","up","down"}
    for i, line in enumerate(lines):
        norm = line.lower().replace(" ", "").replace("-", "")
        if norm in MED_LABEL_MAP:
            canon_key = MED_LABEL_MAP[norm]
            nxt = lines[i+1].strip() if i+1 < len(lines) else "No"
            parsed[canon_key] = nxt if nxt.lower() in med_val_options else "No"

    for k in MED_KEYS:
        if k not in parsed:
            parsed[k] = "No"

    change_raw = find_after(r"change\s+in\s+meds?", cast=str, default="No")
    parsed["change"]      = change_raw if change_raw in ["No","Ch"] else "No"

    dm_raw = find_after(r"diabetes\s+med", cast=str, default="No")
    parsed["diabetesMed"] = dm_raw if dm_raw in ["No","Yes"] else "No"

    return parsed


# ── PDF report generator ──────────────────────────────────────────────────────

def generate_pdf_report(raw_input, pred, probs, patient_name="N/A", patient_id="N/A"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=15*mm, bottomMargin=15*mm)

    dark_bg   = colors.HexColor("#0a0f1e")
    blue_mid  = colors.HexColor("#1a4a7a")
    blue_lite = colors.HexColor("#7eb8f7")
    mono_txt  = colors.HexColor("#dce8f5")
    muted     = colors.HexColor("#4a7fa5")
    green     = colors.HexColor("#2ecc71")
    red       = colors.HexColor("#e74c3c")
    orange    = colors.HexColor("#f39c12")
    row_alt   = colors.HexColor("#0d1a2e")

    label_colors = {0: green, 1: red, 2: orange}
    label_names  = {0: "NO Readmission", 1: "Readmitted < 30 days", 2: "Readmitted > 30 days"}
    risk_levels  = {0: "LOW RISK", 1: "HIGH RISK", 2: "MODERATE RISK"}

    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
                                  textColor=blue_lite, alignment=TA_CENTER, spaceAfter=2)
    sub_style   = ParagraphStyle("sub",   fontName="Helvetica",      fontSize=9,
                                  textColor=muted, alignment=TA_CENTER, spaceAfter=6)
    sec_style   = ParagraphStyle("sec",   fontName="Helvetica-Bold", fontSize=9,
                                  textColor=blue_lite, spaceBefore=10, spaceAfter=4)
    label_style = ParagraphStyle("lbl",   fontName="Helvetica-Bold", fontSize=14,
                                  textColor=label_colors[pred], alignment=TA_CENTER, spaceAfter=4)
    risk_style  = ParagraphStyle("risk",  fontName="Helvetica-Bold", fontSize=9,
                                  textColor=label_colors[pred], alignment=TA_CENTER, spaceAfter=10)
    disc_style  = ParagraphStyle("disc",  fontName="Helvetica-Oblique", fontSize=7,
                                  textColor=muted, alignment=TA_CENTER)

    def mk(txt, fname="Helvetica", size=7.5, clr=None):
        return ParagraphStyle("_", fontName=fname, fontSize=size, textColor=clr or muted)

    def sec(text):
        return [
            Spacer(1, 6),
            Paragraph(f"▸  {text.upper()}", sec_style),
            HRFlowable(width="100%", thickness=0.5, color=blue_mid, spaceAfter=4),
        ]

    def two_col(rows):
        data = []
        for i in range(0, len(rows), 2):
            l = rows[i]; r = rows[i+1] if i+1 < len(rows) else ("","")
            data.append([
                Paragraph(l[0],       mk(l[0])),
                Paragraph(str(l[1]),  mk(l[1], "Helvetica-Bold", 7.5, mono_txt)),
                Paragraph(r[0],       mk(r[0])),
                Paragraph(str(r[1]),  mk(r[1], "Helvetica-Bold", 7.5, mono_txt)),
            ])
        t = Table(data, colWidths=[42*mm]*4)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), dark_bg),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [dark_bg, row_alt]),
            ("BOX",           (0,0),(-1,-1), 0.4, blue_mid),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, colors.HexColor("#1e3a5f")),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ]))
        return t

    story = []
    story.append(Paragraph("DIABETES READMISSION PREDICTOR", title_style))
    story.append(Paragraph("AI-Assisted Clinical Decision Support Report", sub_style))
    story.append(Spacer(1, 2))

    meta = [[
        Paragraph("REPORT GENERATED", mk("","Helvetica",7)),
        Paragraph(datetime.now().strftime("%d %b %Y  %H:%M"), mk("","Helvetica-Bold",7.5,mono_txt)),
        Paragraph("PATIENT NAME", mk("","Helvetica",7)),
        Paragraph(patient_name,   mk("","Helvetica-Bold",7.5,mono_txt)),
        Paragraph("PATIENT ID",   mk("","Helvetica",7)),
        Paragraph(patient_id,     mk("","Helvetica-Bold",7.5,mono_txt)),
    ]]
    mt = Table(meta, colWidths=[30*mm,32*mm,28*mm,32*mm,22*mm,26*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), row_alt),
        ("BOX",           (0,0),(-1,-1), 0.5, blue_mid),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, colors.HexColor("#1e3a5f")),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))

    story += sec("Prediction Result")
    pt = Table([[Paragraph(label_names[pred], label_style), Paragraph(risk_levels[pred], risk_style)]],
               colWidths=[85*mm, 85*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), dark_bg),
        ("BOX",           (0,0),(0,0),   1.5, label_colors[pred]),
        ("BOX",           (1,0),(1,0),   1.5, label_colors[pred]),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 6))

    prob_rows = [[
        Paragraph("CLASS",       mk("","Helvetica-Bold",7.5)),
        Paragraph("PROBABILITY", mk("","Helvetica-Bold",7.5)),
        Paragraph("CONFIDENCE",  mk("","Helvetica-Bold",7.5)),
    ]]
    for i, (lbl, clr) in enumerate(zip(["No Readmission","< 30 Days","> 30 Days"],
                                        [green, red, orange])):
        bar = "█" * int(probs[i]*30) + "░" * (30 - int(probs[i]*30))
        prob_rows.append([
            Paragraph(lbl,                    mk("","Helvetica-Bold",8,clr)),
            Paragraph(f"{probs[i]*100:.1f}%", mk("","Helvetica-Bold",9,clr)),
            Paragraph(bar,                    mk("","Helvetica",6,clr)),
        ])
    pbt = Table(prob_rows, colWidths=[45*mm,35*mm,90*mm])
    pbt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  blue_mid),
        ("BACKGROUND",    (0,1),(-1,-1), dark_bg),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [dark_bg, row_alt, dark_bg]),
        ("BOX",           (0,0),(-1,-1), 0.4, blue_mid),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, colors.HexColor("#1e3a5f")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
    ]))
    story.append(pbt)

    story += sec("Patient Demographics")
    story.append(two_col([
        ("Race",raw_input["race"]),("Gender",raw_input["gender"]),
        ("Age", raw_input["age"]), ("Weight",raw_input["weight"]),
    ]))

    story += sec("Admission Information")
    story.append(two_col([
        ("Admission Type ID",        raw_input["admission_type_id"]),
        ("Discharge Disposition ID", raw_input["discharge_disposition_id"]),
        ("Admission Source ID",      raw_input["admission_source_id"]),
        ("Payer Code",               raw_input["payer_code"]),
        ("Medical Specialty",        raw_input["medical_specialty"]),
    ]))

    story += sec("Hospital Stay & Procedures")
    story.append(two_col([
        ("Time in Hospital (days)", raw_input["time_in_hospital"]),
        ("Lab Procedures",          raw_input["num_lab_procedures"]),
        ("Procedures",              raw_input["num_procedures"]),
        ("Medications",             raw_input["num_medications"]),
        ("Outpatient Visits",       raw_input["number_outpatient"]),
        ("Emergency Visits",        raw_input["number_emergency"]),
        ("Inpatient Visits",        raw_input["number_inpatient"]),
        ("Number of Diagnoses",     raw_input["number_diagnoses"]),
    ]))

    story += sec("Diagnoses & Lab Results")
    story.append(two_col([
        ("Primary Diagnosis (ICD9)",    raw_input["diag_1"]),
        ("Secondary Diagnosis (ICD9)",  raw_input["diag_2"]),
        ("Additional Diagnosis (ICD9)", raw_input["diag_3"]),
        ("Max Glucose Serum",           raw_input["max_glu_serum"]),
        ("A1C Result",                  raw_input["A1Cresult"]),
    ]))

    story += sec("Medication Details")
    med_rows = [(k.replace("-"," ").title(), raw_input.get(k,"No")) for k in MED_KEYS]
    med_rows += [("Change in Meds", raw_input["change"]), ("Diabetes Med", raw_input["diabetesMed"])]
    story.append(two_col(med_rows))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=blue_mid))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This report is generated by an AI model for clinical decision support only. "
        "It does not replace professional medical judgment. Always consult a qualified physician.",
        disc_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── helpers ───────────────────────────────────────────────────────────────────

def ss(key, default):
    return st.session_state.get(key, default)

def idx_in(lst, val, fallback=0):
    try:
        return lst.index(val)
    except ValueError:
        return fallback


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🏥 Diabetes Readmission Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter patient details or upload a PDF to auto-fill the form.</p>',
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Patient Record Identifier ────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Record Identifier</div>', unsafe_allow_html=True)
st.markdown('<div class="record-tag">SINGLE PATIENT RECORD</div>', unsafe_allow_html=True)
ci1, ci2, ci3 = st.columns(3)
patient_name = ci1.text_input("Patient Name", "John Doe")
patient_id   = ci2.text_input("Patient ID",   "PAT-00001")
record_date  = ci3.text_input("Record Date",  datetime.now().strftime("%Y-%m-%d"))

st.markdown("<hr>", unsafe_allow_html=True)

# ── PDF Upload ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Import from PDF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="upload-zone">'
    'Upload a patient medical PDF to auto-fill: '
    '<strong style="color:#7eb8f7">Admission Info · Hospital Stay · '
    'Diagnoses &amp; Lab Results · Medication Details</strong>.<br>'
    'Patient Demographics must still be entered manually.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_pdf = st.file_uploader("Patient Medical PDF", type=["pdf"], label_visibility="collapsed")

pdf_loaded = False
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()
    parsed = parse_pdf_fields(pdf_bytes)
    for k, v in parsed.items():
        st.session_state[f"pdf_{k}"] = v
    pdf_loaded = True
    st.markdown(
        '<div class="parse-ok">'
        '✔  PDF parsed — Admission Info, Hospital Stay, Diagnoses &amp; Medications '
        'have been loaded below. Review before predicting.'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 1: Demographics ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Demographics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
race   = c1.selectbox("Race",   ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","?"])
gender = c2.selectbox("Gender", ["Male","Female","Unknown/Invalid"])
age    = c3.selectbox("Age",    ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                                  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"])
weight = c4.selectbox("Weight", ["?","[0-25)","[25-50)","[50-75)","[75-100)",
                                  "[100-125)","[125-150)","[150-175)","[175-200)",">200"])

# ── Section 2: Admission Info (PDF-fillable) ──────────────────────────────────
badge = ' <span class="pdf-badge">FROM PDF</span>' if pdf_loaded else ""
st.markdown(f'<div class="section-title">Admission Information{badge}</div>', unsafe_allow_html=True)

_ati = list(range(1, 9))
_ddi = list(range(1, 30))
_asi = list(range(1, 26))
c1, c2, c3 = st.columns(3)
admission_type_id        = c1.selectbox("Admission Type ID",        _ati,
    index=idx_in(_ati, ss("pdf_admission_type_id", 1)))
discharge_disposition_id = c2.selectbox("Discharge Disposition ID", _ddi,
    index=idx_in(_ddi, ss("pdf_discharge_disposition_id", 1)))
admission_source_id      = c3.selectbox("Admission Source ID",      _asi,
    index=idx_in(_asi, ss("pdf_admission_source_id", 1)))

c1, c2 = st.columns(2)
payer_code        = c1.selectbox("Payer Code",        PAYER_CODES,
    index=idx_in(PAYER_CODES, ss("pdf_payer_code", "?")))
medical_specialty = c2.selectbox("Medical Specialty", MED_SPECIALTIES,
    index=idx_in(MED_SPECIALTIES, ss("pdf_medical_specialty", "?")))

# ── Section 3: Hospital Stay (PDF-fillable) ───────────────────────────────────
st.markdown(f'<div class="section-title">Hospital Stay &amp; Procedures{badge}</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
time_in_hospital   = c1.number_input("Time in Hospital (days)", 1, 14,  ss("pdf_time_in_hospital",   4))
num_lab_procedures = c2.number_input("Lab Procedures",          1, 132, ss("pdf_num_lab_procedures", 43))
num_procedures     = c3.number_input("Procedures",              0, 6,   ss("pdf_num_procedures",     1))
num_medications    = c4.number_input("Medications",             1, 81,  ss("pdf_num_medications",    16))

c1, c2, c3, c4 = st.columns(4)
number_outpatient = c1.number_input("Outpatient Visits",   0, 42, ss("pdf_number_outpatient", 0))
number_emergency  = c2.number_input("Emergency Visits",    0, 76, ss("pdf_number_emergency",  0))
number_inpatient  = c3.number_input("Inpatient Visits",    0, 21, ss("pdf_number_inpatient",  0))
number_diagnoses  = c4.number_input("Number of Diagnoses", 1, 16, ss("pdf_number_diagnoses",  8))

# ── Section 4: Diagnoses & Lab (PDF-fillable) ─────────────────────────────────
st.markdown(f'<div class="section-title">Diagnoses &amp; Lab Results{badge}</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
diag_1 = c1.text_input("Primary Diagnosis (ICD9)",    ss("pdf_diag_1", "250.01"))
diag_2 = c2.text_input("Secondary Diagnosis (ICD9)",  ss("pdf_diag_2", "276"))
diag_3 = c3.text_input("Additional Diagnosis (ICD9)", ss("pdf_diag_3", "250.01"))

_glu = ["None", ">200", ">300", "Norm"]
_a1c = ["None", ">7",   ">8",   "Norm"]
c1, c2 = st.columns(2)
max_glu_serum = c1.selectbox("Max Glucose Serum", _glu,
    index=idx_in(_glu, ss("pdf_max_glu_serum", "None")))
A1Cresult     = c2.selectbox("A1C Result",        _a1c,
    index=idx_in(_a1c, ss("pdf_A1Cresult", "None")))

# ── Section 5: Medications (PDF-fillable) ─────────────────────────────────────
st.markdown(f'<div class="section-title">Medication Details{badge}</div>', unsafe_allow_html=True)
med_options = ["No", "Steady", "Up", "Down"]
med_cols = st.columns(5)
med_fields = [
    ("metformin","Metformin"),("repaglinide","Repaglinide"),("nateglinide","Nateglinide"),
    ("chlorpropamide","Chlorpropamide"),("glimepiride","Glimepiride"),
    ("acetohexamide","Acetohexamide"),("glipizide","Glipizide"),("glyburide","Glyburide"),
    ("tolbutamide","Tolbutamide"),("pioglitazone","Pioglitazone"),
    ("rosiglitazone","Rosiglitazone"),("acarbose","Acarbose"),("miglitol","Miglitol"),
    ("troglitazone","Troglitazone"),("tolazamide","Tolazamide"),("examide","Examide"),
    ("citoglipton","Citoglipton"),("insulin","Insulin"),
    ("glyburide-metformin","Glyburide-Metformin"),("glipizide-metformin","Glipizide-Metformin"),
    ("glimepiride-pioglitazone","Glimepiride-Pioglitazone"),
    ("metformin-rosiglitazone","Metformin-Rosiglitazone"),
    ("metformin-pioglitazone","Metformin-Pioglitazone"),
]
med_values = {}
for i, (key, label) in enumerate(med_fields):
    default_val = ss(f"pdf_{key}", "No")
    med_values[key] = med_cols[i % 5].selectbox(
        label, med_options,
        index=idx_in(med_options, default_val),
        key=key
    )

_ch = ["No", "Ch"]
_dm = ["No", "Yes"]
c1, c2 = st.columns(2)
change      = c1.selectbox("Change in Meds", _ch, index=idx_in(_ch, ss("pdf_change",      "No")))
diabetesMed = c2.selectbox("Diabetes Med",   _dm, index=idx_in(_dm, ss("pdf_diabetesMed", "No")))

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("PREDICT READMISSION"):
    raw = {
        "race": race, "gender": gender, "age": age, "weight": weight,
        "admission_type_id": str(admission_type_id),
        "discharge_disposition_id": str(discharge_disposition_id),
        "admission_source_id": str(admission_source_id),
        "payer_code": payer_code, "medical_specialty": medical_specialty,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
        "change": change, "diabetesMed": diabetesMed,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        **med_values
    }

    encoded = dict(raw)
    for col in CATEGORICAL_COLS:
        val = str(encoded[col])
        if col in encoders:
            le: LabelEncoder = encoders[col]
            encoded[col] = le.transform([val])[0] if val in le.classes_ else le.transform([le.classes_[0]])[0]
        else:
            encoded[col] = 0

    fitted_cols = scaler.feature_names_in_.tolist()
    row         = pd.DataFrame([encoded])[fitted_cols].astype(float)
    row_scaled  = scaler.transform(row)
    probs       = model.predict(row_scaled, verbose=0)[0]
    pred        = int(np.argmax(probs))

    labels  = {0: "NO Readmission", 1: "Readmitted < 30 days", 2: "Readmitted > 30 days"}
    classes = {0: "result-no", 1: "result-lt30", 2: "result-gt30"}
    icons   = {0: "✅", 1: "⚠️", 2: "🔶"}

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Output</div>', unsafe_allow_html=True)

    col_res, col_info = st.columns([1, 1])
    with col_res:
        st.markdown(
            f'<div class="result-box {classes[pred]}">'
            f'{icons[pred]} &nbsp; <strong>{labels[pred]}</strong><br>'
            f'<span style="font-size:0.78rem;opacity:0.75;margin-top:6px;display:block;">'
            f'NO: {probs[0]*100:.1f}%  |  &lt;30: {probs[1]*100:.1f}%  |  &gt;30: {probs[2]*100:.1f}%'
            f'</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="attr-box">'
            f'PATIENT &nbsp; <span>{patient_name}</span><br>'
            f'ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span>{patient_id}</span><br>'
            f'DATE &nbsp;&nbsp;&nbsp; <span>{record_date}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_info:
        fig_radar = go.Figure(go.Scatterpolar(
            r=[time_in_hospital, num_lab_procedures, num_procedures,
               num_medications, number_outpatient, number_emergency,
               number_inpatient, number_diagnoses],
            theta=["Days","Lab Proc","Procedures","Medications",
                   "Outpatient","Emergency","Inpatient","Diagnoses"],
            fill='toself', line_color='#7eb8f7',
            fillcolor='rgba(126,184,247,0.15)',
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, color="#4a7fa5"),
                angularaxis=dict(color="#4a7fa5"),
                bgcolor="#0d1a2e"
            ),
            paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e",
            font=dict(color="#8faecb", family="monospace", size=10),
            margin=dict(l=20, r=20, t=30, b=20), height=280
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
    bar_colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    marker_colors = [
        bar_colors[i] if i == pred
        else f"rgba({int(bar_colors[i][1:3],16)},{int(bar_colors[i][3:5],16)},{int(bar_colors[i][5:],16)},0.35)"
        for i in range(3)
    ]
    fig_bar = go.Figure(go.Bar(
        x=["No Readmission","< 30 Days","> 30 Days"],
        y=[p*100 for p in probs],
        marker=dict(color=marker_colors),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#dce8f5", family="monospace", size=12),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#111827",
        font=dict(color="#8faecb", family="monospace"),
        yaxis=dict(range=[0,110], gridcolor="#1e3a5f", title="Probability (%)",
                   tickfont=dict(color="#4a7fa5")),
        xaxis=dict(tickfont=dict(color="#dce8f5", size=12)),
        showlegend=False, margin=dict(l=40,r=40,t=30,b=40), height=300,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Key Clinical Indicators</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Time in Hospital", f"{time_in_hospital} days")
    col_b.metric("Lab Procedures",   f"{num_lab_procedures}")
    col_c.metric("Medications",       f"{num_medications}")
    col_d.metric("Diagnoses",         f"{number_diagnoses}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Medical Report</div>', unsafe_allow_html=True)

    pdf_buf = generate_pdf_report(raw, pred, probs, patient_name, patient_id)
    pdf_b64 = base64.b64encode(pdf_buf.read()).decode()
    safe_name = patient_name.replace(" ", "_")
    st.markdown(
        f'<a href="data:application/pdf;base64,{pdf_b64}" '
        f'download="readmission_report_{safe_name}.pdf">'
        f'<button style="background:linear-gradient(135deg,#1a4a7a,#2a7abf);color:white;'
        f'font-family:monospace;font-size:0.82rem;letter-spacing:2px;border:none;'
        f'border-radius:6px;padding:0.55rem 2rem;cursor:pointer;">'
        f'⬇ DOWNLOAD PDF REPORT</button></a>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-text" style="margin-top:0.5rem;">'
        'Full report with all patient attributes, prediction result, and probability breakdown.</p>',
        unsafe_allow_html=True
    )
