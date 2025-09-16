# streamlit run app.py
# ----------------------------------------------------------------------
# Diabetes Risk — 20-Year Canvas (with age_bin categorical)
#  • Loads model: diabetes_final_model.pkl (trained with cat_features=['age_bin','BMI_Class'])
#  • Reads a single-row person profile from Excel (no manual base entry)
#  • Shows a continuous AGE-ONLY baseline (yellow) from 0→20 years
#       - Only aging progresses: +1 age code (~1 age_bin) every 5 years, capped at '80+'
#       - Keeps 'Age' (1..13) and 'age_bin' consistent
#  • Rolling 3-year “Cases”: on each click, append the next 3 years with your settings;
#    earlier cases remain frozen
#  • VARIABLE PATH is ONE connected line with TWO colors only:
#       - GREEN  = ≤ baseline
#       - RED    =  > baseline
#  • Smooth animation reveals the variable path
#  • Fixed, nicely formatted **Baseline Profile** (original uploaded person)
#  • You can override the **starting age_bin** (resets timeline & recomputes baseline)
#  • Percent metrics & quick legend outside the chart
#  • Case counter increments each time you add a 3-year segment
#  • Brand logo in the top-left: “byte+”
#       - “+”  -> #fcd34d
#       - “y”  -> #5271ff
#       - other letters black
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from catboost import Pool
import plotly.graph_objects as go

st.set_page_config(page_title="byte+ Diabetes Prediction Over 20 Years", layout="wide")

# ===============================
# Top-left “byte+” brand (always visible)
# ===============================
BRAND_HTML = """
<div style="position:relative; top:-20px; left:10px;
            font-weight:800; font-size:28px; line-height:1; letter-spacing:.5px;">
  b<span style="color:#5271ff">y</span>te<span style="color:#fcd34d">+</span>
</div>
"""


st.markdown(BRAND_HTML, unsafe_allow_html=True)

# ===============================
# Load trained model
# ===============================
@st.cache_resource(show_spinner=False)
def load_model():
    # Must match your training artifact name
    return joblib.load("diabetes_final_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model file: {e}")
    st.stop()

TRAIN_FEATURES = list(model.feature_names_)              # exact model inputs that CatBoost expects
CAT_COLS_MODEL  = [c for c in ["age_bin", "BMI_Class"] if c in TRAIN_FEATURES]
CAT_IDX = [TRAIN_FEATURES.index(c) for c in CAT_COLS_MODEL]

# ===============================
# Constants & Mappings
# ===============================
TOTAL_YEARS   = 20
SEGMENT_YEARS = 3

AGE_LABELS = [
    '18-24','25-29','30-34','35-39','40-44','45-49',
    '50-54','55-59','60-64','65-69','70-74','75-79','80+'
]  # indices 0..12 map to codes 1..13

# BMI classes (same as training)
BMI_CAT_ORDER = [
    ("Underweight",     0.0, 18.5, 18.0),
    ("Normal",         18.5, 24.9, 22.0),
    ("Overweight",     25.0, 29.9, 27.5),
    ("Obesity Class 1", 30.0, 34.9, 32.5),
    ("Obesity Class 2", 35.0, 39.9, 37.5),
    ("Obesity Class 3", 40.0, 100.0, 42.0),
]
GEN_MAP_L2N = {"Excellent":1,"Very Good":2,"Good":3,"Fair":4,"Poor":5}
GEN_MAP_N2L = {v:k for k,v in GEN_MAP_L2N.items()}

# Colors
YELLOW = "#F5E663"
GREEN  = "#22C55E"  # better or equal vs baseline
RED    = "#EF4444"  # worse vs baseline

# ===============================
# Helpers
# ===============================
def bmi_to_class(bmi: float) -> str:
    for name, lo, hi, _ in BMI_CAT_ORDER:
        if lo <= bmi < hi:
            return name
    return "Obesity Class 3"

def bmi_class_repr(name: str) -> float:
    for n, _lo, _hi, rep in BMI_CAT_ORDER:
        if n == name:
            return rep
    return 42.0

def ment_phys_cat_from_days(days: int) -> int:
    if days == 0: return 0
    if 1 <= days <= 5: return 1
    if 6 <= days <= 13: return 2
    return 3

def days_from_cat(cat: int) -> int:
    # used for UI <-> numeric days
    return {0: 0, 1: 3, 2: 10, 3: 20}.get(int(cat), 0)

def age_code_from_age_bin_label(lbl: str) -> int:
    # 1..13
    try:
        return AGE_LABELS.index(lbl) + 1
    except Exception:
        return 7  # default ~ '50-54'

def age_bin_label_from_code(code: int) -> str:
    # label from 1..13 code
    code = int(np.clip(int(code), 1, 13))
    return AGE_LABELS[code - 1]

def age_advance_over_years(start_age_code: int, elapsed_years: int) -> int:
    """Advance 1 age code (~1 bin) per 5 years, cap at 13."""
    new_code = int(start_age_code) + int(elapsed_years // 5)
    return int(np.clip(new_code, 1, 13))

def predict_prob(row: dict) -> float:
    """Predict % probability using the trained model with correct cat features (age_bin, BMI_Class)."""
    df = pd.DataFrame([row])

    # Ensure required features exist
    for c in TRAIN_FEATURES:
        if c not in df.columns:
            if c == "BMI":
                df[c] = 25.0
            elif c in ("HighBP","HighChol","CholCheck","Smoker","HeartDiseaseorAttack",
                       "PhysActivity","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","MentHlth_Cat"):
                df[c] = 0
            elif c in ("age_bin","BMI_Class"):
                df[c] = "Unknown"
            else:
                df[c] = 0

    # Drop any extra columns not used by model
    extra = [c for c in df.columns if c not in TRAIN_FEATURES]
    if extra:
        df = df.drop(columns=extra)

    # Dtypes for categoricals
    for c in CAT_COLS_MODEL:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Numeric columns: everything else except categoricals
    num_cols = [c for c in TRAIN_FEATURES if c not in CAT_COLS_MODEL]
    for c in num_cols:
        if c == "BMI":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(25.0).astype(float)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Reorder exactly as model expects
    df = df[TRAIN_FEATURES]

    # VERY IMPORTANT: pass cat_features indices so CatBoost treats strings as categories
    pool = Pool(df, cat_features=CAT_IDX)
    return float(model.predict_proba(pool)[0][1]) * 100.0

def build_feature_row(raw_row: pd.Series) -> dict:
    """Normalize a single uploaded row into a dict matching the model's features (with safe extras for display)."""
    r = raw_row.fillna(0)
    out = {}

    # binary flags
    for b in ['HighBP','HighChol','CholCheck','Smoker','HeartDiseaseorAttack','PhysActivity','DiffWalk','Sex']:
        if b in r: out[b] = int(pd.to_numeric(r[b], errors='coerce'))

    # numeric basics
    if 'BMI' in r:
        out['BMI'] = float(pd.to_numeric(r['BMI'], errors='coerce'))
    else:
        out['BMI'] = 25.0

    if 'GenHlth' in r:
        out['GenHlth'] = int(pd.to_numeric(r['GenHlth'], errors='coerce'))
    else:
        out['GenHlth'] = 3

    if 'MentHlth' in r:
        out['MentHlth'] = int(pd.to_numeric(r['MentHlth'], errors='coerce'))
    else:
        out['MentHlth'] = 0

    if 'PhysHlth' in r:
        out['PhysHlth'] = int(pd.to_numeric(r['PhysHlth'], errors='coerce'))
    else:
        out['PhysHlth'] = 0

    # engineered fields
    out['BMI_Class'] = bmi_to_class(out['BMI'])
    out['MentHlth_Cat'] = ment_phys_cat_from_days(out['MentHlth'])

    # Age / age_bin reconciliation:
    # If file has age_bin string → use it; else if Age code (1..13) → derive label; else default '50-54'
    if 'age_bin' in r and str(r['age_bin']) != "":
        age_label = str(r['age_bin'])
        if age_label not in AGE_LABELS:
            if 'Age' in r:
                age_label = age_bin_label_from_code(int(pd.to_numeric(r['Age'], errors='coerce')))
            else:
                age_label = '50-54'
        out['age_bin'] = age_label
        out['Age']     = age_code_from_age_bin_label(age_label)
    elif 'Age' in r:
        age_code = int(pd.to_numeric(r['Age'], errors='coerce'))
        out['Age']     = int(np.clip(age_code, 1, 13))
        out['age_bin'] = age_bin_label_from_code(out['Age'])
    else:
        out['Age']     = 7
        out['age_bin'] = '50-54'

    # Ensure all model features exist
    for c in TRAIN_FEATURES:
        if c not in out:
            out[c] = ("Unknown" if c in ("age_bin","BMI_Class")
                      else (25.0 if c == "BMI" else 0))

    # Extras for display (not sent to model)
    out['_sex_label'] = "Male" if int(out.get('Sex', 0)) == 1 else "Female"
    return out

def prettify_baseline_display(b: dict) -> pd.DataFrame:
    """Return a small, nice-looking dataframe for the fixed 'Baseline Profile' card."""
    yesno = lambda v: "Yes" if int(v) == 1 else "No"
    mh    = {0:"No distress (0d)",1:"Mild (1–5d)",2:"Moderate (6–13d)",3:"Severe (14+d)"}
    gh    = GEN_MAP_N2L.get(int(b.get("GenHlth",3)),"Good")

    rows = [
        ("Age (code 1–13)",  int(b.get("Age",7))),
        ("Age bin",          b.get("age_bin","50-54")),
        ("Sex",              "Male" if int(b.get("Sex",0)) == 1 else "Female"),
        ("BMI",              f"{float(b.get('BMI',25.0)):.1f}"),
        ("BMI class",        b.get("BMI_Class", bmi_to_class(float(b.get("BMI",25.0))))),
        ("High blood pressure",  yesno(b.get("HighBP",0))),
        ("High cholesterol",     yesno(b.get("HighChol",0))),
        ("Cholesterol check",    yesno(b.get("CholCheck",1))),
        ("Smoker (≥100 cig)",    yesno(b.get("Smoker",0))),
        ("Physical activity",    yesno(b.get("PhysActivity",1))),
        ("Heart disease / MI",   yesno(b.get("HeartDiseaseorAttack",0))),
        ("Difficulty walking",   yesno(b.get("DiffWalk",0))),
        ("General health",       gh),
        ("Mental health",        mh.get(int(b.get("MentHlth_Cat",0)), "No distress")),
        ("Physical health days", int(b.get("PhysHlth",0))),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])

def series_age_only_full(baseline_row: dict):
    """Baseline series 0..20y with only aging; all other params fixed."""
    years = list(range(0, TOTAL_YEARS + 1))
    probs = []
    base_age_code = int(baseline_row['Age'])
    for y in years:
        r = dict(baseline_row)
        new_age_code = age_advance_over_years(base_age_code, y)
        r['Age'] = new_age_code
        r['age_bin'] = age_bin_label_from_code(new_age_code)
        r['BMI_Class'] = bmi_to_class(r.get('BMI', baseline_row.get('BMI', 25.0)))
        probs.append(predict_prob(r))
    return years, probs

# ===============================
# Session state
# ===============================
if 'segments' not in st.session_state: st.session_state.segments = []
if 'points' not in st.session_state: st.session_state.points = []
if 'current_profile' not in st.session_state: st.session_state.current_profile = None
if 'cum_years' not in st.session_state: st.session_state.cum_years = 0
if 'base_profile' not in st.session_state: st.session_state.base_profile = None
if 'baseline_display' not in st.session_state: st.session_state.baseline_display = None
if 'base_age_code' not in st.session_state: st.session_state.base_age_code = None
if 'case_counter' not in st.session_state: st.session_state.case_counter = 0
if 'age_only_series' not in st.session_state: st.session_state.age_only_series = None

# ===============================
# UI: Upload Excel & choose row
# ===============================
st.title("🧬 byte+ Diabetes Prediction Over 20 Years")

uploaded = st.file_uploader("📂 Upload Excel (.xlsx/.xls) — one row = one person", type=["xlsx","xls"])
if not uploaded:
    st.info("Upload a file to begin. The canvas will still show 0–20 years.")
else:
    try:
        df_upload = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

    if df_upload.empty:
        st.error("Uploaded sheet is empty."); st.stop()

    with st.expander("Select the person/row", expanded=True):
        if "id" in df_upload.columns:
            ids = df_upload["id"].astype(str).tolist()
            chosen_id = st.selectbox("Pick an ID", ids, index=0)
            base_series = df_upload[df_upload["id"].astype(str) == chosen_id].iloc[0]
        else:
            idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df_upload)-1, value=0, step=1)
            base_series = df_upload.iloc[int(idx)]

    base_row = build_feature_row(base_series)

    # Initialize on new upload/person switch
    signature = hash(str(base_series.to_dict()))
    if st.session_state.base_profile is None or st.session_state.base_profile.get("_hash") != signature:
        st.session_state.base_profile = dict(base_row); st.session_state.base_profile["_hash"] = signature
        st.session_state.current_profile = dict(base_row)
        st.session_state.segments = []
        st.session_state.points = []   # (year, prob)
        st.session_state.cum_years = 0
        st.session_state.case_counter = 0
        st.session_state.base_age_code = int(base_row['Age'])
        st.session_state.age_only_series = series_age_only_full(st.session_state.base_profile)

        # fixed, pretty baseline card/table (immutable)
        st.session_state.baseline_display = prettify_baseline_display(st.session_state.base_profile)

        # Add the very first point (year 0) to the continuous path
        y0 = predict_prob(st.session_state.current_profile)
        st.session_state.points.append((0, y0))

# ===============================
# Starting Age Override (age_bin)
# ===============================
if st.session_state.base_profile is not None:
    st.markdown("### 🧑‍🦳 Starting age bin override")
    with st.container(border=True):
        col_age, col_btn = st.columns([0.75, 0.25])
        with col_age:
            current_label = st.session_state.base_profile.get('age_bin', '50-54')
            if current_label not in AGE_LABELS:
                current_label = '50-54'
            new_age_label = st.selectbox(
                "Select the starting age bin (applies to the entire 0–20y horizon)",
                AGE_LABELS, index=AGE_LABELS.index(current_label), key="ui_start_age_bin"
            )
        with col_btn:
            if st.button("Apply starting age"):
                # Update baseline & current profiles
                new_code = age_code_from_age_bin_label(new_age_label)
                st.session_state.base_profile['Age'] = new_code
                st.session_state.base_profile['age_bin'] = new_age_label
                st.session_state.base_profile['BMI_Class'] = bmi_to_class(st.session_state.base_profile.get('BMI', 25.0))
                st.session_state.current_profile = dict(st.session_state.base_profile)
                # Reset progression/timeline state
                st.session_state.base_age_code = new_code
                st.session_state.age_only_series = series_age_only_full(st.session_state.base_profile)
                st.session_state.baseline_display = prettify_baseline_display(st.session_state.base_profile)
                st.session_state.segments = []
                st.session_state.points = []
                st.session_state.cum_years = 0
                st.session_state.case_counter = 0
                st.session_state.points.append((0, predict_prob(st.session_state.current_profile)))
                st.success("Starting age bin updated and timeline reset.")

# ===============================
# Fixed Baseline Profile (never changes)
# ===============================
if st.session_state.baseline_display is not None:
    st.markdown("### 🧾 Baseline Profile (from uploaded file)")
    colA, colB = st.columns([0.55, 0.45])
    with colA:
        st.dataframe(
            st.session_state.baseline_display.style.hide(axis='index'),
            use_container_width=True
        )
    with colB:
        b = st.session_state.base_profile
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Age bin", b.get("age_bin","50-54"))
        with mcol2:
            st.metric("BMI", f"{float(b.get('BMI',25.0)):.1f}")
        with mcol3:
            st.metric("BMI class", b.get("BMI_Class", bmi_to_class(float(b.get("BMI",25.0)))))
        r_now = predict_prob(b)
        st.markdown("#### Estimated current risk")
        st.markdown(f"<div style='font-size:42px;font-weight:700;color:#0d0c08;'>{r_now:.1f}%</div>", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# Controls for NEXT case (3 years)
# ===============================
st.markdown("### 🎛️ Configure the NEXT 3-year case (applies from current year forward)")
if st.session_state.current_profile is None:
    st.info("Upload and select a person to enable controls.")
    ui_overrides = {}
else:
    cp = st.session_state.current_profile
    c1, c2, c3 = st.columns(3)
    with c1:
        Sex = st.selectbox("Sex", ["Female","Male"], index=cp['Sex'], key="ui_sex")
        HighBP = st.selectbox("High Blood Pressure", ["No","Yes"], index=cp['HighBP'], key="ui_hbp")
        HighChol = st.selectbox("High Cholesterol", ["No","Yes"], index=cp['HighChol'], key="ui_hchol")
        CholCheck = st.selectbox("Cholesterol Check (past 5y)", ["No","Yes"], index=cp['CholCheck'], key="ui_cholcheck")
        Smoker = st.selectbox("Smoker (≥100 cig lifetime)", ["No","Yes"], index=cp['Smoker'], key="ui_smoker")
    with c2:
        HeartDiseaseorAttack = st.selectbox("Heart disease / attack history", ["No","Yes"], index=cp['HeartDiseaseorAttack'], key="ui_hd")
        PhysActivity = st.selectbox("Physical Activity (past 30 days)", ["No","Yes"], index=cp['PhysActivity'], key="ui_pa")
        DiffWalk = st.selectbox("Serious difficulty walking", ["No","Yes"], index=cp['DiffWalk'], key="ui_diffwalk")
        GenHlth_label = st.selectbox("General Health", list(GEN_MAP_L2N.keys()),
                                     index=int(cp['GenHlth'])-1 if 1 <= int(cp['GenHlth']) <= 5 else 2, key="ui_genhlth")
    with c3:
        bmi_classes = [x[0] for x in BMI_CAT_ORDER]
        default_bmi_class = cp['BMI_Class'] if cp['BMI_Class'] in bmi_classes else bmi_to_class(cp.get('BMI', 25.0))
        BMI_Class = st.selectbox("BMI Class", bmi_classes,
                                 index=bmi_classes.index(default_bmi_class), key="ui_bmi_class")
        BMI_num_default = bmi_class_repr(BMI_Class) if BMI_Class != bmi_to_class(cp['BMI']) else cp['BMI']
        BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=float(BMI_num_default), step=0.1, key="ui_bmi")
        BMI_Class_auto = bmi_to_class(BMI)
        if BMI_Class_auto != BMI_Class:
            st.info(f"Adjusted BMI Class based on BMI={BMI:.1f}: **{BMI_Class_auto}**")
            BMI_Class = BMI_Class_auto

        MentHlth_Cat = st.selectbox("Mental Health (0 No → 3 Severe)", [0,1,2,3],
                                    index=int(cp['MentHlth_Cat']), key="ui_mhcat")
        PhysHlth_Cat = st.selectbox("Physical Health (0 No → 3 Severe)", [0,1,2,3],
                                    index=ment_phys_cat_from_days(int(cp['PhysHlth'])), key="ui_phcat")

    ui_overrides = {
        'Sex': 1 if Sex == "Male" else 0,
        'HighBP': 1 if HighBP == "Yes" else 0,
        'HighChol': 1 if HighChol == "Yes" else 0,
        'CholCheck': 1 if CholCheck == "Yes" else 0,
        'Smoker': 1 if Smoker == "Yes" else 0,
        'HeartDiseaseorAttack': 1 if HeartDiseaseorAttack == "Yes" else 0,
        'PhysActivity': 1 if PhysActivity == "Yes" else 0,
        'DiffWalk': 1 if DiffWalk == "Yes" else 0,
        'GenHlth': GEN_MAP_L2N[GenHlth_label],
        'BMI': float(BMI),
        'BMI_Class': BMI_Class,
        'MentHlth_Cat': int(MentHlth_Cat),
        'MentHlth': days_from_cat(int(MentHlth_Cat)),
        'PhysHlth': days_from_cat(int(PhysHlth_Cat)),
    }

# ===============================
# Build a 3-year case → append to continuous points
# ===============================
def build_case_points(start_year: int, base_profile_for_segment: dict, overrides: dict):
    """Return (years, probs, end_profile) for start_year..start_year+3 inclusive."""
    base_age_code = int(st.session_state.base_age_code)
    seg_profile = dict(base_profile_for_segment); seg_profile.update(overrides)

    years = [start_year + dy for dy in [0,1,2,3]]
    probs = []
    for dy in [0,1,2,3]:
        total_elapsed = start_year + dy
        r = dict(seg_profile)
        # Progress age by +1 code every 5 elapsed years
        new_age_code = age_advance_over_years(base_age_code, total_elapsed)
        r['Age'] = new_age_code
        r['age_bin'] = age_bin_label_from_code(new_age_code)
        r['BMI_Class'] = bmi_to_class(r.get('BMI', 25.0))
        probs.append(predict_prob(r))

    end_profile = dict(seg_profile)
    end_profile['Age'] = age_advance_over_years(base_age_code, start_year + 3)
    end_profile['age_bin'] = age_bin_label_from_code(end_profile['Age'])
    return years, probs, end_profile

# ===============================
# Buttons
# ===============================
btn1, btn2, info = st.columns([0.28, 0.22, 0.5])

with btn1:
    add_disabled = (st.session_state.current_profile is None) or (st.session_state.cum_years >= TOTAL_YEARS)
    if st.button("🚀 Add next 3-year case", type="primary", disabled=add_disabled):
        start_year = st.session_state.cum_years
        yrs, ys, end_prof = build_case_points(start_year, st.session_state.current_profile, ui_overrides)

        # Append to continuous points (avoid duplicating last year)
        for x, y in zip(yrs, ys):
            if st.session_state.points and x <= st.session_state.points[-1][0]:
                continue
            if x > TOTAL_YEARS:  # don't go past 20
                continue
            st.session_state.points.append((x, y))

        st.session_state.segments.append({"x": [v for v in yrs if v <= TOTAL_YEARS],
                                          "y": ys[:len([v for v in yrs if v <= TOTAL_YEARS])],
                                          "start": start_year,
                                          "end": min(start_year+3, TOTAL_YEARS)})
        st.session_state.current_profile = dict(end_prof)
        st.session_state.cum_years = min(TOTAL_YEARS, st.session_state.cum_years + SEGMENT_YEARS)
        st.session_state.case_counter += 1
        st.toast(f"✅ Added Case {st.session_state.case_counter}")

with btn2:
    if st.button("🔁 Reset"):
        st.session_state.segments = []
        st.session_state.points = []
        st.session_state.cum_years = 0
        st.session_state.case_counter = 0
        if st.session_state.base_profile is not None:
            st.session_state.current_profile = dict(st.session_state.base_profile)
            st.session_state.base_age_code = int(st.session_state.base_profile['Age'])
            st.session_state.age_only_series = series_age_only_full(st.session_state.base_profile)
            st.session_state.baseline_display = prettify_baseline_display(st.session_state.base_profile)
            st.session_state.points.append((0, predict_prob(st.session_state.current_profile)))
        st.success("Reset to baseline.")

with info:
    st.markdown(f"### 🧮 Cases added: **{st.session_state.case_counter}**")

# ===============================
# Build the figure
# ===============================
st.markdown("### 📊 20-Year Canvas")
fig = go.Figure()

# Vertical markers every 3 years up to 20
for k in range(3, TOTAL_YEARS + 1, 3):
    fig.add_shape(type="line", x0=k, x1=k, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="rgba(200,200,200,0.25)", width=1, dash="dot"))

# Age-only baseline (yellow)
if st.session_state.age_only_series is not None:
    x_base, y_base = st.session_state.age_only_series
    fig.add_trace(go.Scatter(
        x=x_base, y=y_base, mode="lines+markers",
        name="Age-only baseline", line=dict(color=YELLOW, width=4),
        marker=dict(size=7, color=YELLOW),
        hovertemplate="Year %{x}<br>Risk %{y:.1f}%<extra></extra>"
    ))
else:
    x_base, y_base = [0, TOTAL_YEARS], [0, 0]  # safety

def baseline_at_year(x_val: int) -> float:
    if st.session_state.age_only_series is None:
        return 0.0
    idx = int(np.clip(int(round(x_val)), 0, TOTAL_YEARS))
    return y_base[idx]

def split_segment_if_crosses(x0, y0, x1, y1):
    b0 = baseline_at_year(x0); b1 = baseline_at_year(x1)
    s0 = y0 - b0; s1 = y1 - b1
    if s0 == 0 or s1 == 0 or (s0 > 0 and s1 > 0) or (s0 < 0 and s1 < 0):
        return [(x0, y0, x1, y1)]
    # linear interpolation of difference to find crossing
    d0, d1 = s0, s1
    f = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.5
    xc = x0 + f * (x1 - x0)
    yc = y0 + f * (y1 - y0)
    return [(x0, y0, xc, yc), (xc, yc, x1, y1)]

# Connected two-color variable path
pts = sorted(st.session_state.points, key=lambda t: t[0])
if len(pts) >= 2:
    # Legend proxies for the two colors
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color=GREEN, width=6), name="Lower than baseline"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color=RED, width=6), name="Higher than baseline"))

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x1 < 0 or x0 > TOTAL_YEARS:  # out of window
            continue
        xa, ya = max(0, x0), y0
        xb, yb = min(TOTAL_YEARS, x1), y1

        for sx0, sy0, sx1, sy1 in split_segment_if_crosses(xa, ya, xb, yb):
            midx = (sx0 + sx1) / 2.0
            midy = sy0 + (sy1 - sy0) * 0.5
            col = GREEN if (midy - baseline_at_year(midx)) <= 0 else RED
            fig.add_trace(go.Scatter(
                x=[sx0, sx1], y=[sy0, sy1], mode="lines",
                line=dict(color=col, width=6, shape="linear"),
                showlegend=False, hoverinfo="skip"
            ))
    # markers
    for (x, y) in pts:
        col = GREEN if (y - baseline_at_year(x)) <= 0 else RED
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color=col, size=8), showlegend=False,
            hovertemplate="Year %{x}<br>Risk %{y:.1f}%<extra></extra>"
        ))

# Layout
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
    title=dict(text="<b>Connected variable path (Green = better, Red = worse)</b>",
               x=0.02, y=0.98, font=dict(size=22)),
    xaxis=dict(title="Years", tickmode="linear", dtick=1, range=[0, TOTAL_YEARS],
               showline=True, linecolor="rgba(255,255,255,0.15)",
               gridcolor="rgba(136,136,136,0.22)"),
    yaxis=dict(title="Predicted probability (%)", rangemode="tozero",
               showline=True, linecolor="rgba(255,255,255,0.15)",
               gridcolor="rgba(255,255,255,0.06)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
    margin=dict(l=60, r=40, t=70, b=60),
    hovermode="x unified",
)

# Animation frames (progressively reveal the path)
frames = []
if len(pts) >= 2:
    for i in range(1, len(pts)):
        sub_pts = pts[: i+1]
        seg_traces = []
        for (x0, y0), (x1, y1) in zip(sub_pts[:-1], sub_pts[1:]):
            for sx0, sy0, sx1, sy1 in split_segment_if_crosses(x0, y0, x1, y1):
                midx = (sx0 + sx1) / 2.0
                midy = sy0 + (sy1 - sy0) * 0.5
                col = GREEN if (midy - baseline_at_year(midx)) <= 0 else RED
                seg_traces.append(go.Scatter(
                    x=[sx0, sx1], y=[sy0, sy1], mode="lines",
                    line=dict(color=col, width=6), showlegend=False, hoverinfo="skip"
                ))
        frames.append(go.Frame(data=seg_traces, name=f"f{i}"))

fig.update_layout(
    updatemenus=[dict(
        type="buttons", direction="left", pad={"r": 10, "t": 10}, x=0.02, y=1.12,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": 450, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 250}}]),
            dict(label="⏹ Pause", method="animate",
                 args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0}}])
        ]
    )]
)
if frames:
    fig.frames = frames

st.plotly_chart(fig, use_container_width=True)

# ===============================
# Percent cards & quick legend (outside chart)
# ===============================
colL, colR = st.columns([0.55, 0.45])
with colL:
    if st.session_state.points:
        last_x, last_y = st.session_state.points[-1]
        st.metric("Last variable-path risk", f"{last_y:.1f}%")
    if st.session_state.age_only_series is not None:
        idx_now = st.session_state.cum_years if st.session_state.cum_years <= TOTAL_YEARS else TOTAL_YEARS
        base_y_now = st.session_state.age_only_series[1][idx_now]
        st.metric("Baseline risk at current horizon", f"{base_y_now:.1f}%")
with colR:
    st.markdown("**Line meanings:**")
    st.markdown("- 🟡 **Age-only baseline**: only your age bin changes with time; all other factors fixed.")
    st.markdown("- 🟢 **Variable path (better)**: lower/equal risk than baseline.")
    st.markdown("- 🔴 **Variable path (worse)**: higher risk than baseline.")

# ===============================
# Case summary table
# ===============================
if st.session_state.segments:
    st.markdown("### 📑 Case summary")
    rows = []
    for seg in st.session_state.segments:
        x, y = seg["x"], seg["y"]
        if not x:  # safety
            continue
        end_idx = min(len(x)-1, len(y)-1)
        end_x = x[end_idx]
        end_y = y[end_idx]
        base_end = st.session_state.age_only_series[1][int(min(end_x, TOTAL_YEARS))]
        rows.append({
            "Case window": f"{seg['start']}–{seg['end']}",
            "Start Risk (%)": round(y[0], 2),
            "End Risk (%)": round(end_y, 2),
            "Δ vs baseline at end (pp)": round(end_y - base_end, 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.caption(
    "Yellow = age-only baseline (original parameters fixed). The variable path is one connected curve: "
    "green when it’s better than baseline, red when worse. "
    "Upload a single-row Excel with all model features; you can override the starting age bin. "
    "This canvas spans 0–20 years. Model file: diabetes_final_model.pkl (cat_features=['age_bin','BMI_Class'])."
)
