import streamlit as st
import joblib
import numpy as np

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="LoanInsight",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Light styling (kept simple, no extra libraries needed)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 760px;
            padding-top: 2rem;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .app-subtitle {
            color: #9aa0a6;
            margin-bottom: 1.8rem;
        }
        .section-label {
            font-weight: 600;
            font-size: 1.05rem;
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
            border-bottom: 1px solid rgba(150,150,150,0.25);
            padding-bottom: 0.3rem;
        }
        div.stButton > button {
            width: 100%;
            padding: 0.6rem 0;
            font-weight: 600;
            font-size: 1.05rem;
            border-radius: 8px;
        }
        .result-approved {
            background-color: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 1rem 1.2rem;
            border-radius: 10px;
            font-size: 1.15rem;
            font-weight: 600;
            color: #10b981;
            text-align: center;
        }
        .result-rejected {
            background-color: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 1rem 1.2rem;
            border-radius: 10px;
            font-size: 1.15rem;
            font-weight: 600;
            color: #ef4444;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load trained model
# (encoders.pkl / status_encoder.pkl are no longer required — see
#  the manual mappings below, which avoid the "unseen labels" bug)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# ------------------------------------------------------------------
# Form reset support
#
# Each input below is keyed with this id. Bumping it and rerunning
# forces Streamlit to create fresh widgets (empty/default state),
# which is how the Reset button clears the form.
# ------------------------------------------------------------------
if "form_id" not in st.session_state:
    st.session_state.form_id = 0
fid = st.session_state.form_id

# ------------------------------------------------------------------
# Manual encoding maps
#
# These mirror sklearn's default LabelEncoder behaviour, which sorts
# classes alphabetically and assigns 0, 1, 2, ... in that order.
# Using fixed maps instead of the pickled encoders avoids the bug
# where reusing a single LabelEncoder instance across a training loop
# causes every column to end up encoded with the LAST column's
# classes (that's what produced "unseen labels: 'Male'").
#
# IMPORTANT: If your model was trained with a different encoding
# order, update the maps below to match exactly.
# ------------------------------------------------------------------
GENDER_MAP = {"Female": 0, "Male": 1}
MARRIED_MAP = {"No": 0, "Yes": 1}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<div class="app-title">🏦 LoanInsight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Smart, instant loan eligibility predictions based on 11 applicant factors.</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Form
# ------------------------------------------------------------------
with st.form("loan_form"):

    st.markdown('<div class="section-label">Applicant Name</div>', unsafe_allow_html=True)
    applicant_name = st.text_input(
        "Full Name", placeholder="Enter applicant's name", label_visibility="collapsed", key=f"name_{fid}"
    )

    st.markdown('<div class="section-label">Applicant Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()), index=None, placeholder="Select gender", key=f"gender_{fid}")
        education = st.selectbox("Education", list(EDUCATION_MAP.keys()), index=None, placeholder="Select education", key=f"education_{fid}")
        dependents = st.selectbox("Dependents", list(DEPENDENTS_MAP.keys()), index=None, placeholder="Select dependents", key=f"dependents_{fid}")
    with col2:
        married = st.selectbox("Marital Status", list(MARRIED_MAP.keys()), index=None, placeholder="Select marital status", key=f"married_{fid}")
        self_employed = st.selectbox("Self Employed", list(SELF_EMPLOYED_MAP.keys()), index=None, placeholder="Select an option", key=f"self_employed_{fid}")
        property_area = st.selectbox("Property Area", list(PROPERTY_AREA_MAP.keys()), index=None, placeholder="Select property area", key=f"property_area_{fid}")

    st.markdown('<div class="section-label">Financial Details</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        applicant_income = st.number_input(
            "Applicant Income (₹/month)", min_value=0, step=500, value=None, placeholder="Enter amount", key=f"applicant_income_{fid}"
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=0, step=10, value=None, placeholder="Enter amount", key=f"loan_amount_{fid}"
        )
    with col4:
        coapplicant_income = st.number_input(
            "Coapplicant Income (₹/month)", min_value=0, step=500, value=None, placeholder="Enter amount", key=f"coapplicant_income_{fid}"
        )
        loan_amount_term = st.number_input(
            "Loan Term (days)", min_value=0, step=30, value=None, placeholder="Enter term", key=f"loan_term_{fid}"
        )

    st.markdown('<div class="section-label">Credit History</div>', unsafe_allow_html=True)
    credit_history = st.radio(
        "Does the applicant have a good credit history?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        index=None,
        key=f"credit_history_{fid}",
    )

    col_predict, col_reset = st.columns([3, 1])
    with col_predict:
        submitted = st.form_submit_button("Predict Loan Approval")
    with col_reset:
        reset_clicked = st.form_submit_button("Reset")

if reset_clicked:
    st.session_state.form_id += 1
    st.rerun()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    missing = []
    if not applicant_name.strip():
        missing.append("Name")
    if gender is None:
        missing.append("Gender")
    if married is None:
        missing.append("Marital Status")
    if dependents is None:
        missing.append("Dependents")
    if education is None:
        missing.append("Education")
    if self_employed is None:
        missing.append("Self Employed")
    if property_area is None:
        missing.append("Property Area")
    if applicant_income is None:
        missing.append("Applicant Income")
    if coapplicant_income is None:
        missing.append("Coapplicant Income")
    if loan_amount is None:
        missing.append("Loan Amount")
    if loan_amount_term is None:
        missing.append("Loan Term")
    if credit_history is None:
        missing.append("Credit History")

    if missing:
        st.warning(f"Please fill in: {', '.join(missing)}")
        st.stop()

    try:
        input_data = np.array([[
            GENDER_MAP[gender],
            MARRIED_MAP[married],
            DEPENDENTS_MAP[dependents],
            EDUCATION_MAP[education],
            SELF_EMPLOYED_MAP[self_employed],
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            PROPERTY_AREA_MAP[property_area],
        ]])

        prediction = model.predict(input_data)[0]
        # Handle both string ("Y"/"N") and numeric (1/0) model outputs
        approved = prediction in ("Y", 1, "1", True)

        # Confidence score, shown only if the model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = round(max(proba) * 100, 1)

        st.markdown("---")
        if approved:
            st.markdown(
                f'<div class="result-approved">✅ {applicant_name}, the loan is Approved</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-rejected">❌ {applicant_name}, the loan is Not Approved</div>',
                unsafe_allow_html=True,
            )

        if confidence is not None:
            st.caption(f"Model confidence: {confidence}%")

        total_income = applicant_income + coapplicant_income
        with st.expander("View submitted details"):
            st.write(f"**Name:** {applicant_name}")
            st.write(f"**Gender:** {gender}  |  **Marital Status:** {married}  |  **Dependents:** {dependents}")
            st.write(f"**Education:** {education}  |  **Self Employed:** {self_employed}")
            st.write(f"**Applicant Income:** ₹{applicant_income:,.0f}  |  **Coapplicant Income:** ₹{coapplicant_income:,.0f}")
            st.write(f"**Total Household Income:** ₹{total_income:,.0f}")
            st.write(f"**Loan Amount:** ₹{loan_amount:,.0f} thousand  |  **Loan Term:** {loan_amount_term:.0f} days")
            st.write(f"**Credit History:** {'Good' if credit_history == 1 else 'Poor/None'}  |  **Property Area:** {property_area}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "If this persists, double-check that the feature order and encoding "
            "used here match exactly what the model was trained on."
        )