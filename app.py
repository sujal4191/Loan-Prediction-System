
import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# CSS
# =========================
st.markdown("""
<style>

.stApp{
    background:#0E1117;
    color:white;
}

h1,h2,h3{
    color:white;
}

section[data-testid="stSidebar"]{
    background:#161B22;
}

section[data-testid="stSidebar"] label{
    color:white !important;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    background:#1f6feb;
    color:white;
    border-radius:10px;
    border:none;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#388bfd;
}

div[data-testid="metric-container"]{
    background:#1b2430;
    border-radius:12px;
    padding:15px;
    border:1px solid #2f81f7;
}

div[data-testid="metric-container"] label{
    color:#c9d1d9 !important;
}

div[data-testid="metric-container"] div{
    color:white !important;
}

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🏦 Loan Prediction System")

st.markdown(
    "Predict whether a loan application is likely to be **Approved** or **Rejected** using Machine Learning."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Applicant Details")

person_income = st.sidebar.number_input(
    "💰 Annual Income",
    min_value=0.0,
    value=50000.0
)

credit_score = st.sidebar.number_input(
    "💳 Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

loan_amnt = st.sidebar.number_input(
    "🏦 Loan Amount",
    min_value=1000.0,
    value=10000.0
)

person_age = st.sidebar.number_input(
    "👤 Age",
    18,
    100,
    25
)

person_gender = st.sidebar.selectbox(
    "🚻 Gender",
    encoders["person_gender"].classes_
)

person_emp_exp = st.sidebar.number_input(
    "👔 Employment Experience (Years)",
    min_value=0,
    value=2
)

st.sidebar.markdown("---")

st.sidebar.info("""
**Model:** Random Forest Classifier

**Framework:** Streamlit

**Developer:** Sujal Gupta
""")

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Loan Status"):

    input_data = pd.DataFrame([{
        "person_income": person_income,
        "credit_score": credit_score,
        "loan_amnt": loan_amnt,
        "person_age": person_age,
        "person_gender": encoders["person_gender"].transform([person_gender])[0],
        "person_emp_exp": person_emp_exp
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)

    approve = probability[0][0]

    if prediction == 0:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

    st.divider()

    st.subheader("📋 Applicant Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("💰 Income", f"₹{person_income:,.0f}")
    c2.metric("💳 Credit Score", credit_score)
    c3.metric("🏦 Loan Amount", f"₹{loan_amnt:,.0f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("👤 Age", person_age)
    c5.metric("👔 Experience", f"{person_emp_exp} Years")
    c6.metric("🚻 Gender", person_gender)

    st.divider()

    st.subheader("📈 Prediction Confidence")

    st.progress(float(approve))

    st.write(f"### ✅ Approval Chance : {approve*100:.2f}%")
    st.write(f"### ❌ Rejection Chance : {probability[0][1]*100:.2f}%")

    st.divider()

    st.subheader("⚠ Risk Level")

    if approve >= 0.80:
        st.success("🟢 Low Risk Applicant")

    elif approve >= 0.50:
        st.warning("🟡 Medium Risk Applicant")

    else:
        st.error("🔴 High Risk Applicant")

    st.divider()

    st.subheader("📌 Recommendation")

    if prediction == 0:
        st.success("""
✔ Good credit score

✔ Stable income

✔ Good employment history

✔ Eligible for loan approval
""")

    else:
        st.error("""
• Improve your credit score

• Reduce loan amount

• Increase employment experience

• Apply again after improving financial profile
""")
        
       