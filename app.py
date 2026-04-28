import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="HBV-HCC Prediction",
    layout="wide"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
body {
    background-color: #0b1a2a;
    color: white;
}
.stButton>button {
    background-color: #00c2cb;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = joblib.load('model.pkl')

# -------------------- TITLE --------------------
st.title("🧬 HBV-Associated HCC Early Prediction App")

st.markdown("""
This tool predicts **Hepatocellular Carcinoma (HCC)** risk  
based on **HBV mutation patterns and genotype**.

⚠ *Research prototype — not for clinical diagnosis*
""")

# -------------------- INSTRUCTIONS --------------------
with st.expander("ℹ How to use this tool"):
    st.write("""
    - Select mutation status:
        - Absent = mutation not present
        - Present = mutation detected
    - Select genotype (A–F)
    - Click Predict

    Output:
    - Low risk → unlikely HCC
    - High risk → possible HCC
    """)

# -------------------- INPUT SECTION --------------------
st.subheader("🧪 Molecular Input")

col1, col2 = st.columns(2)

def mutation_input(label, col):
    value = col.selectbox(label, ["Absent (0)", "Present (1)"])
    return 1 if value == "Present (1)" else 0

A1762T = mutation_input("A1762T", col1)
G1764A = mutation_input("G1764A", col1)
G1896A = mutation_input("G1896A", col1)

G1899A = mutation_input("G1899A", col2)
C1653T = mutation_input("C1653T", col2)
T1753V = mutation_input("T1753V", col2)

# Genotype
genotype = st.selectbox("Select Genotype", ["A","B","C","D","E","F"])
genotype_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
genotype = genotype_map[genotype]

# -------------------- PREDICTION --------------------
if st.button("🔬 Run Prediction"):

    input_data = np.array([[A1762T, G1764A, G1896A, G1899A, C1653T, T1753V, genotype]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of HCC\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✔ Low Risk\n\nProbability: {probability:.2f}")

    # Interpretation
    st.markdown("### 🧠 Interpretation")
    st.write("""
    The prediction is based on learned patterns between HBV mutations and HCC occurrence.
    A higher probability indicates stronger association with HCC in the dataset.
    """)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Developed for academic research purposes")
