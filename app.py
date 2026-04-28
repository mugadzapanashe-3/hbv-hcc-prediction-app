
import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="HBV-HCC Prediction App",
    layout="centered"
)

# -------------------- CUSTOM STYLE --------------------
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL --------------------
model = joblib.load('model.pkl')

# -------------------- TITLE --------------------
st.title("🧬 HBV-Associated HCC Early Prediction App")

st.markdown("""
### About this tool
This application predicts the risk of Hepatocellular Carcinoma (HCC) 
based on Hepatitis B Virus (HBV) mutation patterns.

⚠ **Disclaimer:** This is a research prototype and not for clinical use.
""")

# -------------------- INSTRUCTIONS --------------------
st.markdown("""
### How to use:

- Select mutation status:
  - **Absent (0)** = mutation not present
  - **Present (1)** = mutation detected

- Select genotype:
  - A–F correspond to HBV genotypes

- Click **Predict**

### Output:

- ✔ **Low risk** → No HCC detected  
- ⚠ **High risk** → Possible HCC
""")

st.info("Tip: If unsure, leave mutations as Absent (0).")

# -------------------- INPUT SECTION --------------------
st.subheader("🧪 Enter Molecular Data")

def mutation_input(label):
    value = st.selectbox(label, ["Absent (0)", "Present (1)"])
    return 1 if value == "Present (1)" else 0

A1762T = mutation_input("A1762T Mutation")
G1764A = mutation_input("G1764A Mutation")
G1896A = mutation_input("G1896A Mutation")
G1899A = mutation_input("G1899A Mutation")
C1653T = mutation_input("C1653T Mutation")
T1753V = mutation_input("T1753V Mutation")

# -------------------- GENOTYPE --------------------
genotype = st.selectbox("Select Genotype", ["A", "B", "C", "D", "E", "F"])
genotype_map = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6}
genotype = genotype_map[genotype]

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict"):

    input_data = np.array([[A1762T, G1764A, G1896A, G1899A, C1653T, T1753V, genotype]])

    prediction = model.predict(input_data)[0]

    st.subheader("🧾 Result")

    if prediction == 1:
        st.error("⚠ High Risk of HCC Detected")
    else:
        st.success("✔ Low Risk (No HCC Detected)")
