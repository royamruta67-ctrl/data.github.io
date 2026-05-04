import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Page settings
st.set_page_config(page_title="AI Return Predictor", page_icon="🛒", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛒 Product Return Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered system to predict product returns</p>", unsafe_allow_html=True)

st.divider()

# Input Section
st.subheader("📥 Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("💰 Product Price", min_value=0)
    discount = st.slider("🏷️ Discount (%)", 0, 100)

with col2:
    age = st.number_input("👤 Customer Age", 18, 80)
    shipping = st.selectbox("🚚 Shipping Method", ["Standard", "Express"])

st.divider()

# Prepare input dictionary
input_dict = {col: 0 for col in columns}

# Fill values (MATCH TRAINING DATA)
if "Product_Price" in input_dict:
    input_dict["Product_Price"] = price

if "Discount" in input_dict:
    input_dict["Discount"] = discount

if "Age" in input_dict:
    input_dict["Age"] = age

if "Shipping_Method_Express" in input_dict:
    input_dict["Shipping_Method_Express"] = 1 if shipping == "Express" else 0

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("🔍 Predict Return"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High chance of return ({round(prob*100,2)}%)")
    else:
        st.success(f"✅ Low chance of return ({round(prob*100,2)}%)")

    st.progress(int(prob * 100))
