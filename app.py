
import streamlit as st

st.set_page_config(
    page_title="Customer Discount Recommender",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Discount Recommender")
st.write(
    "An AI-powered dashboard that predicts whether a customer "
    "should receive a discount using a Random Forest model."
)

st.info(
    "This application predicts discount recommendations, "
    "not actual cart abandonment."
)

st.subheader("Customer Session")
st.write(
    "Our customer input form and trained model will be connected "
    "in the next steps."
)

st.success("Welcome! Your dashboard is being built.")
