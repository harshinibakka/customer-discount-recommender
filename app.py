
import streamlit as st

st.set_page_config(
    page_title="Customer Discount Recommender",
    page_icon="🛍️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #315c8c);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #e4edf8;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛍️ Customer Discount Recommender</h1>
    <p>Understand shopping behaviour and generate
    machine-learning discount recommendations.</p>
</div>
""", unsafe_allow_html=True)

st.caption("MACHINE LEARNING • RANDOM FOREST • STREAMLIT")

left, right = st.columns([2, 1])

with left:
    st.subheader("Project overview")
    st.write(
        "This application is designed to predict whether a "
        "customer should receive a discount using shopping-session "
        "features and a trained Random Forest classifier."
    )

with right:
    st.subheader("Model information")
    st.metric("Training sessions", "800")
    st.metric("Test sessions", "200")
    st.metric("Initial test accuracy", "81.5%")

st.divider()

st.subheader("Customer prediction")
st.write(
    "The customer input form and live model prediction "
    "will be connected in the next step."
)

st.info(
    "This project predicts discount recommendations. "
    "It does not directly predict actual cart abandonment."
)

st.success("Dashboard interface loaded.")
