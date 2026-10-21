import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Cart Abandonment & Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "discount_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #f5f7fa;
}
.hero {
    background: linear-gradient(120deg, #172554, #2563a6);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    margin: 0;
}
.hero p {
    color: #e0e7ff;
    margin-bottom: 0;
}
.result-card {
    padding: 22px;
    border-radius: 12px;
    background: #192331;
    border: 1px solid #334155;
    margin: 8px 0;
}
.small-note {
    color: #aab6c5;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)
    return model, feature_columns


st.markdown("""
<div class="hero">
    <h1>🛒 Cart Abandonment & Purchase Prediction</h1>
    <p>Random Forest Machine Learning • Purchase intention analysis • Targeted discount support</p>
</div>
""", unsafe_allow_html=True)

st.caption("CASE STUDY | E-COMMERCE CUSTOMER CONVERSION")

st.info(
    "This model predicts whether a shopping session is likely to result "
    "in a purchase. A no-purchase prediction indicates potential "
    "non-conversion; it does not confirm that a cart was abandoned."
)

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error(
        "Model files are missing. Check that discount_model.pkl and "
        "feature_columns.json are in the same GitHub repository as app.py."
    )
    st.stop()

try:
    model, feature_columns = load_artifacts()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check the model file and make sure "
        "requirements.txt uses the same scikit-learn version used in training."
    )
    st.code(str(exc))
    st.stop()

# --------------------------------------------------
# INPUT DEFINITIONS
# These match the UCI dataset's feature names.
# --------------------------------------------------
numeric_inputs = {
    "Administrative": ("Administrative pages visited", 0, 0, 20),
    "Administrative_Duration": ("Administrative time (seconds)", 0.0, 0.0, 5000.0),
    "Informational": ("Informational pages visited", 0, 0, 20),
    "Informational_Duration": ("Informational time (seconds)", 0.0, 0.0, 5000.0),
    "ProductRelated": ("Product-related pages visited", 5, 0, 500),
    "ProductRelated_Duration": ("Product-related time (seconds)", 100.0, 0.0, 50000.0),
    "BounceRates": ("Bounce rate", 0.02, 0.0, 1.0),
    "ExitRates": ("Exit rate", 0.05, 0.0, 1.0),
    "PageValues": ("Page value", 0.0, 0.0, 10000.0),
    "SpecialDay": ("Special-day proximity (0 to 1)", 0.0, 0.0, 1.0),
    "OperatingSystems": ("Operating system code", 2, 1, 8),
    "Browser": ("Browser code", 2, 1, 13),
    "Region": ("Region code", 1, 1, 9),
    "TrafficType": ("Traffic source code", 2, 1, 20),
    "Weekend": ("Weekend session", False, None, None),
}

categorical_inputs = {
    "Month": (
        "Month",
        [
            "Feb", "Mar", "May", "June", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
    ),
    "VisitorType": (
        "Visitor type",
        ["Returning_Visitor", "New_Visitor", "Other"]
    ),
}

st.header("1. Shopping session details")
st.write("Enter the session information, then select Predict.")

input_values = {}

with st.form("prediction_form"):
    left, right = st.columns(2)

    numeric_items = list(numeric_inputs.items())
    split_at = (len(numeric_items) + 1) // 2

    for col, items in zip(
        [left, right],
        [numeric_items[:split_at], numeric_items[split_at:]]
    ):
        with col:
            for feature, config in items:
                label, default, minimum, maximum = config

                if feature == "Weekend":
                    input_values[feature] = st.selectbox(
                        label,
                        options=[False, True],
                        format_func=lambda value: "Yes" if value else "No"
                    )
                elif isinstance(default, int):
                    input_values[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1,
                        key=feature
                    )
                else:
                    input_values[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=0.01,
                        key=feature
                    )

    cat_left, cat_right = st.columns(2)
    for col, feature in zip([cat_left, cat_right], categorical_inputs):
        with col:
            label, options = categorical_inputs[feature]
            input_values[feature] = st.selectbox(label, options)

    submitted = st.form_submit_button(
        "🔍 Predict Purchase Probability",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame([input_values])

    # Ensure exact training feature order
    missing = [c for c in feature_columns if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in feature_columns]

    if missing or extra:
        st.error(
            "Input columns do not match the trained model. "
            f"Missing: {missing}; unexpected: {extra}"
        )
        st.stop()

    input_df = input_df[feature_columns]

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        purchase_index = classes.index(1)
        purchase_probability = float(probabilities[purchase_index])
        no_purchase_probability = 1.0 - purchase_probability

        # The model's default prediction is reported separately from risk.
        predicted_class = int(model.predict(input_df)[0])

    except Exception as exc:
        st.error("Prediction failed.")
        st.code(str(exc))
        st.stop()

    st.header("2. Prediction results")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric(
        "Purchase probability",
        f"{purchase_probability:.1%}"
    )
    metric2.metric(
        "No-purchase probability",
        f"{no_purchase_probability:.1%}"
    )
    metric3.metric(
        "Model prediction",
        "Purchase" if predicted_class == 1 else "No purchase"
    )

    st.progress(purchase_probability)

    if predicted_class == 1:
        st.success(
            "The model predicts that this session is more likely to result "
            "in a purchase."
        )
        st.subheader("Discount recommendation")
        st.write(
            "Avoid offering an automatic discount solely on this prediction. "
            "Consider normal marketing rules and the cost of incentives."
        )
    else:
        st.warning(
            "The model predicts that this session is more likely not to "
            "result in a purchase."
        )
        st.subheader("Discount recommendation")
        st.write(
            "Consider a targeted reminder or a limited discount experiment "
            "for eligible shoppers. A no-purchase prediction alone does not "
            "prove that a discount will change the outcome."
        )

    st.subheader("3. Business interpretation")
    st.write(
        f"The model estimates a {purchase_probability:.1%} probability "
        "of purchase for the entered session. This is a model estimate, "
        "not a guarantee."
    )

    st.caption(
        "Dataset: UCI Online Shoppers Purchasing Intention. "
        "The model was evaluated on held-out sessions. Performance on new "
        "customers may differ, and predictions should be validated before "
        "using them to make real discount decisions."
    )

with st.expander("About this project"):
    st.markdown("""
    **Objective:** Predict purchase intention from online shopping-session
    features using a Random Forest classifier.

    **Business use:** Help e-commerce teams identify sessions with lower
    predicted purchase probability and test targeted interventions.

    **Limitation:** The target records whether a purchase occurred during
    a session. It does not identify confirmed cart abandonment or prove
    that discounts cause more purchases.
    """)
