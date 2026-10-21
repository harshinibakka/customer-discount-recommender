import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Discount Recommender",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("discount_model.pkl")
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, features

st.title("🛍️ Customer Discount Recommender")
st.write(
    "Predict whether a customer should receive a discount "
    "based on their shopping-session information."
)

st.info(
    "This model predicts discount recommendations, "
    "not actual cart abandonment."
)

try:
    model, feature_columns = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# Input types and reasonable example defaults
binary_features = {
    "gender", "device_mobile", "basket_icon_click",
    "scrolled_to_reviews", "checked_delivery_pdp",
    "saw_checkout", "sign_in", "is_returning_user"
}

integer_features = {
    "age", "traffic_source", "product_category",
    "pages_viewed", "cart_items_count_this_session",
    "lifetime_purchases_count", "days_since_last_discount",
    "previous_review_sentiment", "social_channels_followed_count"
}

defaults = {
    "age": 30,
    "gender": 0,
    "annual_income": 50000.0,
    "device_mobile": 1,
    "traffic_source": 1,
    "product_category": 1,
    "current_session_duration_sec": 120.0,
    "time_spent_deviation": 0.0,
    "pages_viewed": 4,
    "basket_icon_click": 1,
    "cart_items_count_this_session": 2,
    "cart_total_value": 50.0,
    "scrolled_to_reviews": 0,
    "checked_delivery_pdp": 0,
    "saw_checkout": 0,
    "sign_in": 1,
    "is_returning_user": 0,
    "lifetime_purchases_count": 1,
    "customer_conversion_rate": 0.10,
    "days_since_last_discount": 30,
    "previous_review_sentiment": 1,
    "social_channels_followed_count": 1
}

st.subheader("Customer session details")

with st.form("customer_form"):
    values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):
        container = col1 if i % 2 == 0 else col2

        with container:
            default = defaults.get(feature, 0)

            if feature in binary_features:
                values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=[0, 1],
                    index=int(default)
                )
            elif feature in integer_features:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=int(default),
                    min_value=0,
                    step=1
                )
            else:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(default),
                    step=1.0
                )

    submitted = st.form_submit_button(
        "Predict Discount Recommendation",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [[values[f] for f in feature_columns]],
        columns=feature_columns
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    class_labels = list(model.classes_)
    positive_index = class_labels.index(1)
    probability = float(probabilities[positive_index])

    st.divider()
    st.subheader("Prediction result")

    if prediction == 1:
        st.success("Recommendation: Consider offering a discount.")
    else:
        st.info("Recommendation: A discount is not recommended.")

    st.metric(
        "Model probability for discount recommendation",
        f"{probability:.1%}"
    )

    st.progress(probability)

    st.caption(
        "This probability reflects the model's estimate for the "
        "discount-recommendation label. It is not a guarantee that "
        "a discount will increase purchases."
    )
