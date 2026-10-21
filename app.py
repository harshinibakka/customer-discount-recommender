import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Discount Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM DASHBOARD STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .hero {
            padding: 2rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #14213d, #315c8c);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }

        .hero p {
            color: #e4edf8;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            padding: 1rem;
            border-radius: 12px;
        }

        .small-note {
            font-size: 0.85rem;
            opacity: 0.8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "discount_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.json"


@st.cache_resource
def load_model():
    """Load the trained model and its expected feature order."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "discount_model.pkl was not found in the repository."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "feature_columns.json was not found in the repository."
        )

    trained_model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    return trained_model, features


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛍️ Customer Discount Recommender</h1>
        <p>
            Explore online shopping behaviour and generate
            machine-learning discount recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("MACHINE LEARNING  •  RANDOM FOREST  •  STREAMLIT")

st.info(
    "**What this model predicts:** whether a customer should "
    "receive a discount according to the dataset's "
    "`should_receive_discount` label. It does not directly predict "
    "purchase completion or actual cart abandonment."
)


# --------------------------------------------------
# 5. MODEL INFORMATION
# --------------------------------------------------

with st.expander("About this project", expanded=False):
    st.markdown(
        """
        This application uses a Random Forest classifier trained
        on 1,000 shopping-session records.

        **Model development**
        - Training records: 800
        - Testing records: 200
        - Input features: 22
        - Initial reported test accuracy: 81.5%

        **Important limitations**
        - The target represents discount recommendations.
        - Test accuracy measures prediction of that label only.
        - It does not demonstrate increased sales or reduced
          cart abandonment.
        - Model probability is not a guarantee that a discount
          will cause a customer to purchase.
        """
    )

try:
    model, feature_columns = load_model()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.warning(
        "Check that discount_model.pkl and feature_columns.json "
        "are committed to the repository root."
    )
    st.stop()


# --------------------------------------------------
# 6. CUSTOMER INPUT SETTINGS
# --------------------------------------------------

binary_features = {
    "gender",
    "device_mobile",
    "basket_icon_click",
    "scrolled_to_reviews",
    "checked_delivery_pdp",
    "saw_checkout",
    "sign_in",
    "is_returning_user",
}

integer_features = {
    "age",
    "traffic_source",
    "product_category",
    "pages_viewed",
    "cart_items_count_this_session",
    "lifetime_purchases_count",
    "days_since_last_discount",
    "previous_review_sentiment",
    "social_channels_followed_count",
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
    "social_channels_followed_count": 1,
}

# Basic input constraints for common features.
minimum_values = {
    "age": 0,
    "annual_income": 0.0,
    "current_session_duration_sec": 0.0,
    "cart_items_count_this_session": 0,
    "cart_total_value": 0.0,
    "pages_viewed": 0,
    "lifetime_purchases_count": 0,
    "customer_conversion_rate": 0.0,
    "days_since_last_discount": 0,
    "social_channels_followed_count": 0,
}


# --------------------------------------------------
# 7. CUSTOMER INPUT FORM
# --------------------------------------------------

st.header("Customer session")
st.write(
    "Enter the customer's details and shopping behaviour, "
    "then select Predict to generate a recommendation."
)

st.caption(
    "The initial values are illustrative examples. "
    "Replace them with realistic values for your use case."
)

values = {}

with st.form("customer_prediction_form"):

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(feature_columns):
        column = (
            left_column if index % 2 == 0 else right_column
        )

        label = feature.replace("_", " ").title()
        default = defaults.get(feature, 0)

        with column:
            if feature in binary_features:
                values[feature] = st.selectbox(
                    label,
                    options=[0, 1],
                    index=int(default),
                    format_func=lambda value: (
                        "Yes (1)" if value == 1 else "No (0)"
                    ),
                    help="Choose the value for this session.",
                )

            elif feature in integer_features:
                values[feature] = st.number_input(
                    label,
                    min_value=int(minimum_values.get(feature, 0)),
                    value=int(default),
                    step=1,
                    help="Enter a non-negative whole number.",
                )

            else:
                values[feature] = st.number_input(
                    label,
                    min_value=float(minimum_values.get(feature, 0.0)),
                    value=float(default),
                    step=1.0,
                    help="Enter a numeric value.",
                )

    submitted = st.form_submit_button(
        "🔮 Predict Discount Recommendation",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 8. MAKE A PREDICTION
# --------------------------------------------------

if submitted:
    try:
        # Keep exactly the feature order used during training.
        input_data = pd.DataFrame(
            [[values[feature] for feature in feature_columns]],
            columns=feature_columns,
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]
        class_labels = list(model.classes_)

        if 1 not in class_labels:
            raise ValueError(
                "The trained model does not contain class 1."
            )

        positive_index = class_labels.index(1)
        discount_probability = float(
            probabilities[positive_index]
        )

        st.divider()
        st.header("Prediction result")

        metric_column, detail_column = st.columns([1, 2])

        with metric_column:
            st.metric(
                "Discount recommendation probability",
                f"{discount_probability:.1%}",
            )

        with detail_column:
            if prediction == 1:
                st.success(
                    "### Consider offering a discount\n"
                    "The model predicts class 1."
                )
            else:
                st.info(
                    "### Discount not recommended\n"
                    "The model predicts class 0."
                )

        st.write("**Model probability for class 1**")
        st.progress(discount_probability)

        st.caption(
            "This is the model's estimated probability of the "
            "dataset label `should_receive_discount = 1`. "
            "It is not a probability of purchase or cart abandonment."
        )

        with st.expander("View the submitted customer data"):
            st.dataframe(
                input_data.T.rename(
                    columns={0: "Input value"}
                ),
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.write(
            "Please check the model files and ensure that "
            "the input features match the training data."
        )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p class="small-note">
        Academic machine-learning project · Random Forest
        classification · Results depend on the training dataset.
    </p>
    """,
    unsafe_allow_html=True,
)
