import streamlit as st
import numpy as np
import pickle

# --- 1. PAGE CONFIGURATION ---
# This gives the browser tab a premium look and expands the layout
st.set_page_config(page_title="Churn Interception System", page_icon="⬢", layout="wide")


# --- 2. LOAD THE ENGINE ---
@st.cache_resource  # This caches the model so it doesn't reload on every click
def load_components():
    model = pickle.load(open("churn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model.set_params(device="cpu")
    return model, scaler


model, scaler = load_components()

# --- 3. UI DASHBOARD LAYOUT ---
st.title("⬢ Enterprise Churn Interception System")
st.markdown(
    "Deploying XGBoost Predictive Analytics to calculate real-time flight risk."
)
st.divider()

# Create a clean Sidebar for inputs
with st.sidebar:
    st.header("Customer Profile")
    st.markdown("Input live data to evaluate retention probability.")

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 15.0, 120.0, 75.0)

    # In a full app, you would add the rest of the dropdowns here
    contract_type = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )

    analyze_button = st.button(
        "Run Diagnostic", type="primary", use_container_width=True
    )

# --- 4. PREDICTION LOGIC ---
if analyze_button:
    with st.spinner("Accessing NVIDIA CUDA Cores for inference..."):
        # Create a blank 30-column array matching your training data shape
        input_data = np.zeros((1, 30))

        # Map the inputs to the correct columns (Assuming Tenure is col 1, Charges col 2)
        # Note: In a true production app, you map all dropdowns to their exact One-Hot Encoded indices
        input_data[0][1] = tenure
        input_data[0][2] = monthly_charges

        # Scale and Predict
        scaled_input = scaler.transform(input_data)
        churn_probability = model.predict_proba(scaled_input)[0][1]

    # --- 5. RESULTS RENDER ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="Calculated Flight Risk", value=f"{churn_probability * 100:.1f}%"
        )

    with col2:
        st.subheader("Business Recommendation")
        # Apply the custom 35% business threshold
        if churn_probability >= 0.35:
            st.error("🚨 **CRITICAL RISK DETECTED**")
            st.markdown(
                f"> **Action Required:** Probability of churn exceeds the 35% threshold. Immediately deploy targeted retention protocol or $10 discount initiative."
            )
            st.progress(float(churn_probability))
        else:
            st.success("✅ **STABLE ACCOUNT**")
            st.markdown(
                "> **Status:** Customer shows high loyalty markers. No immediate financial intervention required."
            )
            st.progress(float(churn_probability))
else:
    st.info("Awaiting customer data input in the sidebar.")
