import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app_helpers import load, sidebar_form
from app_specs import CLASSIFICATION_SPEC, LABELS

st.set_page_config(page_title="Order Return Risk Predictor", page_icon="🛍️", layout="wide")
st.title("🛍️ Order Return Risk Predictor")
st.caption("Estimates the chance that an online fashion order will be returned.")

model, features = load("return_model.joblib", "return_features.joblib")
if model is None:
    st.warning("The return model files (return_model.joblib and return_features.joblib) were not found.")
    st.stop()

row = sidebar_form("cls", CLASSIFICATION_SPEC, features)
predict = st.sidebar.button("Predict return risk", type="primary", use_container_width=True)

tab_predict, tab_about = st.tabs(["Prediction", "About the model"])

with tab_predict:
    if predict:
        probability = float(model.predict_proba(row)[0, 1])
        st.metric("Probability the order is returned", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        if probability >= 0.5:
            st.error("High return risk: consider sizing help or a fit warning before shipping.")
        else:
            st.success("Low return risk: the order is likely to be kept.")
    else:
        st.info("Enter the order details in the sidebar, then select Predict return risk.")

with tab_about:
    st.write("Features used by the final model: " + ", ".join(LABELS.get(f, f) for f in features))
    st.write("An order is flagged as high risk when the predicted probability is 50% or more.")
