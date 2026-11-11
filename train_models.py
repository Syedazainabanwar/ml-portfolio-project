import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from app_helpers import load, sidebar_form
from app_specs import REGRESSION_SPEC, LABELS

st.set_page_config(page_title="Influencer Campaign Predictor", page_icon="📈", layout="wide")
st.title("📈 Influencer Campaign Predictor")
st.caption("Estimates the direct sales (USD) a campaign will generate within 14 days of the post.")

model, features = load("revenue_model.joblib", "revenue_features.joblib")
if model is None:
    st.warning("The revenue model files (revenue_model.joblib and revenue_features.joblib) were not found.")
    st.stop()

row = sidebar_form("reg", REGRESSION_SPEC, features)
predict = st.sidebar.button("Predict campaign revenue", type="primary", use_container_width=True)

tab_predict, tab_whatif, tab_about = st.tabs(["Prediction", "What-if: boost spend", "About the model"])

with tab_predict:
    if predict:
        revenue = max(float(model.predict(row)[0]), 0.0)
        st.metric("Estimated campaign revenue", f"${revenue:,.2f}")
        st.caption("This is an estimate. Compare it with the campaign cost (creator fee plus boost spend) before deciding.")
    else:
        st.info("Enter campaign details in the sidebar, then select Predict campaign revenue.")

with tab_whatif:
    st.write("See how the estimated revenue changes when only the paid boost spend changes. "
             "All the other inputs stay as set in the sidebar.")
    if "paid_boost_spend_usd" not in features:
        st.write("Paid boost spend is not one of the features used by the final model.")
    else:
        current = float(row["paid_boost_spend_usd"].iloc[0])
        top = max(current * 2, 1000.0)
        spend_values = np.linspace(0, top, 25)
        scenarios = pd.concat([row] * len(spend_values), ignore_index=True)
        scenarios["paid_boost_spend_usd"] = spend_values
        predicted = np.clip(model.predict(scenarios), 0, None)
        chart = pd.DataFrame({"Paid boost spend (USD)": spend_values,
                              "Estimated revenue (USD)": predicted}).set_index("Paid boost spend (USD)")
        st.line_chart(chart)
        st.caption("Values far outside the range of the training data are less reliable. "
                   "Extra revenue should be compared with the extra spend.")

with tab_about:
    st.write("Features used by the final model: " + ", ".join(LABELS.get(f, f) for f in features))
    st.write("The model was chosen with cross-validation and tested on data it had not seen during training.")
