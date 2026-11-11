from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app_specs import LABELS

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load(model_file, feature_file):
    model_path, feature_path = ROOT / model_file, ROOT / feature_file
    if not (model_path.exists() and feature_path.exists()):
        return None, None
    return joblib.load(model_path), joblib.load(feature_path)


def yes_no(value):
    return "Yes" if value == 1 else "No"


def sidebar_form(prefix, spec, features):
    """Put one input per model feature in the sidebar and return them as a one-row table."""
    st.sidebar.header("Inputs")
    values = {}
    for col in features:
        kind, detail = spec[col]
        label = LABELS.get(col, col)
        key = f"{prefix}_{col}"
        if kind == "number":
            low, high, default, step = detail
            values[col] = st.sidebar.number_input(label, min_value=low, max_value=high,
                                                  value=default, step=step, key=key)
        elif detail == [0, 1]:
            values[col] = st.sidebar.selectbox(label, detail, format_func=yes_no, key=key)
        else:
            values[col] = st.sidebar.selectbox(label, detail, key=key)
    return pd.DataFrame([values])
