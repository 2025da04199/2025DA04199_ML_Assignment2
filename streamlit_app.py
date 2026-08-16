import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "model"
CONFIG_DIR = MODEL_DIR / "config"
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from evaluation_utils import evaluate_trained_model
from preprocess import split_features_target, validate_and_clean


def load_artifacts():
    with (CONFIG_DIR / "model_manifest.json").open("r", encoding="utf-8") as f:
        model_manifest = json.load(f)
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    with (CONFIG_DIR / "feature_columns.json").open("r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model_manifest, preprocessor, feature_columns


def load_selected_model(model_name: str, model_manifest: dict[str, dict[str, object]]):
    return joblib.load(MODEL_DIR / str(model_manifest[model_name]["file_name"]))


def load_uploaded_csv(uploaded_file):
    return pd.read_csv(uploaded_file, sep=";")


def validate_uploaded_test_data(df: pd.DataFrame, feature_columns: list[str]):
    if df.columns.duplicated().any():
        raise ValueError("Uploaded file has duplicate column names")
    cleaned_df = validate_and_clean(df, require_target=True, drop_duplicates=True)
    x_uploaded, y_uploaded = split_features_target(cleaned_df)
    missing_columns = [col for col in feature_columns if col not in x_uploaded.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")
    x_uploaded = x_uploaded[feature_columns]
    return x_uploaded, y_uploaded


def transform_uploaded_features(x_uploaded: pd.DataFrame, preprocessor):
    return preprocessor.transform(x_uploaded)


def evaluate_selected_model(model, x_uploaded, y_uploaded, requires_dense: bool):
    metrics, details = evaluate_trained_model(
        model,
        x_uploaded,
        y_uploaded,
        requires_dense=requires_dense,
    )
    labels = details["labels"]
    cm = details["confusion_matrix"]
    report = details["classification_report"]
    return metrics, labels, cm, report


def render_model_selector(model_names: list[str]):
    return st.selectbox("Select model", model_names)


def render_metrics_and_report(metrics: dict, labels, cm, report):
    st.subheader("Evaluation Metrics")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


def main():
    st.set_page_config(page_title="ML Assignment 2", layout="wide")
    st.title("ML Assignment 2")

    try:
        model_manifest, preprocessor, feature_columns = load_artifacts()
    except Exception as e:
        st.error(f"Artifacts not found. Run training first. Details: {e}")
        return

    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        uploaded_df = load_uploaded_csv(uploaded_file)
        x_uploaded, y_uploaded = validate_uploaded_test_data(uploaded_df, feature_columns)
        x_uploaded_processed = transform_uploaded_features(x_uploaded, preprocessor)
    except Exception as e:
        st.error(f"Uploaded data validation failed: {e}")
        return

    selected_model_name = render_model_selector(list(model_manifest.keys()))
    model = load_selected_model(selected_model_name, model_manifest)
    requires_dense = bool(model_manifest[selected_model_name]["requires_dense"])
    metrics, labels, cm, report = evaluate_selected_model(
        model,
        x_uploaded_processed,
        y_uploaded,
        requires_dense,
    )
    render_metrics_and_report(metrics, labels, cm, report)


if __name__ == "__main__":
    main()
