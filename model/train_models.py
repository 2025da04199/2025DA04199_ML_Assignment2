from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from eda import run_eda
from evaluation_utils import fit_and_evaluate_model
from preprocess import TARGET_COLUMN, preprocess_data, validate_and_clean


TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_model_specs() -> dict[str, dict[str, object]]:
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "file_name": "logistic_regression.joblib",
            "requires_dense": False,
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            "file_name": "decision_tree.joblib",
            "requires_dense": False,
        },
        "KNN": {
            "estimator": KNeighborsClassifier(n_neighbors=5),
            "file_name": "knn.joblib",
            "requires_dense": False,
        },
        "Naive Bayes": {
            "estimator": GaussianNB(),
            "file_name": "naive_bayes.joblib",
            "requires_dense": True,
        },
        "Random Forest (Ensemble)": {
            "estimator": RandomForestClassifier(
                n_estimators=180,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "file_name": "random_forest.joblib",
            "requires_dense": False,
        },
    }


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    print("Dataset loaded")
    return validate_and_clean(df)


def split_data(df: pd.DataFrame):
    train_df, eval_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    print("Train-evaluation split completed")
    return train_df, eval_df


def run_preprocessing(train_df: pd.DataFrame, eval_df: pd.DataFrame):
    x_train_processed, x_eval_processed, y_train, y_eval, preprocessor, feature_columns = preprocess_data(
        train_df,
        eval_df,
    )
    print("Preprocessing completed")
    return x_train_processed, x_eval_processed, y_train, y_eval, preprocessor, feature_columns


def train_and_evaluate(x_train, y_train, x_eval, y_eval):
    model_specs = get_model_specs()
    metrics_rows = []
    details_by_model = {}
    trained_models = {}

    for model_name, model_spec in model_specs.items():
        model = model_spec["estimator"]
        requires_dense = bool(model_spec["requires_dense"])
        print(f"Training: {model_name}")
        metrics, details = fit_and_evaluate_model(
            model,
            x_train,
            y_train,
            x_eval,
            y_eval,
            requires_dense=requires_dense,
        )
        metrics_rows.append({"ML Model Name": model_name, **metrics})
        details_by_model[model_name] = details
        trained_models[model_name] = model

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, details_by_model, trained_models, model_specs


def save_evaluation_outputs(metrics_df: pd.DataFrame, details_by_model: dict, metrics_path: Path, details_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details_by_model, f, indent=2)


def save_artifacts(
    model_dir: Path,
    config_dir: Path,
    trained_models: dict[str, object],
    model_specs: dict[str, dict[str, object]],
    preprocessor,
    feature_columns: list[str],
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config_dir / "model_manifest.json"
    preprocessor_path = model_dir / "preprocessor.joblib"
    feature_columns_path = config_dir / "feature_columns.json"
    model_manifest = {}

    for model_name, model in trained_models.items():
        model_file = str(model_specs[model_name]["file_name"])
        requires_dense = bool(model_specs[model_name]["requires_dense"])
        joblib.dump(model, model_dir / model_file, compress=3)
        model_manifest[model_name] = {
            "file_name": model_file,
            "requires_dense": requires_dense,
        }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)

    joblib.dump(preprocessor, preprocessor_path, compress=3)
    with feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "train_data.csv"
    model_dir = root_dir / "model"
    output_dir = model_dir / "output"
    config_dir = model_dir / "config"
    metrics_path = output_dir / "metrics.csv"
    details_path = output_dir / "evaluation_details.json"

    df = load_and_clean_data(data_path)
    run_eda(df, output_dir)
    train_df, eval_df = split_data(df)
    x_train_processed, x_eval_processed, y_train, y_eval, preprocessor, feature_columns = run_preprocessing(
        train_df,
        eval_df,
    )

    metrics_df, details_by_model, trained_models, model_specs = train_and_evaluate(
        x_train_processed,
        y_train,
        x_eval_processed,
        y_eval,
    )

    save_evaluation_outputs(metrics_df, details_by_model, metrics_path, details_path)
    save_artifacts(model_dir, config_dir, trained_models, model_specs, preprocessor, feature_columns)

    print("Model training and evaluation completed")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
