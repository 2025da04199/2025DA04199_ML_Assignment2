from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocess import TARGET_COLUMN, preprocess_data, validate_and_clean


def get_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def as_dense_if_needed(x):
    return x.toarray() if hasattr(x, "toarray") else x


def compute_auc(y_true, y_proba, classes) -> float:
    unique_classes = list(pd.Series(y_true).unique())
    if len(unique_classes) == 2:
        positive_label = "yes" if "yes" in classes else classes[1]
        pos_index = list(classes).index(positive_label)
        return float(roc_auc_score(y_true, y_proba[:, pos_index]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def evaluate_model(model_name: str, model, x_train, y_train, x_eval, y_eval):
    train_x = as_dense_if_needed(x_train) if model_name == "Naive Bayes" else x_train
    eval_x = as_dense_if_needed(x_eval) if model_name == "Naive Bayes" else x_eval

    model.fit(train_x, y_train)
    y_pred = model.predict(eval_x)
    y_proba = model.predict_proba(eval_x)
    classes = model.classes_

    average_type = "binary" if len(classes) == 2 else "weighted"
    positive_label = "yes" if "yes" in classes else classes[-1]

    metrics = {
        "Accuracy": float(accuracy_score(y_eval, y_pred)),
        "AUC": compute_auc(y_eval, y_proba, classes),
        "Precision": float(
            precision_score(
                y_eval,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_eval,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_eval,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_eval, y_pred)),
    }

    details = {
        "labels": list(classes),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, details


def train_and_evaluate(x_train, y_train, x_eval, y_eval):
    models = get_models()
    metrics_rows = []
    details_by_model = {}

    for model_name, model in models.items():
        print(f"Training: {model_name}")
        metrics, details = evaluate_model(model_name, model, x_train, y_train, x_eval, y_eval)
        metrics_rows.append({"ML Model Name": model_name, **metrics})
        details_by_model[model_name] = details

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, details_by_model


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "train_data.csv"
    metrics_path = root_dir / "model" / "metrics.csv"
    details_path = root_dir / "model" / "evaluation_details.json"

    df = pd.read_csv(data_path, sep=";")
    print("Dataset loaded")
    df = validate_and_clean(df)

    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )
    print("Train-evaluation split completed")
    x_train_processed, x_eval_processed, y_train, y_eval, _ = preprocess_data(train_df, eval_df)
    print("Preprocessing completed")

    metrics_df, details_by_model = train_and_evaluate(
        x_train_processed,
        y_train,
        x_eval_processed,
        y_eval,
    )

    metrics_df.to_csv(metrics_path, index=False)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details_by_model, f, indent=2)

    print("Model training and evaluation completed")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
