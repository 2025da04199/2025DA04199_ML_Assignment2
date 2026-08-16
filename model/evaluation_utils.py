import pandas as pd
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


def as_dense_if_needed(x, requires_dense: bool):
    if requires_dense and hasattr(x, "toarray"):
        return x.toarray()
    return x


def compute_auc(y_true, y_proba, classes) -> float:
    unique_classes = list(pd.Series(y_true).unique())
    if len(unique_classes) == 2:
        positive_label = "yes" if "yes" in classes else classes[1]
        pos_index = list(classes).index(positive_label)
        return float(roc_auc_score(y_true, y_proba[:, pos_index]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def metrics_and_details(y_true, y_pred, y_proba, classes):
    average_type = "binary" if len(classes) == 2 else "weighted"
    positive_label = "yes" if "yes" in classes else classes[-1]

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": compute_auc(y_true, y_proba, classes),
        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_true,
                y_pred,
                average=average_type,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    details = {
        "labels": list(classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, details


def predict_and_score(model, x_eval, y_eval, requires_dense: bool = False):
    eval_x = as_dense_if_needed(x_eval, requires_dense)
    y_pred = model.predict(eval_x)
    y_proba = model.predict_proba(eval_x)
    classes = model.classes_
    return metrics_and_details(y_eval, y_pred, y_proba, classes)


def fit_and_evaluate_model(model, x_train, y_train, x_eval, y_eval, requires_dense: bool = False):
    train_x = as_dense_if_needed(x_train, requires_dense)
    model.fit(train_x, y_train)
    return predict_and_score(model, x_eval, y_eval, requires_dense=requires_dense)


def evaluate_trained_model(model, x_eval, y_eval, requires_dense: bool = False):
    return predict_and_score(model, x_eval, y_eval, requires_dense=requires_dense)
