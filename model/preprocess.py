import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "y"


def validate_target_column(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Missing target column: y")


def detect_outliers_iqr(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return 0.0, {}

    outlier_flags = pd.Series(False, index=df.index)
    outlier_pct_by_col: dict[str, float] = {}
    for col in numeric_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        col_flags = (s < low) | (s > high)
        outlier_flags = outlier_flags | col_flags
        outlier_pct_by_col[col] = round(float(col_flags.mean() * 100), 2)

    total_outlier_pct = round(float(outlier_flags.mean() * 100), 2)
    return total_outlier_pct, outlier_pct_by_col


def validate_and_clean(
    df: pd.DataFrame,
    require_target: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    print(f"Rows before cleaning: {len(df)}")
    null_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    total_outlier_pct, outlier_pct_by_col = detect_outliers_iqr(df)
    print(f"Null values: {null_count}")
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Rows with any numeric outlier (%): {total_outlier_pct}")
    print(f"Outlier % by numeric column: {outlier_pct_by_col}")

    if require_target:
        validate_target_column(df)
        if df[TARGET_COLUMN].nunique() < 2:
            raise ValueError("Target column must have at least two classes")

    if drop_duplicates and duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Rows after duplicate removal: {len(df)}")

    return df


def split_features_target(df: pd.DataFrame):
    validate_target_column(df)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)
    feature_columns = x_train.columns.tolist()

    numeric_cols = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    print("Fitting preprocessor on training data")
    x_train_processed = preprocessor.fit_transform(x_train)
    print("Transforming test data using fitted preprocessor")
    x_test_processed = preprocessor.transform(x_test)
    return x_train_processed, x_test_processed, y_train, y_test, preprocessor, feature_columns
