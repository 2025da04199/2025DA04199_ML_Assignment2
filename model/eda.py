from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_COLUMN = "y"
JOB_COLUMN = "job"


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    target_dist = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="count")
    )

    plt.figure(figsize=(6, 4))
    sns.barplot(data=target_dist, x="class", y="count")
    plt.title("Target Distribution (y)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_target_distribution.png", dpi=150)
    plt.close()


def plot_subscription_rate_by_job(df: pd.DataFrame, output_dir: Path) -> None:
    if JOB_COLUMN not in df.columns:
        raise ValueError("Missing column: job")

    rates = pd.crosstab(df[JOB_COLUMN], df[TARGET_COLUMN], normalize="index") * 100

    positive_col = "yes" if "yes" in rates.columns else rates.columns[-1]
    job_rates = rates[positive_col].sort_values(ascending=False).reset_index()
    job_rates.columns = [JOB_COLUMN, "subscription_rate_pct"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=job_rates, x=JOB_COLUMN, y="subscription_rate_pct")
    plt.title("Subscription Rate by Job")
    plt.xlabel("Job")
    plt.ylabel("Subscription Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_subscription_rate_by_job.png", dpi=150)
    plt.close()


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Missing target column: y")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df, output_dir)
    plot_subscription_rate_by_job(df, output_dir)

    print(f"Saved: {output_dir / 'eda_target_distribution.png'}")
    print(f"Saved: {output_dir / 'eda_subscription_rate_by_job.png'}")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "train_data.csv"
    output_dir = root_dir / "model" / "output"

    df = pd.read_csv(data_path, sep=";")
    run_eda(df, output_dir)


if __name__ == "__main__":
    main()
