import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def visualize_anomaly_distribution(
    train_normal, train_anomaly, val_normal, val_anomaly, test_normal, test_anomaly
):
    """Create a clean bar chart visualization of anomaly distribution"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data preparation
    splits = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    normal_counts = [train_normal, val_normal, test_normal]
    anomaly_counts = [train_anomaly, val_anomaly, test_anomaly]
    total_counts = [n + a for n, a in zip(normal_counts, anomaly_counts)]

    # Create stacked bar chart
    bar_width = 0.6
    index = np.arange(len(splits))

    # Plot normal nodes (green)
    normal_bars = ax.bar(
        index,
        normal_counts,
        bar_width,
        label="Normal",
        color="#2ca02c",
        edgecolor="white",
    )

    # Plot anomaly nodes (red) on top of normal
    anomaly_bars = ax.bar(
        index,
        anomaly_counts,
        bar_width,
        bottom=normal_counts,
        label="Anomaly",
        color="#d62728",
        edgecolor="white",
    )

    # Customize the plot
    ax.set_title(
        "Dataset Split Distribution (70-15-15)",
        pad=20,
        fontsize=14,
        color="white",
    )
    ax.set_xlabel("Dataset Split", color="white")
    ax.set_ylabel("Number of Nodes", color="white")
    ax.set_xticks(index)
    ax.set_xticklabels(splits, color="white")
    ax.tick_params(axis="y", colors="white")

    # Add value labels on each bar segment
    for i in range(len(splits)):
        # Normal count label
        ax.text(
            i,
            normal_counts[i] / 2,
            f"{normal_counts[i]:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

        # Anomaly count label
        ax.text(
            i,
            normal_counts[i] + anomaly_counts[i] / 2,
            f"{anomaly_counts[i]:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Add legend
    ax.legend(loc="upper right")

    # Remove spines for cleaner look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig("anomaly_distribution_chart.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_and_split_data(file_path, test_size=0.15, val_size=0.15):
    """Load data and split into 70% train, 15% val, 15% test"""
    df = pd.read_csv(file_path)

    # First split into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=42, stratify=df["Anomaly"]
    )

    # Then split temp into val (15%) and test (15%)
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio, random_state=42, stratify=temp_df["Anomaly"]
    )

    return train_df, val_df, test_df


def preprocess_data(df, scaler=None):
    """Preprocess a single dataset split"""
    X = df[["Temperature", "Humidity", "Battery_Level"]]
    y = df["Anomaly"]
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler


def get_anomaly_counts_from_dfs(train_df, val_df, test_df, label_col="Anomaly"):
    """Utility to compute normal and anomaly counts from DataFrames."""

    def count(df, value):
        return (df[label_col] == value).sum()

    return (
        count(train_df, 0),
        count(train_df, 1),
        count(val_df, 0),
        count(val_df, 1),
        count(test_df, 0),
        count(test_df, 1),
    )


if __name__ == "__main__":
    # Example usage with real data
    data_file = "data/synthetic_iot_dataset.csv"
    try:
        # Load and split data
        train_df, val_df, test_df = load_and_split_data(data_file)

        # Print split sizes
        print(
            f"Train size: {len(train_df):,} ({(len(train_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )
        print(
            f"Val size: {len(val_df):,} ({(len(val_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )
        print(
            f"Test size: {len(test_df):,} ({(len(test_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )

        # Visualize distribution
        counts = get_anomaly_counts_from_dfs(train_df, val_df, test_df)
        visualize_anomaly_distribution(*counts)

        # Preprocess data
        train_X, train_y, scaler = preprocess_data(train_df)
        val_X, val_y, _ = preprocess_data(val_df, scaler)
        test_X, test_y, _ = preprocess_data(test_df, scaler)

        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        np.savez(
            "data/processed/processed_data.npz",
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
        )
        print("Data successfully processed and saved to data/processed/")

    except FileNotFoundError:
        print(
            f"Data file '{data_file}' not found. Please update the path and try again."
        )
    except Exception as e:
        print(f"Error during processing: {str(e)}")
