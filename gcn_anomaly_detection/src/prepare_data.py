import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def visualize_anomaly_distribution(
    train_normal, train_anomaly, val_normal, val_anomaly, test_normal, test_anomaly
):
    """Create a clean bar chart visualization of anomaly distribution"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data preparation
    splits = ["Train", "Validation", "Test"]
    normal_counts = [train_normal, val_normal, test_normal]
    anomaly_counts = [train_anomaly, val_anomaly, test_anomaly]

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
        "Normal vs Anomaly Distribution Across Splits",
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
    """Load data and split into train/val/test sets before any processing"""
    df = pd.read_csv(file_path)
    # First split off test
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["Anomaly"])
    # Then split train/val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42, stratify=train_val_df["Anomaly"])
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
    device_ids = df["Device_ID"].copy() if "Device_ID" in df.columns else None
    return X_scaled, y, device_ids, scaler

def get_anomaly_counts_from_dfs(train_df, val_df, test_df, label_col="Anomaly"):
    """Utility to compute normal and anomaly counts from DataFrames."""
    def count(df, value):
        return (df[label_col] == value).sum()
    train_normal = count(train_df, 0)
    train_anomaly = count(train_df, 1)
    val_normal = count(val_df, 0)
    val_anomaly = count(val_df, 1)
    test_normal = count(test_df, 0)
    test_anomaly = count(test_df, 1)
    return train_normal, train_anomaly, val_normal, val_anomaly, test_normal, test_anomaly

# Example usage with real data
if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual data file path
    data_file = "data/synthetic_iot_dataset.csv"
    try:
        train_df, val_df, test_df = load_and_split_data(data_file)
        counts = get_anomaly_counts_from_dfs(train_df, val_df, test_df)
        visualize_anomaly_distribution(*counts)
    except FileNotFoundError:
        print(f"Data file '{data_file}' not found. Please update the path and try again.")
