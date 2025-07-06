import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def visualize_anomaly_distribution(
    train_normal, train_anomaly, val_normal, val_anomaly, test_normal, test_anomaly
):
    """Créer un graphique en barres propre de la distribution des anomalies"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Préparation des données
    splits = ["Entraînement (70%)", "Validation (15%)", "Test (15%)"]
    normal_counts = [train_normal, val_normal, test_normal]
    anomaly_counts = [train_anomaly, val_anomaly, test_anomaly]
    total_counts = [n + a for n, a in zip(normal_counts, anomaly_counts)]

    # Créer un graphique en barres empilées
    bar_width = 0.6
    index = np.arange(len(splits))

    # Tracer les nœuds normaux (vert)
    normal_bars = ax.bar(
        index,
        normal_counts,
        bar_width,
        label="Normal",
        color="#2ca02c",
        edgecolor="white",
    )

    # Tracer les nœuds d'anomalie (rouge) au-dessus des normaux
    anomaly_bars = ax.bar(
        index,
        anomaly_counts,
        bar_width,
        bottom=normal_counts,
        label="Anomalie",
        color="#d62728",
        edgecolor="white",
    )

    # Personnaliser le graphique
    ax.set_title(
        "Distribution des Ensembles de Données (70-15-15)",
        pad=20,
        fontsize=14,
        color="white",
    )
    ax.set_xlabel("Ensemble de Données", color="white")
    ax.set_ylabel("Nombre de Nœuds", color="white")
    ax.set_xticks(index)
    ax.set_xticklabels(splits, color="white")
    ax.tick_params(axis="y", colors="white")

    # Ajouter les étiquettes de valeur sur chaque segment de barre
    for i in range(len(splits)):
        # Étiquette du compte normal
        ax.text(
            i,
            normal_counts[i] / 2,
            f"{normal_counts[i]:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

        # Étiquette du compte d'anomalie
        ax.text(
            i,
            normal_counts[i] + anomaly_counts[i] / 2,
            f"{anomaly_counts[i]:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Ajouter la légende
    ax.legend(loc="upper right")

    # Supprimer les bordures pour un look plus propre
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig("anomaly_distribution_chart.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_and_split_data(file_path, test_size=0.15, val_size=0.15):
    """Charger les données et les diviser en 70% train, 15% val, 15% test"""
    df = pd.read_csv(file_path)

    # Première division en train (70%) et temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=42, stratify=df["Anomaly"]
    )

    # Puis diviser temp en val (15%) et test (15%)
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio, random_state=42, stratify=temp_df["Anomaly"]
    )

    return train_df, val_df, test_df


def preprocess_data(df, scaler=None):
    """Prétraiter un seul ensemble de données"""
    # Extraire les caractéristiques et les étiquettes
    X = df[["Temperature", "Humidity", "Battery_Level"]]
    y = df["Anomaly"]
    
    # Standardiser les caractéristiques
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler


def get_anomaly_counts_from_dfs(train_df, val_df, test_df, label_col="Anomaly"):
    """Utilitaire pour calculer les comptes normaux et d'anomalies à partir des DataFrames."""

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
    # Exemple d'utilisation avec des données réelles
    data_file = "data/synthetic_iot_dataset.csv"
    try:
        # Charger et diviser les données
        train_df, val_df, test_df = load_and_split_data(data_file)

        # Afficher les tailles des ensembles
        print(
            f"Taille d'entraînement: {len(train_df):,} ({(len(train_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )
        print(
            f"Taille de validation: {len(val_df):,} ({(len(val_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )
        print(
            f"Taille de test: {len(test_df):,} ({(len(test_df)/(len(train_df)+len(val_df)+len(test_df))):.0%})"
        )

        # Visualiser la distribution
        counts = get_anomaly_counts_from_dfs(train_df, val_df, test_df)
        visualize_anomaly_distribution(*counts)

        # Prétraiter les données
        train_X, train_y, scaler = preprocess_data(train_df)
        val_X, val_y, _ = preprocess_data(val_df, scaler)
        test_X, test_y, _ = preprocess_data(test_df, scaler)

        # Sauvegarder les données prétraitées
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
        print("Données prétraitées et sauvegardées dans data/processed/")

    except FileNotFoundError:
        print(
            f"Fichier de données '{data_file}' non trouvé. Veuillez mettre à jour le chemin et réessayer."
        )
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
