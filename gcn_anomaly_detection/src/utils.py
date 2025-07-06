from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)
import numpy as np


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculer toutes les métriques d'évaluation"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_probs),
    }


def print_metrics_comparison(results):
    """Afficher une comparaison des métriques entre les plis de validation croisée"""
    print("\n=== Performance Comparative ===")
    print("Plis\tPrécision\tScore-F1\tROC-AUC")
    for res in results:
        print(
            f"{res['fold']+1}\t"
            f"{res['best_metrics']['accuracy']:.4f}\t\t"
            f"{res['best_metrics']['f1']:.4f}\t\t"
            f"{res['best_metrics']['roc_auc']:.4f}"
        )

    # Calculer les métriques moyennes
    avg_metrics = {
        "accuracy": np.mean([r["best_metrics"]["accuracy"] for r in results]),
        "f1": np.mean([r["best_metrics"]["f1"] for r in results]),
        "roc_auc": np.mean([r["best_metrics"]["roc_auc"] for r in results]),
    }

    print("\n=== Métriques Moyennes ===")
    print(f"Précision: {avg_metrics['accuracy']:.4f}")
    print(f"Score-F1: {avg_metrics['f1']:.4f}")
    print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f}")
