from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)


def calculate_metrics(y_true, y_pred, y_probs):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_probs),
    }


def print_metrics_comparison(results):
    print("\n=== Comparative Performance ===")
    print("Fold\tAccuracy\tF1-Score\tROC-AUC")
    for res in results:
        print(
            f"{res['fold']+1}\t"
            f"{res['best_metrics']['accuracy']:.4f}\t\t"
            f"{res['best_metrics']['f1']:.4f}\t\t"
            f"{res['best_metrics']['roc_auc']:.4f}"
        )

    avg_metrics = {
        "accuracy": np.mean([r["best_metrics"]["accuracy"] for r in results]),
        "f1": np.mean([r["best_metrics"]["f1"] for r in results]),
        "roc_auc": np.mean([r["best_metrics"]["roc_auc"] for r in results]),
    }

    print("\n=== Average Metrics ===")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"F1-Score: {avg_metrics['f1']:.4f}")
    print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f}")
