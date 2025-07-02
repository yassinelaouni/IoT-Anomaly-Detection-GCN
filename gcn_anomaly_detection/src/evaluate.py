import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
from gcn_model import GCNAnomalyDetector
from torch_geometric.loader import DataLoader
import os
from torch_geometric.data import Dataset
from torch_geometric.data import Data


class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        # Replace with your actual test data
        self.data = [
            Data(
                x=torch.randn(100, 10),  # 100 nodes, 10 features
                edge_index=torch.randint(0, 100, (2, 200)),  # 200 edges
                y=torch.cat([torch.zeros(70), torch.ones(30)]),  # 70 normal, 30 anomaly
            )
        ]

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved as {filename}")


def plot_roc_curve(fpr, tpr, roc_auc, filename="roc_curve.png"):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    print(f"ROC curve saved as {filename}")


def main():
    print("Starting evaluation on test dataset...")

    try:
        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load test dataset
        test_dataset = TestDataset()
        test_loader = DataLoader(
            test_dataset, batch_size=1
        )  # Process one graph at a time

        # Initialize model
        model = GCNAnomalyDetector(input_dim=10, hidden_dim=64, output_dim=2).to(device)

        # Load trained weights
        model_path = "best_model_fold0.pt"  # Or 'best_model_fold0.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                probs = torch.softmax(out, dim=1)
                preds = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(
                    probs[:, 1].cpu().numpy()
                )  # Probability of class 1 (anomaly)

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_scores = np.array(all_probs)

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0
            )
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=["Normal", "Anomaly"])

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig("precision_recall_curve.png")
        plt.close()

        print("\nEvaluation Metrics:")
        print(f"- Accuracy: {np.mean(y_true == y_pred):.4f}")
        print(f"- ROC AUC: {roc_auc:.4f}")
        print(f"- F1 Score: {f1_score(y_true, y_pred):.4f}")

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    main()
