import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import os
from torch.utils.data import Subset

# Assuming you have these files in your project
from gcn_model import GCNAnomalyDetector
from utils import calculate_metrics, print_metrics_comparison

class Trainer:
    def __init__(self, dataset, num_epochs=100, lr=0.005, batch_size=64, k_folds=5):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self):
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty!")

        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"\n=== Fold {fold + 1}/{self.k_folds} ===")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

            train_dataset = Subset(self.dataset, train_idx)
            val_dataset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            model = GCNAnomalyDetector(
                input_dim=self.dataset.num_features, hidden_dim=64, output_dim=2
            ).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.NLLLoss()

            best_val_loss = float("inf")
            train_losses, val_losses = [], []

            for epoch in range(self.num_epochs):
                # Training
                model.train()
                epoch_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                val_loss, val_metrics = self.evaluate(model, val_loader, criterion)
                train_losses.append(epoch_loss / len(train_loader))
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:03d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}"
                    )

            results.append(
                {
                    "fold": fold,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_metrics": val_metrics,
                }
            )

        self.print_metrics_comparison(results)
        return results

    def evaluate(self, model, data_loader, criterion):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = model(batch)
                loss = criterion(out, batch.y)
                total_loss += loss.item()
                
                preds = out.argmax(dim=1).cpu().numpy()
                probs = out.softmax(dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.5  # Default value for undefined ROC-AUC
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc
        }
        return avg_loss, metrics

    def print_metrics_comparison(self, results):
        print("\n=== Comparative Performance ===")
        print("Fold\tAccuracy\tF1-Score\tROC-AUC")
        for res in results:
            metrics = res["best_metrics"]
            print(
                f"{res['fold'] + 1}\t"
                f"{metrics['accuracy']:.4f}\t\t"
                f"{metrics['f1']:.4f}\t\t"
                f"{metrics['roc_auc']:.4f}"
            )

def main():
    print("Initializing training...")

    # Example synthetic dataset
    from torch_geometric.data import Data

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_graphs=5, num_nodes=100, num_features=10):
            self.num_features = num_features
            self.data = [
                Data(
                    x=torch.randn(num_nodes, num_features),
                    edge_index=torch.randint(0, num_nodes, (2, 200)),
                    y=torch.randint(0, 2, (num_nodes,))
                )
                for _ in range(num_graphs)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    try:
        dataset = SyntheticDataset()
        print(f"Dataset loaded with {len(dataset)} graphs")

        trainer = Trainer(dataset, num_epochs=100)
        results = trainer.train()
        print("Training completed successfully!")
        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    main()