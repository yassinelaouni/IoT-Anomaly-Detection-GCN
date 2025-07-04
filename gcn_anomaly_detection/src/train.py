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
    def __init__(self, train_graph_path, val_graph_path, num_epochs=100, lr=0.005, batch_size=64):
        self.train_graph = torch.load(train_graph_path, weights_only=False)
        self.val_graph = torch.load(val_graph_path, weights_only=False)
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self):
        train_loader = DataLoader([self.train_graph], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader([self.val_graph], batch_size=self.batch_size)

        model = GCNAnomalyDetector(
            input_dim=self.train_graph.num_features, 
            hidden_dim=64, 
            output_dim=2
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
                torch.save(model.state_dict(), "best_model.pt")

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}"
                )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_metrics": val_metrics,
        }
    
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

def main():
    print("Initializing training...")
    
    try:
        # Paths to your graph files
        train_graph_path = "data/processed/train_graph.pt"
        val_graph_path = "data/processed/val_graph.pt"
        
        trainer = Trainer(
            train_graph_path=train_graph_path,
            val_graph_path=val_graph_path,
            num_epochs=100
        )
        
        results = trainer.train()
        print("Training completed successfully!")
        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None
if __name__ == "__main__":
    main()