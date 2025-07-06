import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import os

# Supposant que vous avez ces fichiers dans votre projet
from gcn_model import GCNAnomalyDetector

class Trainer:
    """Classe pour entraîner le modèle GCN de détection d'anomalies"""
    def __init__(self, train_graph_path, val_graph_path, num_epochs=100, lr=0.005, batch_size=64):
        # Charger les graphes d'entraînement et de validation
        self.train_graph = torch.load(train_graph_path, weights_only=False)
        self.val_graph = torch.load(val_graph_path, weights_only=False)
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de l'appareil: {self.device}")

    def train(self):
        """Méthode principale d'entraînement"""
        # Créer les chargeurs de données
        train_loader = DataLoader([self.train_graph], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader([self.val_graph], batch_size=self.batch_size)

        # Initialiser le modèle GCN
        model = GCNAnomalyDetector(
            input_dim=self.train_graph.num_features, 
            hidden_dim=64, 
            output_dim=2
        ).to(self.device)

        # Configurer l'optimiseur et la fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.NLLLoss()

        # Variables pour suivre les meilleures performances
        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        # Boucle d'entraînement
        for epoch in range(self.num_epochs):
            # Phase d'entraînement
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

            # Phase de validation
            val_loss, val_metrics = self.evaluate(model, val_loader, criterion)
            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss)

            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

            # Afficher les progrès tous les 10 époques
            if epoch % 10 == 0:
                print(
                    f"Époque {epoch:03d} | Perte d'entraînement: {train_losses[-1]:.4f} | Perte de validation: {val_loss:.4f}"
                )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_metrics": val_metrics,
        }
    
    def evaluate(self, model, data_loader, criterion):
        """Évaluer le modèle sur un ensemble de données"""
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

                # Obtenir les prédictions et probabilités
                preds = out.argmax(dim=1).cpu().numpy()
                probs = out.softmax(dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)

        # Calculer les métriques
        avg_loss = total_loss / len(data_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.5  # Valeur par défaut pour ROC-AUC non défini

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }
        return avg_loss, metrics

def main():
    """Fonction principale pour lancer l'entraînement"""
    print("Initialisation de l'entraînement...")
    
    try:
        # Chemins vers vos fichiers de graphes
        train_graph_path = "data/processed/train_graph.pt"
        val_graph_path = "data/processed/val_graph.pt"
        
        # Créer et entraîner le modèle
        trainer = Trainer(
            train_graph_path=train_graph_path,
            val_graph_path=val_graph_path,
            num_epochs=100
        )
        
        results = trainer.train()
        print("Entraînement terminé avec succès!")
        return results
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        return None

if __name__ == "__main__":
    main()