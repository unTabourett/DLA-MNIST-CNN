import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import torch
import torch.nn as nn
from data.data_loader import get_test_loader  # Vous devez l'implémenter dans data_loader.py
from models.model import CNN  # Importez votre modèle
import argparse

# Fonction pour évaluer le modèle
def evaluate_model(model, test_loader, criterion):
    wandb.init(project="DLA-MNIST-CNN", name=args.experiment_name)

    model.eval()  # Met le modèle en mode évaluation
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Désactiver la dérivation des gradients
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculer la perte
            total_loss += loss.item()

            # Calculer l'exactitude
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Enregistrer les résultats avec wandb
    wandb.log({"test_loss": avg_loss, "accuracy": accuracy})

# Fonction principale
def main(args):
    # Initialisation des paramètres pour wandb
    #wandb.config.experiment_name = args.experiment_name Je crois que c'est k

    # Initialisation du modèle
    model = CNN(num_classes=10)
    
    # Charger le modèle spécifié dans l'argument --model_path
    model.load_state_dict(torch.load(args.model_path))
    print(f"Modèle chargé depuis {args.model_path}")

    # Initialisation du critère de perte
    criterion = nn.CrossEntropyLoss()

    # Charger les données de test
    test_loader = get_test_loader(batch_size=args.batch_size)

    # Évaluer le modèle
    evaluate_model(model, test_loader, criterion)

if __name__ == "__main__":
    # Argument parser pour personnaliser les paramètres via la ligne de commande
    parser = argparse.ArgumentParser(description='Evaluation script for CNN model')

    # Argument pour spécifier le chemin vers le modèle à évaluer
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--experiment_name', type=str, default='default_evaluation', help='Name of the evaluation experiment')

    args = parser.parse_args()

    main(args)
