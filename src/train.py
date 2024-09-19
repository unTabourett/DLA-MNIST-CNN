import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import torch
import torch.nn as nn
from models.model import CNN
from data.data_loader import get_train_loader
import argparse
import os

def train_model(model, train_loader, criterion, optimizer, num_epochs, save_path):
    # Initialiser wandb avec les paramètres
    wandb.init(
        project="DLA-MNIST-CNN",
        name=args.experiment_name,  # Utilisation du nom d'expérience passé en argument
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Enregistrer les métriques avec wandb
        wandb.log({"epoch": epoch, "loss": avg_loss})

    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")


def main(args):
    # Initialisation de wandb avec les paramètres de configuration
    wandb.init(
        project="DLA-MNIST-CNN",
        name=args.experiment_name,  # Utilisation du nom d'expérience passé en argument
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }
    )

    train_loader = get_train_loader(batch_size=args.batch_size)
    model = CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, criterion, optimizer, args.num_epochs, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for CNN model')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_path', type=str, default='models/cnn_model.pth', help='Path to save the trained model')
    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='Name of the experiment')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(args)
