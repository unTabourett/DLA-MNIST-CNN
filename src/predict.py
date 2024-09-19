import torch
from models.model import CNN  # Importez votre modèle
from torchvision import transforms
from PIL import Image

# Fonction pour charger une image et la prétraiter
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Adaptez selon la taille d'entrée de votre modèle
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Charger l'image et appliquer les transformations
    image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris si c'est MNIST
    image = transform(image).unsqueeze(0)  # Ajouter une dimension batch
    return image

# Fonction pour faire une prédiction avec le modèle
def predict(model, image_tensor):
    model.eval()  # Mettre le modèle en mode évaluation
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Obtenir la classe prédite
    return predicted.item()

# Fonction principale
def main(image_path):
    # Charger le modèle
    model = CNN(num_classes=10)  # Ajustez selon votre configuration
    model.load_state_dict(torch.load('path/to/saved_model.pth'))  # Chargez le modèle sauvegardé

    # Prétraiter l'image
    image_tensor = preprocess_image(image_path)

    # Faire une prédiction
    prediction = predict(model, image_tensor)
    print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    # Remplacer par le chemin de l'image à prédire
    image_path = 'path/to/image.png'
    main(image_path)
