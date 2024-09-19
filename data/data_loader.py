from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_train_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Remplacez par votre dataset personnalisé si nécessaire
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


# Fonction pour charger les données de test
def get_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST normalisation
    ])

    # Remplacez par votre dataset personnalisé si nécessaire
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader