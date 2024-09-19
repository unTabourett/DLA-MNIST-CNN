# Definition du modele
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deuxième couche de convolution
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche entièrement connectée
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # MNIST images are 28x28, after two pooling layers: 28/2/2 = 7
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Appliquer la première couche de convolution et le pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Appliquer la deuxième couche de convolution et le pooling
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Aplatir la sortie pour la couche entièrement connectée
        x = x.view(x.size(0), -1)  # Aplatir
        x = self.fc1(x)
        x = self.fc2(x)

        return x