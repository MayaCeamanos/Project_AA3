import torch
from torchvision import datasets, transforms

class ColoredMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True
        )
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        # Convertir a tensor (1,28,28)
        img = transforms.ToTensor()(img)

        # Generar color aleatorio (RGB)
        color = torch.rand(3, 1, 1)

        # Expandir a RGB y aplicar color
        img = img.repeat(3, 1, 1) * color

        # Aplicar transform si quieres (resize, etc)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.mnist)