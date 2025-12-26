import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size, image_size=(32, 32), download=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader