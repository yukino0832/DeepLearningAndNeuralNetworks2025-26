import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

from data.dataset import get_dataloader
from model.autoencoder import AutoEncoder

# Configuration
cfg = edict({
    'channel': 1,
    'image_height': 32,
    'image_width': 32,
    'batch_size': 128,
    'embed_size': 30,
    'data_dir': 'data/MNIST',
    'epochs': 5,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'exp_dir': './exp'
})

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        
        outputs, _ = model(images)
        loss = criterion(outputs, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        recon_images, _ = model(images)
        
    return images, recon_images

def visualize(original, reconstructed, save_path):
    n = 8
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = original[i].cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        recon_img = reconstructed[i].cpu().squeeze().numpy()
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
        
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

def main():
    print(f"Using device: {cfg.device}")
    
    if not os.path.exists(cfg.exp_dir):
        os.makedirs(cfg.exp_dir)

    train_loader, test_loader = get_dataloader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        image_size=(cfg.image_height, cfg.image_width)
    )
    
    input_dim = cfg.image_height * cfg.image_width
    model = AutoEncoder(input_dim=input_dim, embed_dim=cfg.embed_size).to(cfg.device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    print("============== Starting Training ==============")
    for epoch in range(cfg.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, cfg.device)
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {train_loss:.4f}")
        
    print("============== Finish Training ==============")
    
    save_path = os.path.join(cfg.exp_dir, 'autoencoder.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    original, reconstructed = evaluate(model, test_loader, cfg.device)
    vis_path = os.path.join(cfg.exp_dir, 'result.png')
    visualize(original, reconstructed, vis_path)

if __name__ == "__main__":
    main()