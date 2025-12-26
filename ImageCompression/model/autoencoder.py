import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=2):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        encoded = self.encoder(x_flat)
        
        decoded_flat = self.decoder(encoded)
        
        decoded = decoded_flat.view(batch_size, 1, 32, 32)
        decoded = decoded_flat.view(batch_size, 1, 32, 32)
        
        return decoded, encoded

