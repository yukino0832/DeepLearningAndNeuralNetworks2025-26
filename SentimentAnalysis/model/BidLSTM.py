import torch.nn as nn
import torch

class BidLSTM(nn.Module):
    def __init__(
        self,
        args,
        word2idx,
        embeddings,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        is_bid=True,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings,
            freeze=False,
            padding_idx=word2idx["<pad>"]
        )
        self.lstm = nn.LSTM(
            embeddings.size(1),
            hidden_dim,
            num_layers=n_layers,
            bidirectional=is_bid,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        hidden = self.dropout(hidden)
        return self.fc(hidden)