import torch
import torch.nn as nn
import math
from utils import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg, src_pad_idx=None, trg_pad_idx=None):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        
        # Embeddings
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.trg_embedding(trg) * math.sqrt(self.d_model))
        
        # Masks
        # Target mask (look-ahead mask)
        trg_seq_len = trg.shape[1]
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(trg.device)
        
        # Key padding masks
        # True values in mask indicate the position should be IGNORED
        src_key_padding_mask = (src == src_pad_idx) if src_pad_idx is not None else None
        trg_key_padding_mask = (trg == trg_pad_idx) if trg_pad_idx is not None else None
        
        output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Masking memory (encoder output) positions corresponding to padding
        )
        
        output = self.fc_out(output)
        return output

    def encode(self, src, src_pad_idx=None):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        src_key_padding_mask = (src == src_pad_idx) if src_pad_idx is not None else None
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, trg, memory, src_key_padding_mask=None, trg_pad_idx=None):
        trg_emb = self.pos_encoder(self.trg_embedding(trg) * math.sqrt(self.d_model))
        trg_seq_len = trg.shape[1]
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(trg.device)
        trg_key_padding_mask = (trg == trg_pad_idx) if trg_pad_idx is not None else None
        
        output = self.transformer.decoder(
            trg_emb, 
            memory, 
            tgt_mask=trg_mask, 
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask
        )
        return self.fc_out(output)
