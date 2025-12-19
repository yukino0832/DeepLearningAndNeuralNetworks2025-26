import lightning
import torch
import torch.nn as nn
from model.transformer import TransformerModel

class LightningTransformer(lightning.LightningModule):
    def __init__(self, args, de_vocab, en_vocab):
        super().__init__()
        self.args = args

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.args.pad_idx,
            label_smoothing=self.args.label_smoothing
        )

        self.model = TransformerModel(
            src_vocab_size=len(de_vocab),
            trg_vocab_size=len(en_vocab),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        ) if self.args.model == 'Transformer' else None

    def forward(self, src, trg):
        return self.model(
            src,
            trg,
            src_pad_idx=self.args.pad_idx,
            trg_pad_idx=self.args.pad_idx
        )

    def training_step(self, batch, batch_idx):
        src, src_len, trg = batch

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:]

        logits = self(src, trg_input)
        # logits: (B, T-1, vocab_size)

        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            trg_y.reshape(-1)
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, src_len, trg = batch

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:]

        logits = self(src, trg_input)

        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            trg_y.reshape(-1)
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
