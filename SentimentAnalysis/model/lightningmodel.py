import lightning
import torchmetrics
import torch
from model.BidLSTM import BidLSTM

class SentimentAnalysisModel(lightning.LightningModule):
    def __init__(self, args, word2idx, embeddings):
        super().__init__()
        self.args = args
        self.model = BidLSTM(args, word2idx, embeddings) if args.model == 'BidLSTM' else None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

        preds = torch.sigmoid(logits) >= 0.5
        acc = self.train_acc(preds.int(), y.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

        preds = torch.sigmoid(logits) >= 0.5
        acc = self.val_acc(preds.int(), y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        if x.dim() == 1:
            x = x.unsqueeze(0) 
        logits = self.model(x)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5)
        pred_text = ["positive" if p else "negative" for p in preds.squeeze(1).cpu().tolist()]

        return {
            "prob": probs.squeeze(1).cpu(),
            "pred": pred_text
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def forward(self, x):
        return self.model(x)