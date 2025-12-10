import lightning
import torchmetrics
import torch
from model.FNN import FNN
import os
import json

class MinstDetectModel(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = FNN() if args.model == 'FNN' else None
        self.lr = args.lr
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_results = []
        self.output_path = args.output_path

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(x_hat, y)
        acc = self.train_acc(torch.argmax(x_hat, dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(x_hat, y)
        acc = self.val_acc(torch.argmax(x_hat, dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, path_list = batch
        logits = self.forward(x)
        preds = logits.argmax(dim=1)

        pred_list = preds.cpu().tolist()

        results = []
        for pred, path in zip(pred_list, path_list):
            results.append({
                "image_path": path,
                "pred_class": pred
            })

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)