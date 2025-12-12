import lightning
import torchmetrics
import torch
from model.CNN import CNN
from model.ResNet import ResNet
import os
import json

class FlowersDetectModel(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.category_names = args.category_names
        self.num_classes = len(self.category_names)
        if args.model == 'CNN':
            self.model = CNN(self.num_classes)
        elif args.model == 'ResNet':
            self.model = ResNet(self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        self.lr = args.lr
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_results = []
        self.output_path = args.output_path

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.val_acc(torch.argmax(logits, dim=1), y)
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
                "pred_class": self.category_names[int(pred)]
            })

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)