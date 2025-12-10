import lightning
import torch
import argparse
from torchvision import datasets
from model.lightningmodel import MinstDetectModel
from model.FNN import FNN
from data.dataset import MyDataset

def get_args_parser():
    parser = argparse.ArgumentParser('MinstDetect', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int, help='Per GPU batch size')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model', default='FNN', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_path', default='lightning_logs', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--data_path', default=None, type=str, help='Path to the dataset needed for inference')
    return parser

def train(args):

    train_dataset = MyDataset(datasets.MNIST(root='data/datasets', train=True, download=True), args)
    val_dataset = MyDataset(datasets.MNIST(root='data/datasets', train=False, download=True), args)

    model = MinstDetectModel(args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = lightning.pytorch.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        default_root_dir=args.output_path,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

def inference(args):

    model = MinstDetectModel(args)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device)["state_dict"])

    predict_dataset = MyDataset(args.data_path, args)
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = lightning.pytorch.Trainer(
        devices=1,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        default_root_dir=args.output_path
    )

    result = trainer.predict(model, predict_loader)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MinstDetect', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
