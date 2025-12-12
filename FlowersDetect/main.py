import lightning
import torch
import argparse
import utils
from model.lightningmodel import FlowersDetectModel
from model.CNN import CNN
from data.dataset import MyDataset

def get_args_parser():
    parser = argparse.ArgumentParser('FlowersDetect', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Per GPU batch size')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model', default='ResNet', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_path', default='lightning_logs', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--data_path', default=None, type=str, help='Path to the dataset needed for train or inference')
    parser.add_argument('--category_names', default=None, type=utils.str2list, help='Category names list')
    parser.add_argument('--input_size', default=100, type=int)
    return parser

def train(args):

    train_dataset = MyDataset(args, "train")
    val_dataset = MyDataset(args, "val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = FlowersDetectModel(args)

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_acc:.4f}",
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

    model = FlowersDetectModel(args)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device)["state_dict"])

    predict_dataset = MyDataset(args)
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = lightning.pytorch.Trainer(
        devices=1,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        default_root_dir=args.output_path
    )

    result = trainer.predict(model, predict_loader)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FlowersDetect', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)