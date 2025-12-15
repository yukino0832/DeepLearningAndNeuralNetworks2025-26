import lightning
import torch
import argparse
from data.dataset import MyDataset
from model.lightningmodel import SentimentAnalysisModel
from model.BidLSTM import BidLSTM

def get_args_parser():
    parser = argparse.ArgumentParser('SentimentAnalysis', add_help=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model', default='BidLSTM', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_path', default='lightning_logs', type=str)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'inference'])
    parser.add_argument('--data_path', default=None, type=str, help='Path to the dataset needed for inference')
    parser.add_argument('--max_len', default=500, type=int, help='Maximum length of the text sequence')
    parser.add_argument('--glove_path', default=None, type=str, help='Path to the glove file')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    return parser

def train(args):
    full_dataset = MyDataset(args)
    val_size = int(len(full_dataset) * 0.3)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = SentimentAnalysisModel(args, full_dataset.word2idx, full_dataset.embeddings)

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
    predict_dataset = MyDataset(args)
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SentimentAnalysisModel(args, predict_dataset.word2idx, predict_dataset.embeddings)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device)["state_dict"])

    trainer = lightning.pytorch.Trainer(
        devices=1,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        default_root_dir=args.output_path
    )

    result = trainer.predict(model, predict_loader)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentimentAnalysis', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)