import lightning
import torch
import re
from nltk.translate.bleu_score import corpus_bleu
import argparse
from data.dataset import MyDataset
import utils
from model.transformer import TransformerModel
from model.lightningmodel import LightningTransformer

def get_args_parser():
    parser = argparse.ArgumentParser('TranslateGer2En', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model', default='Transformer', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--output_path', default='lightning_logs', type=str)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'inference', 'calculate_bleu'])
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--sentence_need_translate', default=None, type=str)
    return parser

def inference(model, sentence, de_vocab, en_vocab, device, max_len=32):
    model.eval()
    if isinstance(sentence, str):
        tokens = [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', sentence.rstrip())]
    else:
        tokens = [token.lower() for token in sentence]

    if len(tokens) > max_len - 2:
        src_len = max_len
        tokens = ['<bos>'] + tokens[:max_len - 2] + ['<eos>']
    else:
        src_len = len(tokens) + 2
        tokens = ['<bos>'] + tokens + ['<eos>'] + ['<pad>'] * (max_len - src_len)

    indexes = de_vocab.encode(tokens)
    enc_inputs = torch.tensor(indexes).long().unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode
        memory, src_key_padding_mask = model.encode(enc_inputs, src_pad_idx=de_vocab.pad_idx)
        
        # Decode
        dec_inputs = torch.tensor([[en_vocab.bos_idx]], device=device).long()
        
        for _ in range(max_len):
            predictions = model.decode(dec_inputs, memory, src_key_padding_mask, trg_pad_idx=en_vocab.pad_idx)
            dec_logits = predictions
            
            # Helper to get last token
            last_token_logits = dec_logits[0, -1, :]
            pred_token = last_token_logits.argmax(dim=0).item()
            
            dec_inputs = torch.cat([dec_inputs, torch.tensor([[pred_token]], device=device)], dim=1)
            
            if pred_token == en_vocab.eos_idx:
                break
                
    trg_indexes = dec_inputs[0].tolist()
    
    try:
        eos_idx = trg_indexes.index(en_vocab.eos_idx)
        trg_tokens = en_vocab.decode(trg_indexes[1:eos_idx])
    except ValueError:
        trg_tokens = en_vocab.decode(trg_indexes[1:])
        
    return trg_tokens

def calculate_bleu(dataset, model, de_vocab, en_vocab, device, max_len=50):
    trgs = []
    pred_trgs = []
    
    model.eval()
    for i, data in enumerate(dataset):
        # Taking only first 10 for demonstration as per notebook or full? 
        # Notebook says "for data in dataset[:10]:" but then prints "BLEU score = 42.06" which implies full set?
        # Actually checking notebook code: `for data in dataset[:10]:` -> returns `corpus_bleu`
        # Wait, the notebook output block 2035 says "BLEU score = 42.06", but the code block 2040 uses `[:10]`.
        # Likely the 42.06 was from a full run or the user edited it.
        # But given constraints, I will follow the explicit code in the notebook component I read: `[:10]`.
        # But `[:10]` samples is too small for meaningful BLEU. 
        # However, the user request says "mimic notebook". I will stick to what the notebook snippet showed or typical behavior.
        # Let's use tqdm if available and run on full test set? Or just a subset as snippet?
        # The snippet definitely showed `[:10]`. I will comment this.
        if i >= 10: break # Matching notebook snippet limitation
        
        src = data[0]
        trg = data[1]

        pred_trg = inference(model.model, src, de_vocab, en_vocab, device, max_len)
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return corpus_bleu(trgs, pred_trgs)

def train(args):
    train_dataset = MyDataset(args, type="train")
    valid_dataset = MyDataset(args, type="valid")
    test_dataset = MyDataset(args, type="test")

    de_vocab, en_vocab = utils.build_vocabs(train_dataset)
    args.pad_idx = de_vocab.pad_idx

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: utils.collate_fn(b, de_vocab, en_vocab, max_len=32)
    )

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: utils.collate_fn(b, de_vocab, en_vocab, max_len=32)
    )

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: utils.collate_fn(b, de_vocab, en_vocab, max_len=32)
    )

    model = LightningTransformer(args, de_vocab, en_vocab)

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    trainer = lightning.pytorch.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_path,
    )

    if args.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)
        # Load best checkpoint for testing
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            model = LightningTransformer.load_from_checkpoint(best_model_path, args=args, de_vocab=de_vocab, en_vocab=en_vocab)

    elif args.mode == 'calculate_bleu':
        if args.checkpoint_path:
            print(f"Loading model from {args.checkpoint_path}")
            model = LightningTransformer.load_from_checkpoint(args.checkpoint_path, args=args, de_vocab=de_vocab, en_vocab=en_vocab)
        else:
            print("Warning: No checkpoint provided for inference, using untrained model.")

    # Test/Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    model.to(device)
    model.eval()
    
    print("Calculating BLEU score...")
    bleu_score = calculate_bleu(test_dataset, model, de_vocab, en_vocab, device)
    print(f'BLEU score = {bleu_score*100:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TranslateGer2En', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.mode == 'train' or args.mode == 'calculate_bleu':
        train(args)
    elif args.mode == 'inference':
        train_dataset = MyDataset(args, type="train")
        de_vocab, en_vocab = utils.build_vocabs(train_dataset)
        args.pad_idx = de_vocab.pad_idx
        model = LightningTransformer.load_from_checkpoint(
                args.checkpoint_path,
                args=args,
                de_vocab=de_vocab,
                en_vocab=en_vocab
            )
    
        device = torch.device(
            'cuda' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
        )
        model.to(device)
        model.eval()

        translation = inference(
            model.model,
            args.sentence_need_translate,
            de_vocab,
            en_vocab,
            device,
            max_len=32
        )

        print("German :", args.sentence_need_translate)
        print("English:", " ".join(translation))
