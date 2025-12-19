import torch
from collections import Counter, OrderedDict
import math

def collate_fn(batch, de_vocab, en_vocab, max_len=32):
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    src_batch, trg_batch = [], []
    src_lens = []

    for de, en in batch:
        de_idx = [de_vocab.bos_idx] + de_vocab.encode(de)[:max_len-2] + [de_vocab.eos_idx]
        en_idx = [en_vocab.bos_idx] + en_vocab.encode(en)[:max_len-2] + [en_vocab.eos_idx]

        src_pad = de_idx + [de_vocab.pad_idx] * (max_len - len(de_idx))
        trg_pad = en_idx + [en_vocab.pad_idx] * (max_len - len(en_idx))

        src_batch.append(src_pad)
        trg_batch.append(trg_pad)
        src_lens.append(len(de_idx))

    return (torch.LongTensor(src_batch), 
            torch.LongTensor(src_lens), 
            torch.LongTensor(trg_batch))

class Vocab:
    """构建词元与数字索引的映射字典"""
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=2):
        self.word2idx = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        
        # 过滤低频词并按词频排序
        for word, count in word_count_dict.items():
            if count >= min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.pad_idx = self.word2idx['<pad>']
        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.unk_idx = self.word2idx['<unk>']

    def encode(self, tokens):
        if isinstance(tokens, list):
            return [self.word2idx.get(t, self.unk_idx) for t in tokens]
        return self.word2idx.get(tokens, self.unk_idx)

    def decode(self, indices):
        if isinstance(indices, list):
            return [self.idx2word.get(i, '<unk>') for i in indices]
        return self.idx2word.get(indices, '<unk>')

    def __len__(self):
        return len(self.word2idx)

def build_vocabs(train_dataset):
    de_words, en_words = [], []
    for de, en in train_dataset:
        de_words.extend(de)
        en_words.extend(en)
    
    # 统计词频并排序
    de_count = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))
    
    return Vocab(de_count), Vocab(en_count)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Batch first: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)