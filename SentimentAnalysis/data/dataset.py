import os
import string
import torch
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.word2idx, self.embeddings = self.load_glove(args.glove_path)
        self.data_list = self.get_textIdx_label(os.path.join(args.data_path, "train")) if args.mode == "train" else [self.process_text(args.data_path)]

    def load_glove(self, glove_path):
        tokens = []
        vectors = []

        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(maxsplit=1)
                tokens.append(word)
                vectors.append(
                    np.fromstring(vec, sep=" ", dtype=np.float32)
                )

        vectors.append(np.random.rand(100))
        vectors.append(np.zeros(100, dtype=np.float32))

        word2idx = {word: idx for idx, word in enumerate(tokens)}
        word2idx["<unk>"] = len(word2idx)
        word2idx["<pad>"] = len(word2idx)

        embeddings = torch.tensor(np.array(vectors), dtype=torch.float32)

        return word2idx, embeddings

    def get_textIdx_label(self, data_dir):
        data_list = []
        for label_name, label_value in [("pos", 1), ("neg", 0)]:
            label_dir = os.path.join(data_dir, label_name)
            for fname in os.listdir(label_dir):
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(label_dir, fname)
                text_idx = self.process_text(path)
                data_list.append({"text_idx": text_idx, "label": label_value})
        return data_list

    def process_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.lower()
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )
        tokens = text.split()
        ids = [
            self.word2idx.get(token, self.word2idx["<unk>"])
            for token in tokens
        ]
        if len(ids) > self.args.max_len:
            ids = ids[:self.args.max_len]
        else:
            ids += [self.word2idx["<pad>"]] * (self.args.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.args.mode == 'train':
            text_idx, label = self.data_list[index]['text_idx'], self.data_list[index]['label']
            return text_idx, torch.tensor([label], dtype=torch.float32)
        elif self.args.mode == 'inference':
            return self.data_list[index]