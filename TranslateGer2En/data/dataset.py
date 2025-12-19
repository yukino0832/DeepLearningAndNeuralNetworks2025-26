import torch
import os
import re

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, args, type="train"):
        if type == "train":
            self.data_list = self.get_data(args.data_path + "train/")
        elif type == "valid":
            self.data_list = self.get_data(args.data_path + "valid/")
        elif type == "test":
            self.data_list = self.get_data(args.data_path + "test/")

    def get_data(self, path):
        def tokenize(text):
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', text)]

        def read_data(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as data_file:
                data = data_file.readlines()[:-1]
                return [tokenize(i) for i in data]

        members = {i.split('.')[-1]: path + i for i in os.listdir(path)}
        ret = [read_data(members['de']), read_data(members['en'])]
        return list(zip(*ret))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]