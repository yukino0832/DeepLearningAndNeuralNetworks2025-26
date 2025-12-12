import torch
from torchvision import datasets, transforms
import os
from PIL import Image

def Get_Transforms(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform_train, transform_val

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, args, train_or_val=None):
        self.args = args
        if args.mode == 'train':
            self.data_list = self.get_label_lists(args.data_path, train_or_val)
            self.transform = Get_Transforms(args)[0] if train_or_val == "train" else Get_Transforms(args)[1]
        elif args.mode == 'inference':
            self.data_list = [{"image_path": img_path} for img_path in self.get_image_paths(args.data_path)]
            self.transform = Get_Transforms(args)[1]

    def get_image_paths(self, dir_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_paths = []
        for root, dirs, files in sorted(os.walk(dir_path)):
            for file in sorted(files):
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def get_label_lists(self, dir_path, train_or_val):
        category_names = self.args.category_names
        label_lists = []
        for root, dirs, files in sorted(os.walk(dir_path, followlinks=True)):
            for dir_name in sorted(dirs):
                if dir_name in category_names:
                    dir_path_one_class = os.path.join(root, dir_name)
                    if train_or_val == "train":
                        train_dir = os.path.join(dir_path_one_class, "train")
                        if os.path.isdir(train_dir):
                            label_lists.extend([{"image_path": image_path, "label" : category_names.index(dir_name)} for image_path in self.get_image_paths(train_dir)])
                        else:
                            print("数据集未划分train和val")
                            exit()
                    elif train_or_val == "val":
                        val_dir = os.path.join(dir_path_one_class, "val")
                        if os.path.isdir(val_dir):
                            label_lists.extend([{"image_path": image_path, "label" : category_names.index(dir_name)} for image_path in self.get_image_paths(val_dir)])
                        else:
                            print("数据集未划分train和val")
                            exit()
        return label_lists

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.args.mode == 'train':
            image_path, targets = self.data_list[index]['image_path'], self.data_list[index]['label']
            image = Image.open(image_path).convert('RGB')
            return self.transform(image), torch.tensor(int(targets))
        elif self.args.mode == 'inference':
            image_path = self.data_list[index]['image_path']
            image = Image.open(image_path).convert('RGB')
            return self.transform(image), image_path