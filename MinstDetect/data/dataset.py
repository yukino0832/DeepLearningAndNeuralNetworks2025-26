import torch
from torchvision import datasets, transforms
import os
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, source, args):
        self.args = args
        if args.mode == 'train':
            self.data_list = source
            self.transform = transforms.Compose([
                lambda x: x.unsqueeze(0),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif args.mode == 'inference':
            self.data_list = [{"image_path": img_path} for img_path in self.get_image_paths(args.data_path)]
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    def get_image_paths(self, dir_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_paths = []
        for root, dirs, files in sorted(os.walk(dir_path)):
            for file in sorted(files):
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.args.mode == 'train':
            return self.transform(self.data_list.data[index]), self.data_list.targets[index]
        elif self.args.mode == 'inference':
            image_path = self.data_list[index]['image_path']
            image = Image.open(image_path).convert('L')
            image = Image.eval(image, lambda x: 255 - x) # 输入图片是白底黑字才需要这步
            return self.transform(image), image_path
