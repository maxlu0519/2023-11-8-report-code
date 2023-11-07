# 数据集的设立
from torch.utils.data import Dataset
from PIL import Image

import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)  # 连接路径

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "cats_and_dogs_v2/train"
cats_label_dir = "cats"
dogs_label_dir = "dogs"
cats_dataset = MyData(root_dir, cats_label_dir)
dogs_dataset = MyData(root_dir, dogs_label_dir)

img, label = cats_dataset[0]
img.show()
