import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, filename, src_dir, tgt_dir, src_transform, tgt_transform):
        self.image_label_list = self.read_file(filename)
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.len = len(self.image_label_list)
        self.src_transform = src_transform
        self.tgt_transform = tgt_transform
    
    def __getitem__(self, i):
        index = i % self.len
        image_name , _ = self.image_label_list[index]#这里设置label为-1，因为是生成图像，因此不需要label
        src_path = os.path.join(self.src_dir, image_name)
        tgt_path = os.path.join(self.tgt_dir, image_name)
        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)

        #对图片进行预处理
        src_img = self.src_transform(src_img)
        tgt_img = self.tgt_transform(tgt_img)
        
        return src_img, tgt_img
    
    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split()
                name = content[0]
                label = content[1]
                image_label_list.append((name, label))
        return image_label_list