import os
import random
import numpy as np
import torch
from PIL import Image
import dataloaders.data_utils

class CGTrader3000Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir='./data/cgtrader/cgtrader-3000/images', tag_file='./data/cgtrader/cgtrader-3000/tags_needed.txt', label_file='./data/cgtrader/cgtrader-3000/data.csv', image_transform=None, num_known_labels=0, testing=False, training_ratio=0.95):
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.num_known_labels = num_known_labels
        self.testing = testing
        self.epoch = 1

        # 获取所有互不相同的标签，为每一个标签赋予从0开始递增的标签，并储存在self.category_info中。
        with open(tag_file) as file:
            lines = file.read().split('\n')[:-1]
            self.num_labels = len(lines)
            self.category_info = {}
            for i in range(self.num_labels):
                self.category_info[lines[i]] = i
            file.close()

        # 读取每张图片的文件名及其对应标签。
        # 标签采用维度为(num_labels,)的ndarray储存，其中值为1.0的位置为对应标签位置。
        self.img_names = []
        self.labels = []
        with open(label_file) as file:
            lines = file.read().split('\n')
            lines = lines[1:-1]
            random.shuffle(lines)
            num_samples = len(lines)
            num_samples_training = int(num_samples * training_ratio)
            if testing:
                lines = lines[num_samples_training:]
            else:
                lines = lines[0:num_samples_training]
            for line in lines:
                _, image_filename, tag_string, _ = line.split(',')
                self.img_names.append(image_filename)
                label_vector = np.zeros(self.num_labels, dtype=int)
                for tag in tag_string.split(';'):
                    label_vector[self.category_info[tag]] = 1
                self.labels.append(label_vector)
            file.close()

    # 获取一个sample。
    def __getitem__(self, index):
        name = self.img_names[index]
        image = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        
        """
        使用它处定义的image transformation模块对图片进行变换，变换操作可能包括：
        1. Resize
        2. Crop
        3. Normalization
        """
        if self.image_transform:
            image = self.image_transform(image)

        labels = torch.Tensor(self.labels[index])

        # 生成mask。mask的维度是(num_labels,)，值为-1的位置为unknown label的位置，为0的位置表示negative label的位置，为1的位置表示positive label的位置。
        mask = labels.clone()
        unk_mask_indices = dataloaders.data_utils.get_unk_mask_indices(image, self.testing, self.num_labels, self.num_known_labels, self.epoch)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': name,
        }

        return sample

    # 获取sample的总数。
    def __len__(self):
        return len(self.img_names)
