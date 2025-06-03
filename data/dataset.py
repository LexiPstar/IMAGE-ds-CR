"""
数据集模块
定义图像描述数据集类
"""
# data/dataset.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.vocabulary import Vocabulary


class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.vocab = vocab

        self.images = self.df['image']
        self.captions = self.df['caption']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        caption = self.captions[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        numericalized = [self.vocab.stoi[Vocabulary.START_TOKEN]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi[Vocabulary.END_TOKEN])

        return image, torch.tensor(numericalized)
