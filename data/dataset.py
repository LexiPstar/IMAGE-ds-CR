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
        """
        初始化图像描述数据集
        :param root_dir: 图像数据所在的根目录
        :param captions_file: 图像描述文件的路径
        :param vocab: 词汇表
        :param transform: 图像预处理的转换操作
        """
        self.root_dir = root_dir
        # 读取图像描述文件
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.vocab = vocab

        # 获取图像文件名和描述
        self.images = self.df['image']
        self.captions = self.df['caption']

    def __len__(self):
        """
        返回数据集的长度
        :return: 数据集的长度
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        :param idx: 样本索引
        :return: 图像张量和描述的索引列表
        """
        # 构建图像文件的完整路径
        img_path = os.path.join(self.root_dir, self.images[idx])
        caption = self.captions[idx]

        # 打开图像并转换为 RGB 格式
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            # 应用预处理转换操作
            image = self.transform(image)

        # 将描述转换为索引列表，并添加开始和结束标记
        numericalized = [self.vocab.stoi[Vocabulary.START_TOKEN]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi[Vocabulary.END_TOKEN])

        return image, torch.tensor(numericalized)