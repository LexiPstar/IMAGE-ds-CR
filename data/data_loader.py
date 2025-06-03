# data/data_loader.py
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from .dataset import ImageCaptionDataset
from .vocabulary import Vocabulary
import torchvision.transforms as transforms


class collate_fn:
    def __init__(self, pad_idx):
        """
        初始化 collate 函数
        :param pad_idx: 填充标记的索引
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        对一个批次的数据进行处理
        :param batch: 一个批次的数据
        :return: 处理后的图像张量、描述张量和描述长度列表
        """
        # 分离图像和描述
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # 将图像堆叠成一个张量
        images = torch.stack(images)  # [B, C, H, W]

        # 对描述进行填充处理
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        # 计算每个描述的长度
        lengths = [len(cap) for cap in captions]

        return images, captions, lengths


def get_loader(image_folder, captions_file, batch_size=32, freq_threshold=5, shuffle=True, num_workers=0):
    """
    获取数据加载器
    :param image_folder: 图像数据所在的文件夹路径
    :param captions_file: 图像描述文件的路径
    :param batch_size: 每个批次的样本数量
    :param freq_threshold: 词汇频率阈值，低于该阈值的词汇将被忽略
    :param shuffle: 是否打乱数据
    :param num_workers: 数据加载的线程数
    :return: 数据加载器、数据集和词汇表
    """
    # 定义图像预处理的转换操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 读取图像描述文件
    df = pd.read_csv(captions_file)
    # 初始化词汇表
    vocab = Vocabulary(freq_threshold)
    # 构建词汇表
    vocab.build_vocabulary(df['caption'].tolist())

    # 创建图像描述数据集
    dataset = ImageCaptionDataset(
        root_dir=image_folder,
        captions_file=captions_file,
        vocab=vocab,
        transform=transform
    )

    # 获取填充标记的索引
    pad_idx = vocab.stoi[Vocabulary.PAD_TOKEN]

    # 创建数据加载器
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn(pad_idx)
    )

    return loader, dataset, vocab