# training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from data.vocabulary import Vocabulary
from models.model import EncoderCNN, DecoderRNN
from data.data_loader import get_loader

import logging

def setup_logger(log_path):
    """
    设置日志记录器
    :param log_path: 日志文件的保存路径
    """
    # 配置日志记录的基本信息
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format="%(asctime)s [%(levelname)s] %(message)s",  # 设置日志格式
        handlers=[
            logging.FileHandler(log_path),  # 将日志记录到文件中
            logging.StreamHandler()  # 将日志输出到控制台
        ]
    )

def train(config):
    """
    训练模型
    :param config: 配置字典
    """
    # 设置日志记录器
    setup_logger(config["training"]["log_path"])
    # 记录训练开始的信息
    logging.info("Training started.")

    # 选择计算设备（GPU 优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    loader, dataset, vocab = get_loader(
        image_folder=config["data"]["image_folder"],
        captions_file=config["data"]["captions_file"],
        batch_size=config["training"]["batch_size"],
        freq_threshold=config["data"]["freq_threshold"],
        shuffle=True
    )

    # 构建编码器和解码器模型
    encoder = EncoderCNN(config["model"]["embed_size"]).to(device)
    decoder = DecoderRNN(
        config["model"]["embed_size"],
        config["model"]["hidden_size"],
        len(vocab)
    ).to(device)

    # 定义损失函数和优化器
    pad_idx = vocab.stoi[Vocabulary.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 定义需要优化的参数
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=config["training"]["lr"])

    # 从检查点恢复训练
    start_epoch = 0
    if config["training"]["resume"] and os.path.exists(config["training"]["checkpoint_path"]):
        checkpoint = torch.load(config["training"]["checkpoint_path"])
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        # 记录恢复训练的信息
        logging.info(f"Resumed from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        # 将模型设置为训练模式
        encoder.train()
        decoder.train()

        # 使用 tqdm 显示训练进度
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{config['training']['num_epochs']}]")
        total_loss = 0

        for imgs, captions, lengths in loop:
            # 将数据移动到计算设备上
            imgs, captions = imgs.to(device), captions.to(device)

            # 提取图像特征
            features = encoder(imgs)
            # 生成描述
            outputs = decoder(features, captions)

            # 对描述进行填充处理，以便计算损失
            targets = pack_padded_sequence(captions[:, 1:], lengths=[l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
            outputs = pack_padded_sequence(outputs, lengths=[l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]

            # 计算损失
            loss = criterion(outputs, targets)

            # 清空优化器的梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            total_loss += loss.item()
            # 更新 tqdm 进度条的信息
            loop.set_postfix(loss=loss.item())

        # 保存模型的检查点
        os.makedirs(os.path.dirname(config["training"]["checkpoint_path"]), exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, config["training"]["checkpoint_path"])

        # 记录当前轮次的训练损失
        logging.info(f"Epoch {epoch + 1} completed, loss: {total_loss / len(loader):.4f}")