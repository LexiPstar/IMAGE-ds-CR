# eval/evaluate.py

import os
import torch
from torchvision import transforms
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from models.model import EncoderCNN, DecoderRNN
from data.vocabulary import Vocabulary
from utils.utils import load_config, clean_caption


def load_image(image_path, transform=None):
    """
    加载图像并进行预处理
    :param image_path: 图像文件的路径
    :param transform: 图像预处理的转换操作
    :return: 处理后的图像张量
    """
    # 打开图像并转换为 RGB 格式
    image = Image.open(image_path).convert("RGB")
    if transform:
        # 应用预处理转换操作
        image = transform(image).unsqueeze(0)
    return image


def evaluate_single(image_path, encoder, decoder, vocab, transform, device, max_len=20):
    """
    对单张图像进行评估，生成描述
    :param image_path: 图像文件的路径
    :param encoder: 图像编码器模型
    :param decoder: 描述解码器模型
    :param vocab: 词汇表
    :param transform: 图像预处理的转换操作
    :param device: 计算设备（CPU 或 GPU）
    :param max_len: 生成描述的最大长度
    :return: 生成的描述字符串
    """
    # 加载并处理图像
    image = load_image(image_path, transform).to(device)
    with torch.no_grad():
        # 提取图像特征
        feature = encoder(image)
        # 生成描述的 ID 列表
        output_ids = decoder.sample(feature, max_len=max_len)

    # 将 ID 列表转换为单词列表，去除特殊标记
    caption = [vocab.idx2word[idx] for idx in output_ids if idx not in (vocab.word2idx["<pad>"], vocab.word2idx["<start>"], vocab.word2idx["<end>"])]

    return " ".join(caption)


def evaluate_folder(config_path):
    """
    对指定文件夹中的所有图像进行评估，计算平均 BLEU 分数
    :param config_path: 配置文件的路径
    """
    # 加载配置文件
    config = load_config(config_path)
    # 选择计算设备（GPU 优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载词汇表
    vocab = Vocabulary.load("data/vocab.pkl")

    # 初始化编码器和解码器模型
    encoder = EncoderCNN(config["model"]["embed_size"]).to(device)
    decoder = DecoderRNN(
        config["model"]["embed_size"],
        config["model"]["hidden_size"],
        len(vocab)
    ).to(device)

    # 加载模型的检查点
    checkpoint = torch.load(config["training"]["checkpoint_path"], map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    # 将模型设置为评估模式
    encoder.eval()
    decoder.eval()

    # 定义图像预处理的转换操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 获取参考描述文件和图像文件夹的路径
    references_file = config["eval"]["reference_captions"]
    image_folder = config["eval"]["image_folder"]

    # 读取参考描述文件的所有行
    with open(references_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_score = 0
    # 定义平滑函数，用于计算 BLEU 分数
    smooth = SmoothingFunction().method4
    for line in lines:
        # 解析图像文件名和参考描述
        img_name, ref_caption = line.strip().split("\t")
        # 构建图像文件的完整路径
        image_path = os.path.join(image_folder, img_name)
        # 对单张图像进行评估，生成描述
        pred_caption = evaluate_single(image_path, encoder, decoder, vocab, transform, device)

        # 计算当前图像的 BLEU 分数
        score = sentence_bleu(
            [clean_caption(ref_caption).split()],
            clean_caption(pred_caption).split(),
            smoothing_function = smooth
        )
        # 打印评估结果
        print(f"[{img_name}]\nPred: {pred_caption}\nRef:  {ref_caption}\nBLEU: {score:.4f}\n")
        total_score += score

    # 计算平均 BLEU 分数
    avg_bleu = total_score / len(lines)
    print(f"Average BLEU: {avg_bleu:.4f}")