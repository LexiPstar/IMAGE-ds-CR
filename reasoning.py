# 推理
import torch
import yaml
from PIL import Image
from torchvision import transforms
from data.vocab import Vocabulary
from models.model import CaptionModel
from utils import load_checkpoint
import argparse


def load_model_and_vocab(config_path, model_path, vocab_path):
    """加载模型和词汇表"""
    # 读取配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 加载词汇表
    vocab = Vocabulary(freq_threshold=config['freq_threshold'])
    vocab.load_vocab(vocab_path)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptionModel(
        config['embed_size'],
        config['hidden_size'],
        len(vocab.word2idx),
        train_CNN=False
    ).to(device)

    # 加载模型权重
    load_checkpoint(model, None, model_path)
    model.eval()

    return model, vocab, device


def preprocess_image(image_path):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def generate_caption(model, vocab, image, device, max_len=20):
    """生成图像描述"""
    image = image.to(device)

    with torch.no_grad():
        # 获取图像特征
        features = model.encoder(image)

        # 初始化序列
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        states = None

        for _ in range(max_len):
            # LSTM前向传播
            hiddens, states = model.decoder.lstm(inputs, states)
            outputs = model.decoder.linear(hiddens.squeeze(1))

            # 选择概率最高的词
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())

            # 如果遇到结束符，停止生成
            if predicted.item() == vocab.word2idx.get('<end>', 2):
                break

            # 下一次输入是当前预测的词的嵌入
            inputs = model.decoder.embed(predicted).unsqueeze(1)

    # 将ID转换为单词
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word.get(word_id, '<unk>')
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption.append(word)

    return ' '.join(caption)


def main():
    parser = argparse.ArgumentParser(description='图像描述生成推理')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', default='best_model.pth', help='模型文件路径')
    parser.add_argument('--vocab', default='vocab.json', help='词汇表文件路径')

    args = parser.parse_args()

    try:
        # 加载模型和词汇表
        print("加载模型和词汇表...")
        model, vocab, device = load_model_and_vocab(args.config, args.model, args.vocab)

        # 预处理图像
        print("预处理图像...")
        image = preprocess_image(args.image)

        # 生成描述
        print("生成描述...")
        caption = generate_caption(model, vocab, image, device)

        print(f"\n图像: {args.image}")
        print(f"生成的描述: {caption}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()