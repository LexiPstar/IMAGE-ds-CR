import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import Flickr8kDataset
from data.vocab import Vocabulary
from models.model import CaptionModel
from utils import load_checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
import numpy as np

nltk.download('punkt')


def evaluate_model(model, dataloader, vocab, device, num_samples=1000):
    """评估模型性能"""
    model.eval()
    bleu_scores = []
    smoothie = SmoothingFunction().method4

    print(f"评估 {min(num_samples, len(dataloader))} 个样本...")

    with torch.no_grad():
        for i, (images, captions, _) in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break

            images = images.to(device)

            for j in range(images.size(0)):
                # 生成描述
                generated_caption = generate_single_caption(
                    model, vocab, images[j:j + 1], device
                )

                # 获取真实描述
                real_caption = []
                for idx in captions[j]:
                    if idx.item() == vocab.word2idx.get('<end>', 2):
                        break
                    if idx.item() not in [vocab.word2idx.get('<start>', 1),
                                          vocab.word2idx.get('<pad>', 0)]:
                        word = vocab.idx2word.get(idx.item(), '<unk>')
                        real_caption.append(word)

                # 计算BLEU分数
                if generated_caption and real_caption:
                    generated_tokens = generated_caption.split()
                    reference_tokens = [real_caption]  # BLEU需要参考句子列表

                    bleu = sentence_bleu(
                        reference_tokens,
                        generated_tokens,
                        smoothing_function=smoothie
                    )
                    bleu_scores.append(bleu)

    return np.mean(bleu_scores), bleu_scores


def generate_single_caption(model, vocab, image, device, max_len=20):
    """为单张图像生成描述"""
    with torch.no_grad():
        features = model.encoder(image)
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_len):
            hiddens, states = model.decoder.lstm(inputs, states)
            outputs = model.decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())

            if predicted.item() == vocab.word2idx.get('<end>', 2):
                break

            inputs = model.decoder.embed(predicted).unsqueeze(1)

    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word.get(word_id, '<unk>')
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption.append(word)

    return ' '.join(caption)


def main():
    # 读取配置
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # 加载词汇表
    vocab = Vocabulary(freq_threshold=config['freq_threshold'])
    try:
        vocab.load_vocab('vocab.json')
    except:
        # 如果没有保存的词汇表，重新构建
        with open(config['captions_file'], 'r') as f:
            captions_list = [line.strip().split(',')[1] for line in f.readlines()]
        vocab.build_vocab(captions_list)

    # 数据加载（使用验证集或测试集）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = Flickr8kDataset(config['images_path'], config['captions_file'], vocab, transform)
    # 这里可能需要分割数据集，暂时使用全集的子集
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = CaptionModel(
        config['embed_size'],
        config['hidden_size'],
        len(vocab.word2idx),
        train_CNN=False
    ).to(device)

    try:
        load_checkpoint(model, None, 'best_model.pth')
    except:
        print("警告: 无法加载最佳模型，使用默认检查点")
        load_checkpoint(model, None, config['checkpoint_path'])

    # 评估模型
    avg_bleu, bleu_scores = evaluate_model(model, dataloader, vocab, device, num_samples=500)

    print(f"\n评估结果:")
    print(f"平均BLEU分数: {avg_bleu:.4f}")
    print(f"BLEU分数标准差: {np.std(bleu_scores):.4f}")
    print(f"最高BLEU分数: {np.max(bleu_scores):.4f}")
    print(f"最低BLEU分数: {np.min(bleu_scores):.4f}")

    # 显示一些示例
    print("\n生成示例:")
    model.eval()
    for i, (images, captions, _) in enumerate(dataloader):
        if i >= 5:  # 只显示5个示例
            break

        images = images.to(device)
        generated = generate_single_caption(model, vocab, images, device)

        # 获取真实描述
        real_caption = []
        for idx in captions[0]:
            if idx.item() == vocab.word2idx.get('<end>', 2):
                break
            if idx.item() not in [vocab.word2idx.get('<start>', 1),
                                  vocab.word2idx.get('<pad>', 0)]:
                word = vocab.idx2word.get(idx.item(), '<unk>')
                real_caption.append(word)

        print(f"\n示例 {i + 1}:")
        print(f"真实描述: {' '.join(real_caption)}")
        print(f"生成描述: {generated}")


if __name__ == '__main__':
    main()