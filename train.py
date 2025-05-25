import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.dataset import Flickr8kDataset
from data.vocab import Vocabulary
from models.model import CaptionModel
from utils import save_checkpoint, load_checkpoint

nltk.download('punkt')


def collate_fn(batch):
    images = []
    captions = []
    lengths = []
    for img, cap in batch:
        images.append(img.unsqueeze(0))
        captions.append(cap)
        lengths.append(len(cap))
    images = torch.cat(images, 0)
    captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, lengths


def train():
    # 读取配置
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # 词汇表准备
    with open(config['captions_file'], 'r') as f:
        captions_list = [line.strip().split(',')[1] for line in f.readlines()]

    vocab = Vocabulary(freq_threshold=config['freq_threshold'])
    vocab.build_vocab(captions_list)

    # 保存词汇表以供后续使用
    vocab.save_vocab('vocab.json')

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = Flickr8kDataset(config['images_path'], config['captions_file'], vocab, transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = CaptionModel(config['embed_size'], config['hidden_size'], len(vocab.word2idx),
                         train_CNN=config['train_CNN']).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    epochs = config['epochs']
    best_loss = float('inf')
    start_epoch = 0

    # 尝试加载检查点
    try:
        start_epoch = load_checkpoint(model, optimizer, config['checkpoint_path'])
        print(f"从epoch {start_epoch}继续训练")
    except FileNotFoundError:
        print("未找到检查点，从头开始训练")

    # 初始化日志文件
    log_path = config.get("loss_log_path", "loss_log.txt")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("")  # 清空日志文件

    model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, (imgs, captions, lengths) in loop:
            imgs, captions = imgs.to(device), captions.to(device)

            # 前向传播
            outputs = model(imgs, captions)

            # 计算损失
            targets = captions[:, 1:]
            outputs = outputs[:, :-1, :]
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.4f}")

        # 记录损失
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.4f}\n")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, "best_model.pth")
            print(f"保存最佳模型，损失: {best_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, config['checkpoint_path'])

        # 更新学习率
        scheduler.step()

    print("训练完成！")


if __name__ == '__main__':
    train()