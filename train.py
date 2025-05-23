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
from utils import save_checkpoint

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

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    dataset = Flickr8kDataset(config['images_path'], config['captions_file'], vocab, transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CaptionModel(config['embed_size'], config['hidden_size'], len(vocab.word2idx), train_CNN=config['train_CNN']).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    epochs = config['epochs']
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (imgs, captions, lengths) in loop:
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs, captions)

            targets = captions[:, 1:]
            outputs = outputs[:, :-1, :]
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 这里只加一次！

            loop.set_postfix(loss=loss.item())
        # 在 train() 函数中，avg_loss = ... 那行之后添加：

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.4f}")
        # 在 train() 函数中，avg_loss = ... 那行之后添加：

        log_path = config.get("loss_log_path", "loss_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.4f}\n")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, "checkpoint.pth")


if __name__ == '__main__':
    train()
