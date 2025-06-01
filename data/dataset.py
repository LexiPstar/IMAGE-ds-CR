import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import random

class OptimizedFlickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, mode='train'):
        self.root_dir = root_dir
        self.vocab = vocab
        self.mode = mode
        
        # 数据增强
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            self.transform = transform
            
        # 加载数据
        self.data = []
        try:
            df = pd.read_csv(captions_file)
            for _, row in df.iterrows():
                self.data.append({
                    'image': row['image'],
                    'caption': row['caption']
                })
        except:
            # 兼容原格式
            with open(captions_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append({
                        'image': row['image'],
                        'caption': row['caption']
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img_name = item['image']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # 如果图像加载失败，返回随机图像
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)

        caption = item['caption']
        
        # 数字化描述
        numericalized_caption = [self.vocab.word2idx.get("<start>", 1)]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx.get("<end>", 2))

        caption_tensor = torch.LongTensor(numericalized_caption)
        return image, caption_tensor, len(numericalized_caption)