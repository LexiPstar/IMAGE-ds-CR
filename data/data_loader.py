import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nltk
from collections import Counter
import pickle

class Vocabulary:
    """
    词汇表类：处理文本标记化和词汇映射
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # 添加特殊标记
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    
    def tokenize(self, sentence):
        """将句子分词"""
        return nltk.tokenize.word_tokenize(str(sentence).lower())
    
    @staticmethod
    def build_vocab(json_file, threshold=5):
        """从标题文件构建词汇表"""
        counter = Counter()
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for item in data['annotations']:
            caption = item['caption']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            counter.update(tokens)
            
        # 过滤掉低频词
        words = [word for word, count in counter.items() if count >= threshold]
        
        # 创建词汇表
        vocab = Vocabulary()
        for word in words:
            vocab.add_word(word)
            
        return vocab
    
    def save(self, file_path):
        """保存词汇表到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        """从文件加载词汇表"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class CocoDataset(Dataset):
    """
    COCO数据集加载器
    """
    def __init__(self, root_dir, json_file, vocab, transform=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        
        # 加载标题数据
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # 组织图像和标题对
        self.images = []
        self.captions = []
        
        for item in self.data['annotations']:
            image_id = item['image_id']
            caption = item['caption']
            
            # 查找图像文件名
            image_path = None
            for img in self.data['images']:
                if img['id'] == image_id:
                    image_path = os.path.join(self.root_dir, img['file_name'])
                    break
            
            if image_path and os.path.exists(image_path):
                self.images.append(image_path)
                self.captions.append(caption)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        caption = self.captions[idx]
        
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理标题
        tokens = self.vocab.tokenize(caption)
        
        # 添加开始和结束标记
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        
        # 转换为张量
        caption = torch.tensor(caption, dtype=torch.long)
        
        return image, caption

def get_data_loader(root_dir, json_file, vocab, batch_size, transform, shuffle=True):
    """创建数据加载器"""
    dataset = CocoDataset(root_dir=root_dir,
                          json_file=json_file,
                          vocab=vocab,
                          transform=transform)
    
    # 自定义排序函数，按标题长度排序
    def collate_fn(data):
        """创建小批量数据"""
        # 按标题长度排序
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)
        
        # 堆叠图像
        images = torch.stack(images, 0)
        
        # 获取标题长度
        lengths = [len(cap) for cap in captions]
        
        # 创建填充的标题张量
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        
        # 转换为列表，因为pack_padded_sequence需要列表形式的长度
        lengths = torch.tensor(lengths)
        
        return images, targets, lengths
    
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)
    
    return data_loader