"""
词汇表模块
处理文本分词和词汇映射
"""
import pandas as pd
import pickle
from collections import Counter
from tqdm import tqdm
import re

class Vocabulary:
    """词汇表类"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 特殊标记
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def tokenize(self, caption):
        """分词函数"""
        # 简单的英文分词
        tokens = re.findall(r'\b\w+\b', caption.lower())
        return tokens
    
    def build_vocab(self, captions, min_count=2):
        """从描述列表构建词汇表"""
        print("正在构建词汇表...")
        
        # 统计词频
        for caption in tqdm(captions, desc="统计词频"):
            tokens = self.tokenize(caption)
            for token in tokens:
                self.word_count[token] += 1
        
        print(f"总词汇数: {len(self.word_count)}")
        
        # 添加高频词到词汇表
        for word, count in self.word_count.items():
            if count >= min_count:
                self.add_word(word)
        
        print(f"过滤后词汇表大小: {len(self.word2idx)}")
        print(f"最小词频: {min_count}")
    
    def build_vocab_from_csv(self, csv_file, caption_col='caption', min_count=2):
        """从CSV文件构建词汇表"""
        df = pd.read_csv(csv_file, encoding='utf-8')
        if caption_col not in df.columns:
            raise ValueError(f"列 '{caption_col}' 在CSV文件中不存在")
        
        captions = df[caption_col].dropna().astype(str).tolist()
        self.build_vocab(captions, min_count)
    
    def encode_caption(self, caption, max_length):
        """将描述文本编码为索引序列"""
        tokens = self.tokenize(caption)
        encoded = [self.word2idx['<start>']]
        
        for token in tokens[:max_length-2]:
            encoded.append(self.word2idx.get(token, self.word2idx['<unk>']))
        
        encoded.append(self.word2idx['<end>'])
        
        # 填充到指定长度
        while len(encoded) < max_length:
            encoded.append(self.word2idx['<pad>'])
        
        return encoded[:max_length]
    
    def decode_caption(self, indices):
        """将索引序列解码为文本"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<unk>')
            if word in ['<start>', '<pad>']:
                continue
            if word == '<end>':
                break
            words.append(word)
        
        return ' '.join(words)
    
    def save(self, filepath):
        """保存词汇表"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"词汇表已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"词汇表已从 {filepath} 加载")
        print(f"词汇表大小: {len(vocab)}")
        return vocab
    
    def get_vocab_info(self):
        """获取词汇表信息"""
        info = {
            'vocab_size': len(self.word2idx),
            'total_words': sum(self.word_count.values()),
            'unique_words': len(self.word_count),
            'most_common': self.word_count.most_common(10),
            'special_tokens': ['<pad>', '<start>', '<end>', '<unk>']
        }
        return info
    
    def print_vocab_info(self):
        """打印词汇表信息"""
        info = self.get_vocab_info()
        print("=== 词汇表信息 ===")
        print(f"词汇表大小: {info['vocab_size']}")
        print(f"总词数: {info['total_words']}")
        print(f"唯一词数: {info['unique_words']}")
        print(f"最常见词汇: {info['most_common']}")
        print(f"特殊标记: {info['special_tokens']}")
        print("=" * 50)
    
    def __len__(self):
        return len(self.word2idx)
    
    def __contains__(self, word):
        return word in self.word2idx
    
    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])