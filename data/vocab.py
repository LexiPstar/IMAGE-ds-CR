import json
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import re

class EnhancedVocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.freqs = Counter()
        self.idx = 4
        
        # 下载必要的nltk数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def preprocess_text(self, text):
        """文本预处理"""
        # 转小写
        text = text.lower()
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        # 处理标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    def build_vocab(self, sentence_list):
        """构建词汇表"""
        print("构建词汇表...")
        for sentence in sentence_list:
            processed_sentence = self.preprocess_text(sentence)
            tokens = word_tokenize(processed_sentence)
            self.freqs.update(tokens)
            
        # 添加高频词到词汇表
        for word, freq in self.freqs.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
                
        print(f"词汇表构建完成，共 {len(self.word2idx)} 个词")
        print(f"词频阈值: {self.freq_threshold}")

    def numericalize(self, text):
        """将文本转换为数字序列"""
        processed_text = self.preprocess_text(text)
        tokenized = word_tokenize(processed_text)
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokenized]

    def save_vocab(self, filepath):
        """保存词汇表"""
        try:
            vocab_data = {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'freqs': dict(self.freqs),
                'freq_threshold': self.freq_threshold,
                'vocab_size': len(self.word2idx)
            }
            
            if filepath.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(vocab_data, f)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            print(f"词汇表已保存到: {filepath}")
        except Exception as e:
            print(f"保存词汇表失败: {e}")

    def load_vocab(self, filepath):
        """加载词汇表"""
        try:
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    vocab_data = pickle.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                    
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
            self.freqs = Counter(vocab_data.get('freqs', {}))
            self.freq_threshold = vocab_data.get('freq_threshold', 5)
            self.idx = len(self.word2idx)
            
            print(f"词汇表加载成功，共 {len(self.word2idx)} 个词")
        except Exception as e:
            print(f"加载词汇表失败: {e}")
            raise e