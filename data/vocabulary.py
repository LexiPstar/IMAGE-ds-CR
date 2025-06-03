# data/vocabulary.py

import nltk
import pickle
from collections import Counter

class Vocabulary:
    PAD_TOKEN = "<PAD>"  # 填充标记
    START_TOKEN = "<START>"  # 句子开始标记
    END_TOKEN = "<END>"  # 句子结束标记
    UNK_TOKEN = "<UNK>"  # 未知单词标记

    def __init__(self, freq_threshold=5):
        """
        初始化词汇表
        :param freq_threshold: 词汇频率阈值，低于该阈值的词汇将被忽略
        """
        self.itos = {
            0: self.PAD_TOKEN,
            1: self.START_TOKEN,
            2: self.END_TOKEN,
            3: self.UNK_TOKEN,
        }  # 索引到单词的映射
        self.stoi = {v: k for k, v in self.itos.items()}  # 单词到索引的映射
        self.freq_threshold = freq_threshold  # 词汇频率阈值
        self.word_freq = Counter()  # 词汇频率计数器
        self.index = 4  # 下一个可用的索引

    def tokenizer(self, text):
        """
        对文本进行分词
        :param text: 待分词的文本
        :return: 分词后的单词列表
        """
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        """
        根据句子列表构建词汇表
        :param sentence_list: 句子列表
        """
        frequencies = Counter()
        idx = len(self.itos)  # 从 4 开始编号

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

        # 过滤掉低频词，加入词表
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """
        将文本转换为索引列表
        :param text: 待转换的文本
        :return: 索引列表
        """
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

    def __len__(self):
        """
        返回词汇表的长度
        :return: 词汇表的长度
        """
        return len(self.itos)

    def save(self, path):
        """
        将词汇表保存到文件中
        :param path: 文件路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        从文件中加载词汇表
        :param path: 文件路径
        :return: 加载的词汇表对象
        """
        with open(path, 'rb') as f:
            return pickle.load(f)