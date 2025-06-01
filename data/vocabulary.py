# data/vocabulary.py

import nltk
import pickle
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        """
        词汇表初始化，传入词频阈值，低于该频率的词会被过滤掉。
        """
        self.freq_threshold = freq_threshold

        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}  # id->word
        self.stoi = {v: k for k, v in self.itos.items()}                # word->id

    def tokenizer(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)  # 从4开始编号

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
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
