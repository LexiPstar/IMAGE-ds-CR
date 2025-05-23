import json
from collections import Counter
from nltk.tokenize import wordpunct_tokenize

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>":0, "<start>":1, "<end>":2, "<unk>":3}
        self.idx2word = {0:"<pad>", 1:"<start>", 2:"<end>", 3:"<unk>"}
        self.freqs = Counter()
        self.idx = 4

    def build_vocab(self, sentence_list):
        for sentence in sentence_list:
            tokens = wordpunct_tokenize(sentence.lower())
            self.freqs.update(tokens)  # 统计词频
        for word, freq in self.freqs.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        print(f"Vocabulary built with {len(self.word2idx)} words.")

    def numericalize(self, text):
        tokenized = wordpunct_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokenized]

    def save_vocab(self, filepath):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.word2idx, f)
        except Exception as e:
            print(f"Failed to save vocab: {e}")

    def load_vocab(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self.word2idx = json.load(f)
            self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
            self.idx = max(self.idx2word.keys()) + 1
            print(f"Vocabulary loaded with {len(self.word2idx)} words.")
        except Exception as e:
            print(f"Failed to load vocab: {e}")
