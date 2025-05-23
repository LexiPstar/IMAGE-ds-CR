import torch
import torch.nn as nn
from .encoder import EncoderCNN
from .decoder import DecoderRNN

class CaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, train_CNN=False):
        super().__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            feature = self.encoder(image).unsqueeze(0)
            sampled_ids = []
            inputs = feature
            states = None
            for _ in range(max_len):
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)
                hiddens, states = self.decoder.lstm(inputs, states)
                outputs = self.decoder.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1)
                sampled_ids.append(predicted.item())
                if predicted == vocab.word2idx['<end>']:
                    break
                inputs = self.decoder.embed(predicted).unsqueeze(1)
            caption = []
            for idx in sampled_ids:
                word = vocab.idx2word.get(idx, '<unk>')
                if word == '<end>':
                    break
                caption.append(word)
            return ' '.join(caption)
