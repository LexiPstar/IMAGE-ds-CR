import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        # 使用新版加载预训练权重
        weights = ResNet50_Weights.DEFAULT if not train_CNN else None
        resnet = models.resnet50(weights=weights)

        for param in resnet.parameters():
            param.requires_grad = train_CNN  # 冻结或解冻 ResNet 权重

        modules = list(resnet.children())[:-1]  # 去掉最后一个全连接层
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()
        features = self.fc(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        """生成预测序列，返回 ID list"""
        output_ids = []
        inputs = features.unsqueeze(1)

        states = None
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            output_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)

        return output_ids
