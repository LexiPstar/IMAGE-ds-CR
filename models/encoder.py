from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.resnet.parameters():
            param.requires_grad = train_CNN

        # 把 resnet 的最后一层 fc 替换成 Identity，保留特征
        self.resnet.fc = nn.Identity()

        # 这里直接用 resnet 的 fc 输入特征数
        self.linear = nn.Linear(2048, embed_size)  # 2048 是 resnet50 fc 的输入特征数

    def forward(self, images):
        features = self.resnet(images)  # (batch, 2048)
        features = self.linear(features)  # (batch, embed_size)
        return features
