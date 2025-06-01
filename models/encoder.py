import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 移除最后两层以保留空间特征
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        for param in self.resnet.parameters():
            param.requires_grad = train_CNN
            
        # 自适应池化确保输出尺寸一致
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        
        # 投影到嵌入维度
        self.embed = nn.Linear(2048, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        # 提取空间特征
        features = self.resnet(images)  # (batch, 2048, 7, 7)
        features = self.adaptive_pool(features)  # (batch, 2048, 14, 14)
        
        batch_size = features.size(0)
        # 重塑用于注意力：(batch, 196, 2048)
        features = features.view(batch_size, 2048, -1).permute(0, 2, 1)
        
        # 投影到嵌入空间
        features = self.embed(features)  # (batch, 196, embed_size)
        
        # 应用批归一化和dropout
        features = features.view(-1, features.size(-1))
        features = self.batch_norm(features)
        features = features.view(batch_size, -1, features.size(-1))
        features = self.dropout(features)
        
        return features