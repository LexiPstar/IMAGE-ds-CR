import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    """
    图像编码器:使用预训练的CNN提取图像特征
    """
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        
        # 使用预训练的ResNet-50
        resnet = models.resnet50(pretrained=True)
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # 添加新的全连接层，将特征映射到嵌入空间
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        """提取图像特征"""
        # 设置是否训练CNN
        for param in self.resnet.parameters():
            param.requires_grad = self.train_cnn
            
        # 提取特征
        with torch.set_grad_enabled(self.train_cnn):
            features = self.resnet(images)
        
        # 调整特征形状
        features = features.reshape(features.size(0), -1)
        features = self.dropout(self.bn(self.fc(features)))
        
        return features

class DecoderRNN(nn.Module):
    """
    文本解码器：使用LSTM生成图像描述
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """训练阶段的前向传播"""
        # 嵌入词向量
        embeddings = self.dropout(self.embed(captions))
        
        # 将图像特征与词嵌入连接
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # 打包序列以处理变长序列
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        # LSTM前向传播
        hiddens, _ = self.lstm(packed)
        
        # 预测下一个词
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    def sample(self, features, states=None):
        """推理阶段：生成图像描述"""
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        
        for i in range(self.max_seq_length):
            # LSTM前向传播
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            
            # 预测下一个词
            predicted = outputs.argmax(dim=1)
            sampled_ids.append(predicted)
            
            # 将预测结果作为下一步的输入
            inputs = self.embed(predicted).unsqueeze(1)
        
        # 将采样的ID连接成序列
        sampled_ids = torch.stack(sampled_ids, dim=1)
        
        return sampled_ids

class ImageCaptioningModel(nn.Module):
    """
    完整的图像描述生成模型，结合编码器和解码器
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
        
    def forward(self, images, captions, lengths):
        """训练阶段的前向传播"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(self, image):
        """生成图像描述"""
        feature = self.encoder(image)
        sampled_ids = self.decoder.sample(feature)
        return sampled_ids