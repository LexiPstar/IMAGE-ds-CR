import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        """
        初始化图像编码器
        :param embed_size: 嵌入向量的维度
        :param train_CNN: 是否训练 CNN 模型
        """
        super(EncoderCNN, self).__init__()
        # 根据是否训练 CNN 模型选择加载预训练权重
        weights = ResNet50_Weights.DEFAULT if not train_CNN else None
        # 加载预训练的 ResNet50 模型
        resnet = models.resnet50(weights=weights)

        # 冻结或解冻 ResNet 权重
        for param in resnet.parameters():
            param.requires_grad = train_CNN

        # 去掉最后一个全连接层
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # 定义全连接层，将特征向量映射到嵌入向量维度
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        # 定义批量归一化层
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        前向传播
        :param images: 输入的图像张量
        :return: 编码后的特征向量
        """
        with torch.no_grad():
            # 通过 ResNet 提取特征
            features = self.resnet(images).squeeze()
        # 通过全连接层和批量归一化层处理特征
        features = self.fc(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        初始化描述解码器
        :param embed_size: 嵌入向量的维度
        :param hidden_size: 隐藏层的维度
        :param vocab_size: 词汇表的大小
        :param num_layers: LSTM 层的数量
        """
        super(DecoderRNN, self).__init__()
        # 定义嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 定义 LSTM 层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，将隐藏层输出映射到词汇表大小
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        前向传播
        :param features: 图像编码后的特征向量
        :param captions: 输入的描述序列
        :return: 预测的描述序列
        """
        # 对描述序列进行嵌入处理，去掉最后一个标记
        embeddings = self.embed(captions[:, :-1])
        # 将图像特征和嵌入后的描述序列拼接
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        # 通过 LSTM 层处理输入
        hiddens, _ = self.lstm(inputs)
        # 通过全连接层输出预测结果
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        """
        生成预测序列，返回 ID list
        :param features: 图像编码后的特征向量
        :param max_len: 生成序列的最大长度
        :return: 生成的描述序列的 ID 列表
        """
        output_ids = []
        # 对特征向量进行维度扩展
        inputs = features.unsqueeze(1)

        states = None
        for _ in range(max_len):
            # 通过 LSTM 层处理输入
            hiddens, states = self.lstm(inputs, states)
            # 通过全连接层输出预测结果
            output = self.linear(hiddens.squeeze(1))
            # 选择概率最大的单词 ID
            predicted = output.argmax(1)
            # 将预测的单词 ID 添加到输出列表中
            output_ids.append(predicted.item())
            # 将预测的单词 ID 进行嵌入处理，作为下一次输入
            inputs = self.embed(predicted).unsqueeze(1)

        return output_ids