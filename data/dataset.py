"""
数据集模块
定义图像描述数据集类
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    """图像描述数据集类"""
    
    def __init__(self, image_dir, caption_file, vocab, transform=None, max_seq_len=20, 
                 image_col='image_id', caption_col='caption'):
        """
        Args:
            image_dir: 图像文件夹路径
            caption_file: 描述文件路径 (CSV格式)
            vocab: 词汇表对象
            transform: 图像预处理变换
            max_seq_len: 最大序列长度
            image_col: CSV中图像ID列名
            caption_col: CSV中描述文本列名
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.image_col = image_col
        self.caption_col = caption_col
        
        # 加载CSV图像描述数据
        self.df = pd.read_csv(caption_file, encoding='utf-8')
        
        # 验证必要的列是否存在
        if image_col not in self.df.columns:
            raise ValueError(f"列 '{image_col}' 在CSV文件中不存在")
        if caption_col not in self.df.columns:
            raise ValueError(f"列 '{caption_col}' 在CSV文件中不存在")
        
        # 清理数据：去除空值
        self.df = self.df.dropna(subset=[image_col, caption_col])
        
        print(f"加载了 {len(self.df)} 个图像-描述对")
        print(f"唯一图像数量: {self.df[image_col].nunique()}")
        print(f"CSV列名: {list(self.df.columns)}")
        
        # 构建图像-描述对
        self.image_caption_pairs = []
        for _, row in self.df.iterrows():
            image_id = str(row[image_col])
            caption = str(row[caption_col])
            self.image_caption_pairs.append((image_id, caption))
    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_id, caption = self.image_caption_pairs[idx]
        
        # 加载图像
        image = self._load_image(image_id)
        
        if self.transform:
            image = self.transform(image)
        
        # 处理描述文本
        caption_tensor = self._encode_caption(caption)
        
        return image, caption_tensor
    
    def _load_image(self, image_id):
        """加载图像文件"""
        # 支持多种图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            # 如果找不到文件，尝试不加扩展名
            potential_path = os.path.join(self.image_dir, str(image_id))
            if os.path.exists(potential_path):
                image_path = potential_path
            else:
                raise FileNotFoundError(f"找不到图像文件: {image_id}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise RuntimeError(f"无法加载图像 {image_path}: {e}")
    
    def _encode_caption(self, caption):
        """编码描述文本"""
        tokens = self.vocab.tokenize(caption)
        caption_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        caption_tensor[0] = self.vocab.word2idx['<start>']
        
        for i, token in enumerate(tokens[:self.max_seq_len-2]):
            caption_tensor[i+1] = self.vocab.word2idx.get(token, self.vocab.word2idx['<unk>'])
        
        if len(tokens) < self.max_seq_len-2:
            caption_tensor[len(tokens)+1] = self.vocab.word2idx['<end>']
        else:
            caption_tensor[self.max_seq_len-1] = self.vocab.word2idx['<end>']
        
        return caption_tensor
    
    def get_sample(self, idx):
        """获取样本的详细信息（用于调试）"""
        image_id, caption = self.image_caption_pairs[idx]
        image = self._load_image(image_id)
        caption_tensor = self._encode_caption(caption)
        
        # 解码回文本用于验证
        decoded_caption = self.vocab.decode_caption(caption_tensor.tolist())
        
        return {
            'image_id': image_id,
            'original_caption': caption,
            'decoded_caption': decoded_caption,
            'caption_tensor': caption_tensor,
            'image_size': image.size
        }
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_samples': len(self),
            'unique_images': self.df[self.image_col].nunique(),
            'avg_caption_length': self.df[self.caption_col].str.len().mean(),
            'max_caption_length': self.df[self.caption_col].str.len().max(),
            'min_caption_length': self.df[self.caption_col].str.len().min(),
        }
        return info
    
    def print_dataset_info(self):
        """打印数据集信息"""
        info = self.get_dataset_info()
        print("=== 数据集信息 ===")
        print(f"总样本数: {info['total_samples']}")
        print(f"唯一图像数: {info['unique_images']}")
        print(f"平均描述长度: {info['avg_caption_length']:.1f} 字符")
        print(f"最长描述: {info['max_caption_length']} 字符")
        print(f"最短描述: {info['min_caption_length']} 字符")
        print("=" * 50)

class CollateFunction:
    """自定义批处理函数"""
    
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        """
        将批次数据进行填充和对齐
        """
        images, captions = zip(*batch)
        
        # 图像直接堆叠
        images = torch.stack(images, dim=0)
        
        # 描述文本已经在数据集中填充好了
        captions = torch.stack(captions, dim=0)
        
        # 计算每个序列的实际长度（用于训练时的损失计算）
        lengths = []
        for caption in captions:
            # 找到<end>标记的位置
            length = (caption != self.pad_idx).sum().item()
            lengths.append(length)
        
        return images, captions, lengths