import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import os
from data.dataset import OptimizedFlickr8kDataset
from data.vocab import EnhancedVocabulary
from models.model import CaptionModel
from utils import save_checkpoint, load_checkpoint, EarlyStopping


def collate_fn(batch):
    """改进的批处理函数"""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, lengths = zip(*batch)
    
    images = torch.stack(images, 0)
    lengths = torch.LongTensor(lengths)
    
    # 填充描述到相同长度
    padded_captions = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
        
    return images, padded_captions, lengths

class Trainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化wandb（可选）
        if self.config.get('use_wandb', False):
            wandb.init(project="image-captioning", config=self.config)
            
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
    def setup_data(self):
        """设置数据加载器"""
        # 构建词汇表
        vocab = EnhancedVocabulary(freq_threshold=self.config['freq_threshold'])
        
        # 如果词汇表存在则加载，否则构建
        vocab_path = self.config.get('vocab_path', 'vocab.json')
        try:
            vocab.load_vocab(vocab_path)
        except:
            print("构建新词汇表...")
            with open(self.config['captions_file'], 'r', encoding='utf-8') as f:
                captions_list = [line.strip().split(',', 1)[1] for line in f.readlines()[1:]]
            vocab.build_vocab(captions_list)
            vocab.save_vocab(vocab_path)
            
        self.vocab = vocab
        
        # 创建数据集
        dataset = OptimizedFlickr8kDataset(
            self.config['images_path'], 
            self.config['captions_file'], 
            vocab, 
            mode='train'
        )
        
        # 划分训练和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
    def setup_model(self):
        """设置模型"""
        self.model = CaptionModel(
            embed_size=self.config['embed_size'],
            hidden_size=self.config['hidden_size'],
            vocab_size=len(self.vocab.word2idx),
            attention_dim=self.config.get('attention_dim', 512),
            train_CNN=self.config['train_CNN']
        ).to(self.device)
        
        # 损失函数（忽略填充标记）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def setup_training(self):
        """设置训练参数"""
        # 分层学习率
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        self.optimizer = optim.Adam([
            {'params': encoder_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': decoder_params, 'lr': self.config['learning_rate']}
        ])
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 早停机制
        self.early_stopping = EarlyStopping(patience=7, verbose=True)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, captions, lengths) in enumerate(pbar):
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)
            
            # 前向传播
            predictions, _, decode_lengths, alphas, _ = self.model(images, captions, lengths)
            
            # 计算损失
            targets = captions[:, 1:]  # 移除<start>标记
            loss = 0
            for i in range(len(decode_lengths)):
                loss += self.criterion(predictions[i, :decode_lengths[i]], targets[i, :decode_lengths[i]])
            loss /= len(decode_lengths)
            
            # 添加正则化项
            if self.config.get('alpha_c', 0) > 0:
                alpha_reg = self.config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()
                loss += alpha_reg
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 记录到wandb
            if self.config.get('use_wandb', False):
                wandb.log({'train_loss': loss.item(), 'step': epoch * len(self.train_loader) + batch_idx})
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)
                
                predictions, _, decode_lengths, _, _ = self.model(images, captions, lengths)
                
                targets = captions[:, 1:]
                loss = 0
                for i in range(len(decode_lengths)):
                    loss += self.criterion(predictions[i, :decode_lengths[i]], targets[i, :decode_lengths[i]])
                loss /= len(decode_lengths)
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        """主训练循环"""
        best_val_loss = float('inf')
        start_epoch = 0
        
        # 尝试加载检查点
        checkpoint_path = self.config.get('checkpoint_path', 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            try:
                start_epoch = load_checkpoint(self.model, self.optimizer, checkpoint_path)
                print(f"从epoch {start_epoch}继续训练")
            except:
                print("加载检查点失败，从头开始训练")
        
        for epoch in range(start_epoch, self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, 'best_model.pth', val_loss)
                print(f'保存最佳模型，验证损失: {val_loss:.4f}')
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                save_checkpoint(self.model, self.optimizer, epoch, checkpoint_path, val_loss)
            
            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("早停触发，训练结束")
                break
                
            # 记录到wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss_epoch': train_loss,
                    'val_loss_epoch': val_loss,
                    'epoch': epoch
                })
        
        print("训练完成！")

if __name__ == '__main__':
    trainer = Trainer('config.yaml')
    trainer.train()