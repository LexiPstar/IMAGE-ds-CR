import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import os
import time
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_dir='checkpoints'):
    """
    训练图像描述生成模型
    """
    # 创建保存检查点的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm显示进度条
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, captions, lengths in train_progress:
            # 将数据移到设备上
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images, captions, lengths)
            
            # 计算损失（不包括<start>标记）
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for images, captions, lengths in val_progress:
                # 将数据移到设备上
                images = images.to(device)
                captions = captions.to(device)
                lengths = lengths.to(device)
                
                # 前向传播
                outputs = model(images, captions, lengths)
                
                # 计算损失
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                loss = criterion(outputs, targets)
                
                # 累计损失
                val_loss += loss.item()
                val_progress.set_postfix({'loss': loss.item()})
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 打印统计信息
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Time: {elapsed_time:.2f}s')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved with val_loss: {val_loss:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    print('Training complete!')
    return model