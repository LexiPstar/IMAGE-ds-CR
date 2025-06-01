import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def evaluate_model(model, data_loader, criterion, device):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress = tqdm(data_loader, desc='Evaluating')
        
        for images, captions, lengths in progress:
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
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    print(f'Evaluation - Loss: {avg_loss:.4f}')
    
    return avg_loss

def evaluate_metrics(model, data_loader, vocab, device):
    """
    使用标准指标评估模型
    """
    model.eval()
    
    # 存储所有参考标题和生成的标题
    references = []
    hypotheses = []
    
    with torch.no_grad():
        progress = tqdm(data_loader, desc='Generating captions')
        
        for images, captions, _ in progress:
            # 将图像移到设备上
            images = images.to(device)
            
            # 生成标题
            sampled_ids = model.generate_caption(images)
            
            # 将索引转换为单词
            for i, sample in enumerate(sampled_ids):
                # 生成的标题
                hyp = []
                for idx in sample.cpu().numpy():
                    word = vocab.idx2word[idx]
                    if word == '<end>':
                        break
                    if word not in ['<start>', '<pad>', '<unk>']:
                        hyp.append(word)
                
                # 参考标题
                ref = []
                for idx in captions[i].cpu().numpy():
                    word = vocab.idx2word[idx]
                    if word == '<end>':
                        break
                    if word not in ['<start>', '<pad>', '<unk>']:
                        ref.append(word)
                
                # 添加到列表中
                hypotheses.append(hyp)
                references.append([ref])  # 每个参考是一个列表的列表
    
    # 计算BLEU分数
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # 为其他指标准备数据
    hyp_dict = {i: [' '.join(hyp)] for i, hyp in enumerate(hypotheses)}
    ref_dict = {i: [' '.join(ref[0])] for i, ref in enumerate(references)}
    
    # 计算CIDEr分数
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)
    
    # 计算METEOR分数
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(ref_dict, hyp_dict)
    
    # 计算ROUGE-L分数
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(ref_dict, hyp_dict)
    
    # 打印结果
    print(f'BLEU-1: {bleu1:.4f}')
    print(f'BLEU-4: {bleu4:.4f}')
    print(f'CIDEr: {cider_score:.4f}')
    print(f'METEOR: {meteor_score:.4f}')
    print(f'ROUGE-L: {rouge_score:.4f}')
    
    return {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'cider': cider_score,
        'meteor': meteor_score,
        'rouge': rouge_score
    }