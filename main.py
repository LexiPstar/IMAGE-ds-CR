import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import os
from models import ImageCaptioningModel
from data_loader import Vocabulary, get_data_loader
from training.train import train_model
from eval.evaluate import evaluate_model, evaluate_metrics
from inference import generate_caption, visualize_caption, load_image

def main():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'infer'],
                        help='运行模式: train, eval, infer')
    parser.add_argument('--data_dir', type=str, default='data/coco',
                        help='数据目录')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='词汇表路径')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--image_path', type=str, default=None,
                        help='推理模式下的图像路径')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='词嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='LSTM层数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载词汇表
    if os.path.exists(args.vocab_path):
        vocab = Vocabulary.load(args.vocab_path)
        print(f'加载词汇表，大小: {len(vocab)}')
    else:
        if args.mode != 'train':
            raise ValueError("词汇表不存在，请先训练模型或提供有效的词汇表路径")
        # 构建词汇表
        json_file = os.path.join(args.data_dir, 'annotations/captions_train2014.json')
        vocab = Vocabulary.build_vocab(json_file)
        vocab.save(args.vocab_path)
        print(f'创建并保存词汇表，大小: {len(vocab)}')
    
    # 创建模型
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers
    ).to(device)
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 准备数据加载器
        train_json = os.path.join(args.data_dir, 'annotations/captions_train2014.json')
        val_json = os.path.join(args.data_dir, 'annotations/captions_val2014.json')
        train_image_dir = os.path.join(args.data_dir, 'train2014')
        val_image_dir = os.path.join(args.data_dir, 'val2014')
        
        train_loader = get_data_loader(
            root_dir=train_image_dir,
            json_file=train_json,
            vocab=vocab,
            batch_size=args.batch_size,
            transform=transform,
            shuffle=True
        )
        
        val_loader = get_data_loader(
            root_dir=val_image_dir,
            json_file=val_json,
            vocab=vocab,
            batch_size=args.batch_size,
            transform=transform,
            shuffle=False
        )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # 训练模型
        print("开始训练...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device
        )
        
    elif args.mode == 'eval':
        # 加载模型
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'加载模型检查点: {args.model_path}')
        else:
            raise ValueError(f"模型检查点不存在: {args.model_path}")
        
        # 准备数据加载器
        val_json = os.path.join(args.data_dir, 'annotations/captions_val2014.json')
        val_image_dir = os.path.join(args.data_dir, 'val2014')
        
        val_loader = get_data_loader(
            root_dir=val_image_dir,
            json_file=val_json,
            vocab=vocab,
            batch_size=args.batch_size,
            transform=transform,
            shuffle=False
        )
        
        # 评估模型
        criterion = nn.CrossEntropyLoss()
        evaluate_model(model, val_loader, criterion, device)
        
        # 计算评估指标
        metrics = evaluate_metrics(model, val_loader, vocab, device)
        
    elif args.mode == 'infer':
        # 加载模型
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'加载模型检查点: {args.model_path}')
        else:
            raise ValueError(f"模型检查点不存在: {args.model_path}")
        
        # 检查图像路径
        if not args.image_path:
            raise ValueError("请提供图像路径")
        
        # 加载图像并生成标题
        image = load_image(args.image_path, transform)
        caption = generate_caption(model, image, vocab, device)
        
        # 显示结果
        print(f'生成的标题: {caption}')
        visualize_caption(args.image_path, caption)
        
if __name__ == '__main__':
    main()