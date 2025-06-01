import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, transform):
    """加载和预处理图像"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def generate_caption(model, image, vocab, device):
    """为图像生成描述"""
    # 将图像移到设备上
    image = image.to(device)
    
    # 生成标题
    model.eval()
    with torch.no_grad():
        sampled_ids = model.generate_caption(image)
    
    # 将索引转换为单词
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    # 从<start>后开始，到<end>结束
    words = []
    for idx in sampled_ids:
        word = vocab.idx2word[idx]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>', '<unk>']:
            words.append(word)
    
    # 将单词连接成句子
    caption = ' '.join(words)
    return caption

def visualize_caption(image_path, caption):
    """可视化图像及其生成的标题"""
    image = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    plt.show()

def batch_inference(model, image_paths, transform, vocab, device):
    """批量为多张图像生成描述"""
    results = []
    
    for path in image_paths:
        image = load_image(path, transform)
        caption = generate_caption(model, image, vocab, device)
        results.append((path, caption))
    
    return results