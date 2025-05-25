import torch
import os


def save_checkpoint(model, optimizer, epoch, filename, loss=None):
    """保存模型检查点"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    torch.save(checkpoint, filename)
    print(f"检查点已保存: {filename}")


def load_checkpoint(model, optimizer, filename):
    """加载模型检查点"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"检查点文件不存在: {filename}")

    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"检查点已加载: {filename}, epoch: {epoch}")
    return epoch


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    return total_params, trainable_params


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device