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


# 添加早停类
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

# 添加BLEU评估函数
def calculate_bleu_scores(model, dataloader, vocab, device, num_samples=100):
    """计算BLEU分数"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    model.eval()
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            
            # 生成描述
            generated = model.generate_caption(images[0:1], vocab, max_len=20)
            
            # 获取真实描述
            real_caption = []
            for idx in captions[0]:
                if idx.item() == vocab.word2idx.get('<end>', 2):
                    break
                if idx.item() not in [vocab.word2idx.get('<start>', 1), vocab.word2idx.get('<pad>', 0)]:
                    word = vocab.idx2word.get(idx.item(), '<unk>')
                    real_caption.append(word)
            
            if generated and real_caption:
                generated_tokens = generated.split()
                reference_tokens = [real_caption]
                
                bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
    
    return bleu_scores