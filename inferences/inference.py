# inferences/inference.py
import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms

from models.model import EncoderCNN, DecoderRNN
from data.vocabulary import Vocabulary
from utils.utils import load_config
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 应用实例
app = FastAPI()

# CORS 允许的源
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 如果你本地前端其他端口也要允许，可以都加上
]

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 允许的前端地址列表
    allow_credentials=True,
    allow_methods=["*"],            # 允许所有方法 POST/GET/PUT...
    allow_headers=["*"],            # 允许所有请求头
)

# 配置 & 模型加载
config = load_config("config/config.yaml")
# 选择计算设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化词汇表
vocab = Vocabulary.load("data/vocab.pkl")

# 加载模型
encoder = EncoderCNN(config["model"]["embed_size"]).to(device).eval()
decoder = DecoderRNN(
    config["model"]["embed_size"],
    config["model"]["hidden_size"],
    len(vocab)
).to(device).eval()

# 加载模型的检查点
checkpoint = torch.load(config["training"]["checkpoint_path"], map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def generate_caption(image: Image.Image, strategy="greedy") -> str:
    """
    根据输入的图像生成描述
    :param image: 输入的图像
    :param strategy: 生成策略，默认为贪心策略
    :return: 生成的描述字符串
    """
    # 对图像进行预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # 提取图像特征
        feature = encoder(image_tensor)
        # 生成描述的 ID 列表
        output_ids = decoder.sample(feature, strategy=strategy)

    # 将 ID 列表转换为单词列表，去除特殊标记
    words = [vocab.itos[idx] for idx in output_ids if idx not in (
        vocab.stoi["<START>"], vocab.stoi["<END>"], vocab.stoi["<PAD>"])]
    return " ".join(words)


@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    """
    处理图像上传请求，生成描述
    :param file: 上传的图像文件
    :return: 包含文件名和生成描述的字典，或错误信息
    """
    try:
        # 读取上传的图像文件内容
        content = await file.read()
        # 打开图像并转换为 RGB 格式
        image = Image.open(io.BytesIO(content)).convert("RGB")
        # 生成描述
        caption = generate_caption(image)
        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        return {"error": str(e)}