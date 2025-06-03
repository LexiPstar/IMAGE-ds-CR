import pickle
from data.data_loader import get_loader
from utils.utils import load_config
import sys
import os

# 将当前脚本所在目录的上级目录添加到系统路径中，以便能够导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # 加载配置文件
    config = load_config("config/config.yaml")

    # 使用 get_loader 函数加载数据，这里主要是为了提取词汇表
    # 由于我们只需要词汇表，所以忽略前两个返回值（数据加载器和数据集）
    _, _, vocab = get_loader(
        image_folder=config["data"]["image_folder"],
        captions_file=config["data"]["captions_file"],
        batch_size=config["training"]["batch_size"],
        freq_threshold=config["data"]["freq_threshold"],
        shuffle=False
    )

    # 将提取的词汇表保存到指定文件中
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 打印成功信息
    print("✅ Vocabulary saved to data/vocab.pkl")


if __name__ == "__main__":
    main()