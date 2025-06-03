# main.py

import argparse
import uvicorn

from training.train import train
from eval.evaluate import evaluate_folder
from utils.utils import load_config
import nltk
# 下载 nltk 的 punkt_tab 数据，用于分词
nltk.download('punkt_tab')


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="图文描述模型 主入口")
    # 添加模式参数，可选值为 train、eval、serve
    parser.add_argument(
        "mode",
        choices=["train", "eval", "serve"],
        help="选择运行模式：train / eval / serve"
    )
    # 添加配置文件路径参数，默认值为 config/config.yaml
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径（默认：config/config.yaml）"
    )
    # 解析命令行参数
    args = parser.parse_args()

    if args.mode == "train":
        """ python main.py train --config config/config.yaml"""
        # 打印开始训练的信息
        print("开始训练...")
        # 加载配置文件
        config = load_config(args.config)
        # 调用训练函数开始训练
        train(config)

    elif args.mode == "eval":
        """python main.py eval --config config/config.yaml"""
        # 打印开始评估的信息
        print("开始评估...")
        # 调用评估函数进行评估
        evaluate_folder(args.config)

    elif args.mode == "serve":
        """python main.py serve"""
        # 打印启动推理服务的信息
        print("启动 FastAPI 推理服务...")
        # 启动 FastAPI 服务
        uvicorn.run("inferences.inference:app", host="0.0.0.0", port=8000, reload=True)

    else:
        # 打印未知模式的信息
        print("未知模式")


if __name__ == "__main__":
    main()