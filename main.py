# main.py

import argparse
import uvicorn

from training.train import train
from eval.evaluate import evaluate_folder
from utils.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="图文描述模型 主入口")
    parser.add_argument(
        "mode",
        choices=["train", "eval", "serve"],
        help="选择运行模式：train / eval / serve"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径（默认：config/config.yaml）"
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("🚀 开始训练...")
        config = load_config(args.config)
        train(config)

    elif args.mode == "eval":
        print("🔎 开始评估...")
        evaluate_folder(args.config)

    elif args.mode == "serve":
        print("🌐 启动 FastAPI 推理服务...")
        uvicorn.run("inferences.inference:app", host="0.0.0.0", port=8000, reload=True)

    else:
        print("❌ 未知模式")


if __name__ == "__main__":
    main()
