import pickle
from data.data_loader import get_loader
from utils.utils import load_config
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    config = load_config("config/config.yaml")

    # 用 get_loader 只为了提取 vocab
    _, _, vocab = get_loader(
        image_folder=config["data"]["image_folder"],
        captions_file=config["data"]["captions_file"],
        batch_size=config["training"]["batch_size"],
        freq_threshold=config["data"]["freq_threshold"],
        shuffle=False
    )

    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("✅ Vocabulary saved to data/vocab.pkl")


if __name__ == "__main__":
    main()
