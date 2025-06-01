# eval/evaluate.py

import os
import torch
from torchvision import transforms
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from models.model import EncoderCNN, DecoderRNN
from data.vocabulary import Vocabulary
from utils.utils import load_config, clean_caption


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def evaluate_single(image_path, encoder, decoder, vocab, transform, device, max_len=20):
    image = load_image(image_path, transform).to(device)
    with torch.no_grad():
        feature = encoder(image)
        output_ids = decoder.sample(feature, max_len=max_len)

    caption = [vocab.idx2word[idx] for idx in output_ids if idx not in (vocab.word2idx["<pad>"], vocab.word2idx["<start>"], vocab.word2idx["<end>"])]

    return " ".join(caption)


def evaluate_folder(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    vocab = Vocabulary.load("data/vocab.pkl")

    # Load model
    encoder = EncoderCNN(config["model"]["embed_size"]).to(device)
    decoder = DecoderRNN(
        config["model"]["embed_size"],
        config["model"]["hidden_size"],
        len(vocab)
    ).to(device)

    checkpoint = torch.load(config["training"]["checkpoint_path"], map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    references_file = config["eval"]["reference_captions"]
    image_folder = config["eval"]["image_folder"]

    with open(references_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_score = 0
    smooth = SmoothingFunction().method4
    for line in lines:
        img_name, ref_caption = line.strip().split("\t")
        image_path = os.path.join(image_folder, img_name)
        pred_caption = evaluate_single(image_path, encoder, decoder, vocab, transform, device)

        score = sentence_bleu(
            [clean_caption(ref_caption).split()],
            clean_caption(pred_caption).split(),
            smoothing_function = smooth
        )
        print(f"[{img_name}]\nPred: {pred_caption}\nRef:  {ref_caption}\nBLEU: {score:.4f}\n")
        total_score += score

    avg_bleu = total_score / len(lines)
    print(f"Average BLEU: {avg_bleu:.4f}")
