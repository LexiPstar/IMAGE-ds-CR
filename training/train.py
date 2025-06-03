# training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from data.vocabulary import Vocabulary
from models.model import EncoderCNN, DecoderRNN
from data.data_loader import get_loader

import logging

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def train(config):
    setup_logger(config["training"]["log_path"])
    logging.info("Training started.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader, dataset, vocab = get_loader(
        image_folder=config["data"]["image_folder"],
        captions_file=config["data"]["captions_file"],
        batch_size=config["training"]["batch_size"],
        freq_threshold=config["data"]["freq_threshold"],
        shuffle=True
    )

    # Build model
    encoder = EncoderCNN(config["model"]["embed_size"]).to(device)
    decoder = DecoderRNN(
        config["model"]["embed_size"],
        config["model"]["hidden_size"],
        len(vocab)
    ).to(device)

    # Loss and optimizer
    pad_idx = vocab.stoi[Vocabulary.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=config["training"]["lr"])

    # Resume from checkpoint
    start_epoch = 0
    if config["training"]["resume"] and os.path.exists(config["training"]["checkpoint_path"]):
        checkpoint = torch.load(config["training"]["checkpoint_path"])
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Resumed from epoch {start_epoch}")


    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        encoder.train()
        decoder.train()

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{config['training']['num_epochs']}]")
        total_loss = 0

        for imgs, captions, lengths in loop:
            imgs, captions = imgs.to(device), captions.to(device)

            features = encoder(imgs)
            outputs = decoder(features, captions)

            # Flatten for loss calculation
            targets = pack_padded_sequence(captions[:, 1:], lengths=[l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
            outputs = pack_padded_sequence(outputs, lengths=[l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Save checkpoint
        os.makedirs(os.path.dirname(config["training"]["checkpoint_path"]), exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, config["training"]["checkpoint_path"])

        logging.info(f"Epoch {epoch + 1} completed, loss: {total_loss / len(loader):.4f}")
