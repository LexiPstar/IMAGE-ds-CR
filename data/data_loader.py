# data/data_loader.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import ImageCaptionDataset
from .vocabulary import Vocabulary
import torchvision.transforms as transforms


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(c) for c in captions]
    padded_captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, padded_captions, lengths


def get_loader(image_folder, captions_file, batch_size=32, freq_threshold=5, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    df = pd.read_csv(captions_file)
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(df['caption'].tolist())

    dataset = ImageCaptionDataset(
        root_dir=image_folder,
        captions_file=captions_file,
        vocab=vocab,
        transform=transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return loader, dataset, vocab
