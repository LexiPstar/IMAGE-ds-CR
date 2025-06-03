# data/data_loader.py
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from .dataset import ImageCaptionDataset
from .vocabulary import Vocabulary
import torchvision.transforms as transforms


class collate_fn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images)  # [B, C, H, W]

        # Pad captions
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        lengths = [len(cap) for cap in captions]

        return images, captions, lengths


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

    pad_idx = vocab.stoi[Vocabulary.PAD_TOKEN]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn(pad_idx)
    )

    return loader, dataset, vocab
