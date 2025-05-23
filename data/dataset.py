import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import csv

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform if transform else transforms.ToTensor()
        self.imgs = []
        self.captions = []

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgs.append(row['image'])
                self.captions.append(row['caption'])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.captions[index]
        numericalized_caption = [self.vocab.word2idx["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<end>"])

        caption_tensor = torch.Tensor(numericalized_caption).long()
        return image, caption_tensor
