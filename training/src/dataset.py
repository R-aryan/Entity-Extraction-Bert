import torch
from src.settings import Settings


class BERTEntityDataset:
    def __init__(self, texts, pos, tags):
        self.settings = Settings
        self.texts = texts
        self.pos = pos
        self.tags = tags
        self.tokenizer = self.settings.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]
