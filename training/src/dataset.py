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
        input_ids = []
        target_pos = []
        target_tag = []
        for index, word in enumerate(text):
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            # ritesh: ri ##te ##sh
            input_len = len(ids)
            input_ids.extend(ids)
            target_pos.extend([pos[index]] * input_len)
            target_tag.extend([tags[index]] * input_len)

        input_ids = input_ids[:self.settings.MAX_LEN - 2]  # for adding [CLS] and [SEP] token
        target_pos = target_pos[:self.settings.MAX_LEN - 2]
        target_tag = target_tag[:self.settings.MAX_LEN - 2]

        # adding token for [CLS] and [SEP] token
        input_ids = [101] + input_ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        # creating attention mask and token type ids
        mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # calculating padding length
        padding_len = self.settings.MAX_LEN - len(input_ids)

        # applying padding to the data
        input_ids = input_ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
