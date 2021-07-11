import torch
import torch.nn as nn
from src.settings import Settings
from transformers import BertModel


class BERTEntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(BERTEntityModel, self).__init__()
        self.settings = Settings
        self.tag = num_tag
        self.pos = num_pos
        self.bert = BertModel.from_pretrained(self.settings.bert_model_name)
        self.bert_drop_1 = nn.Dropout(self.settings.DROPOUT)
        self.bert_drop_2 = nn.Dropout(self.settings.DROPOUT)
        self.out_tag = nn.Linear(self.settings.input_dim, self.num_tag)
        self.out_pos = nn.Linear(self.settings.input_dim, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag,):
        output1, output2 = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        bo_tag = self.bert_drop_1(output1)
        bo_pos = self.bert_drop_2(output1)

        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)

        loss_tag = self.loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = self.loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss

    def loss_fn(self, output, target, mask, num_labels):
        lfn = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, num_labels)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target)
        )
        loss = lfn(active_logits, active_labels)
        return loss
