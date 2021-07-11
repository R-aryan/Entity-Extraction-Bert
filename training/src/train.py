import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from src.engine import Engine
from src.model import BERTEntityModel
from src.preprocess import Preprocess
from src.settings import Settings


class Train:
    def __init__(self):
        # initialize required class
        self.settings = Settings
        self.engine = Engine()
        self.preprocess = Preprocess()

        # initialize required variables
        self.bert_classifier = None
        self.optimizer = None
        self.scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.total_steps = None
        self.best_accuracy = 0
        self.param_optimizer = None
        self.optimizer_parameters = None
        self.total_steps = None

    def __initialize(self, num_tag, num_pos):
        # Instantiate Bert Classifier
        self.bert_classifier = BERTEntityModel(num_tag, num_pos)
        self.bert_classifier.to(self.settings.DEVICE)
        self.param_optimizer = list(self.bert_classifier.named_parameters())

        self.optimizer_parameters = [
            {
                "params": [
                    p for n, p in self.param_optimizer if not any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in self.param_optimizer if any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Create the optimizer
        self.optimizer = AdamW(self.optimizer_parameters,
                               lr=5e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.total_steps)

    def load_data(self, csv_data_path):
        df = pd.read_csv(csv_data_path,encoding="latin-1")
        sentences, pos_labels, tag_labels, pos_label_dict, tag_label_dict = self.preprocess.prepprocess_data(df)

        meta_data = {
            "enc_pos": pos_label_dict,
            "enc_tag": tag_label_dict
        }
        joblib.dump(meta_data, "mapping.bin")



    def run(self):
        pass
