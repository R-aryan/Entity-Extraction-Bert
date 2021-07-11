import joblib
import pandas as pd
import numpy as np
import torch
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from src.engine import Engine
from src.model import BERTEntityModel
from src.preprocess import Preprocess
from src.settings import Settings
from src.dataset import BERTEntityDataset


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
        self.train_data_loader = None
        self.validation_data_loader = None

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

    def create_data_loaders(self, sentences, pos, tag, batch_size, num_workers):
        dataset = BERTEntityDataset(texts=sentences, pos=pos, tags=tag)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        return data_loader

    def load_data(self, csv_data_path):
        df = pd.read_csv(csv_data_path, encoding="latin-1")
        sentences, pos_labels, tag_labels, pos_label_dict, tag_label_dict = self.preprocess.prepprocess_data(df)

        meta_data = {
            "enc_pos": pos_label_dict,
            "enc_tag": tag_label_dict
        }
        joblib.dump(meta_data, "mapping.bin")

        # splitting data into train and test set
        (
            train_sentences,
            test_sentences,
            train_pos,
            test_pos,
            train_tag,
            test_tag
        ) = model_selection.train_test_split(sentences, pos_labels, tag_labels, random_state=self.settings.seed_value,
                                             test_size=self.settings.test_size)

        # creating Data Loaders
        # train dataloader
        self.train_data_loader = self.create_data_loaders(train_sentences, train_pos, train_tag,
                                                          self.settings.TRAIN_BATCH_SIZE,
                                                          self.settings.TRAIN_NUM_WORKERS)
        # validation data loader
        self.validation_data_loader = self.create_data_loaders(test_sentences, test_pos, test_tag,
                                                               self.settings.VALID_BATCH_SIZE,
                                                               self.settings.VAL_NUM_WORKERS)

        self.total_steps = int(len(train_sentences) / self.settings.TRAIN_BATCH_SIZE * self.settings.EPOCHS)

    def run(self):
        pass
