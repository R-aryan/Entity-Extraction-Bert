import joblib
import pandas as pd
import numpy as np
import torch
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from backend.services.entity_extraction.application.ai.training.src.engine import Engine
from backend.services.entity_extraction.application.ai.model import BERTEntityModel
from backend.services.entity_extraction.application.ai.training.src.preprocess import Preprocess
from backend.services.entity_extraction.application.ai.settings import Settings
from backend.services.entity_extraction.application.ai.training.src.dataset import BERTEntityDataset


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
        self.best_loss = np.inf
        self.meta_data = None

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

        self.meta_data = {
            "enc_pos": pos_label_dict,
            "enc_tag": tag_label_dict
        }
        joblib.dump(self.meta_data, self.settings.MAPPING_PATH)

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
        # train data loader
        self.train_data_loader = self.create_data_loaders(train_sentences, train_pos, train_tag,
                                                          self.settings.TRAIN_BATCH_SIZE,
                                                          self.settings.TRAIN_NUM_WORKERS)
        # validation data loader
        self.validation_data_loader = self.create_data_loaders(test_sentences, test_pos, test_tag,
                                                               self.settings.VALID_BATCH_SIZE,
                                                               self.settings.VAL_NUM_WORKERS)

        self.total_steps = int(len(train_sentences) / self.settings.TRAIN_BATCH_SIZE * self.settings.EPOCHS)

    def train(self):
        for epochs in range(self.settings.EPOCHS):
            train_loss = self.engine.train_fn(data_loader=self.train_data_loader,
                                              model=self.bert_classifier,
                                              optimizer=self.optimizer,
                                              device=self.settings.DEVICE,
                                              schedular=self.scheduler)

            test_loss = self.engine.eval_fn(data_loader=self.validation_data_loader,
                                            model=self.bert_classifier,
                                            device=self.settings.DEVICE)

            if test_loss < self.best_loss:
                torch.save(self.bert_classifier.state_dict(), self.settings.MODEL_PATH)
                self.best_loss = test_loss

    def run(self):
        try:
            print("Loading and Preparing the Dataset-----!! ")
            self.load_data(self.settings.TRAIN_DATA)
            print("Dataset Successfully Loaded and Prepared-----!! ")
            print()
            print("-" * 70)
            print("Loading and Initializing the Bert Model -----!! ")
            self.__initialize(num_pos=len(self.meta_data["enc_pos"]), num_tag=len(self.meta_data["enc_tag"]))
            print("Model Successfully Loaded and Initialized-----!! ")
            print()
            print("-" * 70)
            print("------------------Starting Training-----------!!")
            self.engine.set_seed()
            self.train()
            print("Training complete-----!!!")

        except BaseException as ex:
            print("Following Exception Occurred---!! ", str(ex))
