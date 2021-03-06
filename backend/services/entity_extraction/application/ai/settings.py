import os

import torch
import transformers


class Settings:
    PROJ_NAME = 'Entity-Extraction-Bert'
    root_path = os.getcwd().split(PROJ_NAME)[0] + PROJ_NAME + "\\"
    APPLICATION_PATH = root_path + "backend\\services\\entity_extraction\\application\\"
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    RANDOM_STATE = 42
    MODEL_PATH = 'entity_model.bin'
    TRAIN_NUM_WORKERS = 4
    VAL_NUM_WORKERS = 2

    # training data directory
    TRAIN_DATA = APPLICATION_PATH + "ai\\training\\data\\ner_dataset.csv"

    # weights path
    WEIGHTS_PATH = APPLICATION_PATH + "ai\\weights\\entity_model.bin"
    # Mapping path
    MAPPING_PATH = APPLICATION_PATH + "ai\\mapping.bin"

    # setting up logs path
    LOGS_DIRECTORY = root_path + "backend\\services\\entity_extraction\\logs\\logs.txt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768
    bert_model_name = 'bert-base-uncased'

    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        bert_model_name,
        do_lower_case=True
    )

    DROPOUT = 0.3
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    seed_value = 42
    test_size = 0.1


