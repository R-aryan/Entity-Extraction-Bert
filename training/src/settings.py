import os

import torch
import transformers


class Settings:
    # PROJ_NAME = 'Entity-Extraction-Bert'
    # root_path = os.getcwd().split(PROJ_NAME)[0] + PROJ_NAME + "\\"
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    RANDOM_STATE = 42
    MODEL_PATH = 'entity_model.bin'
    TRAIN_NUM_WORKERS = 4
    VAL_NUM_WORKERS = 1

    # training data directory
    TRAIN_DATA = "training\\data\\train.csv"

    # test data directory
    TEST_DATA = "training\\data\\test.csv"

    # weights path
    WEIGHTS_PATH = "entity_model.bin"

    # setting up logs path
    # LOGS_DIRECTORY = root_path + "backend\\services\\toxic_comment_jigsaw\\logs\\logs.txt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768
    hidden_dim = 50
    output_dim = 6
    bert_model_name = 'bert-base-uncased'

    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        bert_model_name,
        do_lower_case=True
    )

    DROPOUT = 0.3
