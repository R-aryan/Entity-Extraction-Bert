import torch

from backend.common.logging.console_loger import ConsoleLogger
from entity_extraction.application.ai.model import BERTEntityModel
from entity_extraction.application.ai.settings import Settings
from entity_extraction.application.ai.training.src.dataset import BERTEntityDataset
from entity_extraction.application.ai.training.src.preprocess import Preprocess
import joblib


class PredictionManager:
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.preprocess = preprocess
        self.logger = logger
        self.settings = Settings
        self.enc_pos = None
        self.enc_tag = None
        self.__model = None

    def __load_model(self):
        try:
            self.__get_tag_mappings()

            self.logger.info(message="Loading Bert Base Uncased Model.")
            self.__model = BERTEntityModel(num_tag=len(self.enc_tag),
                                           num_pos=len(self.enc_pos))

            self.logger.info(message="Bert Base Model Successfully Loaded.")

            self.logger.info(message="Loading Model trained Weights.")
            self.__model.load_state_dict(torch.load(self.settings.WEIGHTS_PATH,
                                                    map_location=torch.device(self.settings.DEVICE)))
            self.__model.to(self.settings.DEVICE)
            self.__model.eval()
            self.logger.info(message="Model Weights loaded Successfully--!!")

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while loading model---!! " + str(ex))

    def __predict(self, sentence):

        try:
            tokens, valid_positions = self.preprocess.tokenize(sentence)
            self.logger.info(message="Performing prediction on the given data.")
            test_dataset = BERTEntityDataset(
                texts=[sentence],
                pos=[[0] * len(sentence)],
                tags=[[0] * len(sentence)]
            )

            with torch.no_grad():
                data = test_dataset[0]
                b_input_ids = data['input_ids']
                b_attention_mask = data['attention_mask']
                b_token_type_ids = data['token_type_ids']
                b_target_pos = data['target_pos']
                b_target_tag = data['target_tag']

                # moving tensors to device
                b_input_ids = b_input_ids.to(self.settings.DEVICE).unsqueeze(0)
                b_attention_mask = b_attention_mask.to(self.settings.DEVICE).unsqueeze(0)
                b_token_type_ids = b_token_type_ids.to(self.settings.DEVICE).unsqueeze(0)
                b_target_pos = b_target_pos.to(self.settings.DEVICE).unsqueeze(0)
                b_target_tag = b_target_tag.to(self.settings.DEVICE).unsqueeze(0)

                tag, pos, loss = self.__model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    token_type_ids=b_token_type_ids,
                    target_pos=b_target_pos,
                    target_tag=b_target_tag
                )

                tags = tag.argmax(2).cpu().numpy().reshape(-1)
                pos_s = pos.argmax(2).cpu().numpy().reshape(-1)

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while prediction---!! " + str(ex))

    def __get_tag_mappings(self):

        # pos mappings
        mappings = joblib.load(self.settings.MAPPING_PATH)
        for k, v in mappings["enc_pos"].items():
            self.enc_pos[v] = k

        # tag mappings
        for k, v in mappings["enc_tag"].items():
            self.enc_tag[v] = k

    def run_inference(self, data):
        self.logger.info("Received ", data, " for inference--!!")

        return data
