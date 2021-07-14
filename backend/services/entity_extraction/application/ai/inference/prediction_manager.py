import torch

from backend.common.logging.console_loger import ConsoleLogger
from entity_extraction.application.ai.model import BERTEntityModel
from entity_extraction.application.ai.settings import Settings
from entity_extraction.application.ai.training.src.preprocess import Preprocess
import joblib


class PredictionManager:
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.preprocess = preprocess
        self.logger = logger
        self.settings = Settings
        self.enc_pos = None
        self.enc_tag = None

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
