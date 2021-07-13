from backend.common.logging.console_loger import ConsoleLogger
from entity_extraction.application.ai.training.src.preprocess import Preprocess


class PredictionManager:
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.preprocess = preprocess
        self.logger = logger

    def run_inference(self, data):
        return data
