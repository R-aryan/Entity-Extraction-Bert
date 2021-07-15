from injector import Module, singleton

from backend.common.logging.console_loger import ConsoleLogger
from backend.common.logging.logger import Logger
from backend.services.entity_extraction.application.ai.inference.prediction_manager import PredictionManager
from backend.services.entity_extraction.application.ai.training.src.preprocess import Preprocess
from backend.services.entity_extraction.application.ai.settings import Settings


class Configuration(Module):
    def configure(self, binder):
        logger = ConsoleLogger(filename=Settings.LOGS_DIRECTORY)
        binder.bind(Logger, to=logger, scope=singleton)
        binder.bind(PredictionManager, to=PredictionManager(preprocess=Preprocess(), logger=logger), scope=singleton)
