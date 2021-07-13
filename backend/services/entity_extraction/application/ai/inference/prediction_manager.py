from backend.common.logging.console_loger import ConsoleLogger


class PredictionManager:
    def __init__(self, logger: ConsoleLogger):
        self.logger = logger

    def run_inference(self, data):
        return data
