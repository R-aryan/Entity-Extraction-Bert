from backend.common.logging.console_loger import ConsoleLogger
from backend.services.entity_extraction.application.ai.inference.prediction_manager import PredictionManager
from backend.services.entity_extraction.application.ai.training.src.preprocess import Preprocess
from backend.services.entity_extraction.application.ai.settings import Settings

sentence = "Ritesh is coming to india"
p1 = PredictionManager(preprocess=Preprocess(), logger=ConsoleLogger(filename=Settings.LOGS_DIRECTORY))

print("Sample Input, ", sentence)
output = p1.run_inference(sentence)
print(output)
