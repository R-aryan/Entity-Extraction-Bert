import torch
import copy
import joblib
from nltk import word_tokenize
from backend.common.logging.console_loger import ConsoleLogger
from entity_extraction.application.ai.model import BERTEntityModel
from entity_extraction.application.ai.settings import Settings
from entity_extraction.application.ai.training.src.dataset import BERTEntityDataset
from entity_extraction.application.ai.training.src.preprocess import Preprocess


class PredictionManager:
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.preprocess = preprocess
        self.logger = logger
        self.settings = Settings
        self.mappings = joblib.load(self.settings.MAPPING_PATH)
        self.enc_pos = {}
        self.enc_tag = {}
        self.__model = None
        self.__load_model()

    def __load_model(self):
        try:
            self.get_tag_mappings()

            self.logger.info(message="Loading Bert Base Uncased Model.")
            self.__model = BERTEntityModel(num_tag=len(self.mappings["enc_tag"]),
                                           num_pos=len(self.mappings["enc_pos"]))

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
            # tokenized_sentence = self.settings.TOKENIZER.encode(sentence)
            self.logger.info(message="Performing prediction on the given data.")
            sentence = sentence.split()
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

                tag, pos, _ = self.__model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    token_type_ids=b_token_type_ids,
                    target_pos=b_target_pos,
                    target_tag=b_target_tag
                )

                tags = tag.argmax(2).cpu().numpy().reshape(-1)
                pos_s = pos.argmax(2).cpu().numpy().reshape(-1)

                # l_pos = []
                # l_tag = []
                # for i in range(len(tokenized_sentence)):
                #     if i == 0 or i == len(tokenized_sentence) - 1:
                #         continue
                #     l_pos.append(self.enc_pos[pos_s[i]])
                #     l_tag.append(self.enc_tag[tags[i]])
                #
                # print("Original Sentence -- ", sentence)
                # print("Tokenized Sentence -- ", tokenized_sentence)
                # print("POS  -- ", str(l_pos))
                # print("TAG--- ", str(l_tag))

            return tags, pos_s

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while prediction---!! " + str(ex))

        return None, None

    def get_tag_mappings(self):

        # pos mappings
        # mappings = joblib.load(self.settings.MAPPING_PATH)
        for k, v in self.mappings["enc_pos"].items():
            self.enc_pos[v] = k

        # tag mappings
        for k, v in self.mappings["enc_tag"].items():
            self.enc_tag[v] = k

    def __post_process(self, tags, pos_s, valid_positions, data):
        result = []
        tag_label = []
        pos_label = []
        for index, mask in enumerate(valid_positions):
            if index == 0 or index == len(valid_positions)-1:
                continue
            if mask == 1:
                tag_label.append(tags[index])
                pos_label.append(pos_s[index])

        output_tags = [self.enc_tag[label] for label in tag_label]
        output_pos = [self.enc_pos[label] for label in pos_label]

        words = word_tokenize(data)
        assert len(output_tags) == len(words) and len(output_pos) == len(words)

        for i in range(len(words)):
            output = {
                "word": words[i],
                "tag": output_tags[i],
                "pos": output_pos[i]
            }

            result.append(output)

        return result

    def run_inference(self, data):
        self.logger.info("Received " + data + " for inference--!!")
        tokens, valid_positions = self.preprocess.tokenize(copy.copy(data))
        tags, pos_s = self.__predict(data)
        result = self.__post_process(
            tags=tags,
            pos_s=pos_s,
            valid_positions=valid_positions,
            data=data
        )

        return result
