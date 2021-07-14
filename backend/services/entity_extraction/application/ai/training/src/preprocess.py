import re
import string
from nltk import word_tokenize

from backend.services.entity_extraction.application.ai.settings import Settings


class Preprocess:
    def __init__(self):
        self.settings = Settings

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def get_unique_labels(self, df, label):
        possible_labels = df[label].unique()
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index

        df[label + '_label'] = df[label].replace(label_dict)

        return df, label_dict

    def prepprocess_data(self, df):
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        df, pos_label_dict = self.get_unique_labels(df, "POS")
        df, tag_label_dict = self.get_unique_labels(df, "Tag")

        pos_labels = df.groupby("Sentence #")["POS_label"].apply(list).values
        pos = df.groupby("Sentence #")["POS"].apply(list).values

        tag_labels = df.groupby("Sentence #")["Tag_label"].apply(list).values
        tag = df.groupby("Sentence #")["Tag"].apply(list).values

        return sentences, pos_labels, tag_labels, pos_label_dict, tag_label_dict

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.settings.TOKENIZER.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions
