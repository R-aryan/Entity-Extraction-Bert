import re
import string
import pandas as pd


class Preprocess:
    def __init__(self):
        pass

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

    def prepprocess_data(self, data_path):
        df = pd.read_csv(data_path, encoding="latin-1")
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
        sentences = df.groupby("Sentence #")["Word"].apply(list).values
