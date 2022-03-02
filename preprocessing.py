from settings import BASE_DIR
import string
import re
from nltk.stem import SnowballStemmer
import pandas as pd


class Preprocessing:
    def remove_mentions(text):
        text = " ".join(filter(lambda x: x[0] != '@', text.split()))
        return text

    def remove_links(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        return text

    def remove_punctuation(text):
        text = "".join([i for i in text if i not in string.punctuation])
        return text

    def sub_emoji(text):
        emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
                  ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                  ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused',
                  ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                  '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                  '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
                  ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

        for emoji in emojis.keys():
            text = text.replace(emoji, emojis[emoji])
        return text

    def sub_negations(text):
        negations = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                     "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                     "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                     "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                     "mustn't": "must not"}
        for negation in negations.keys():
            text = text.replace(negation, negations[negation])
        return text

    def stemming(text):
        sbStem = SnowballStemmer("english", True)
        stemmed_text = ""
        for word in text.split():
            word = sbStem.stem(word)
            stemmed_text += (word + " ")
        return stemmed_text

    def remove_extra_space(text):
        text = re.sub(' +', ' ', text)
        return text

    def text_processing(text):
        text = text.lower()
        text = Preprocessing.remove_mentions(text)
        text = Preprocessing.remove_links(text)
        text = Preprocessing.remove_punctuation(text)
        text = Preprocessing.sub_emoji(text)
        text = Preprocessing.sub_negations(text)
        text = Preprocessing.stemming(text)
        text = Preprocessing.remove_extra_space(text)
        return text

    def sentiment_preprocess(sentiment):
        if sentiment == 4:
            sentiment = 1
        return sentiment

    def preprocess(df):
        print("starting preprocessing...")
        df["text"] = df["text"].apply(lambda x: Preprocessing.text_processing(x))
        df["sentiment"] = df["sentiment"].apply(lambda x: Preprocessing.sentiment_preprocess(x))
        print("...preprocessing completed")
        return df

    def main(self):
        path = "{BaseDir}/TwiiterSentimentAnalysis/data/dataset.csv".format(BaseDir=BASE_DIR)
        columns = ["sentiment", "ids", "date", "flag", "user", "text"]
        final_columns = ["text", "sentiment"]
        df = pd.read_csv(path, encoding='ISO-8859-1', names=columns)
        df = df[final_columns]
        return df

