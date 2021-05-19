import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import pymorphy2
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя '
        self.ru_alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        self.en_alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.stop_words = stopwords.words(['english', 'russian'])
        self.ru_stemmer = SnowballStemmer('russian')
        self.en_stemmer = SnowballStemmer('english')

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")
        stammer = self.en_stemmer
        if x[0] in self.ru_alphabet:
            stammer = self.ru_stemmer

        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()
        words = [stammer.stem(w) for w in words if w not in self.stop_words]
        return ' '.join([self.morph.parse(w)[0].normal_form for w in words])

    def _fit_predict(self, train, test):
        vectorizer_a = TfidfVectorizer(max_features=1000)
        vectorizer_b = TfidfVectorizer(max_features=1000)

        train["message_a"] = train["message_a"].apply(self.normalize_text_with_morph)
        train["message_b"] = train["message_b"].apply(self.normalize_text_with_morph)
        train_a = vectorizer_a.fit_transform(train["message_a"]).toarray()
        train_b = vectorizer_b.fit_transform(train["message_b"]).toarray()
        _train = np.hstack([train_a, train_b])

        test["message_a"] = test["message_a"].apply(self.normalize_text_with_morph)
        test["message_b"] = test["message_b"].apply(self.normalize_text_with_morph)
        test_a = vectorizer_a.transform(test["message_a"]).toarray()
        test_b = vectorizer_b.transform(test["message_b"]).toarray()
        _test = np.hstack([test_a, test_b])

        regressor = CatBoostRegressor(random_state=42)
        regressor.fit(_train, train["target"])
        return pd.DataFrame(np.round(np.clip(regressor.predict(_test), 0, 1)), columns=["target"])

    def fit_predict(self,
                    train_1, test_1,
                    train_2, test_2,
                    train_3, test_3,
                    train_4, test_4,
                    train_5, test_5):
        predicted_1 = self._fit_predict(train_1, test_1)
        predicted_2 = self._fit_predict(train_2, test_2)
        predicted_3 = self._fit_predict(train_3, test_3)
        predicted_4 = self._fit_predict(train_4, test_4)
        predicted_5 = self._fit_predict(train_5, test_5)
        return [predicted_1, predicted_2, predicted_3, predicted_4, predicted_5]



