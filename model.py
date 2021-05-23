import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import pymorphy2
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


seed = 42
class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя '
        self.ru_alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        self.en_alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.stop_words = stopwords.words(['english', 'russian'])
        self.ru_stemmer = SnowballStemmer('russian')
        self.en_stemmer = SnowballStemmer('english')

    def stemming(self, w):
        if not w:
            return ''
        if w[0] in self.en_alphabet:
            return self.en_stemmer.stem(w)
        return self.ru_stemmer.stem(w)

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")

        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()  # токенизация
        words = [self.morph.normal_forms(w)[0] for w in words]  # лемматизация
        words = [self.stemming(w) for w in words]  # стемминг
        return ' '.join(words)

    def _fit_predict(self, train, test):
        vectorizer_a = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=None)
        vectorizer_b = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=None)


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


        model = CatBoostRegressor(random_state=seed, learning_rate=0.1, l2_leaf_reg=5, thread_count=-1, depth=10)

        model.fit(_train, train["target"])

        pred = np.round(np.clip(model.predict(_test), 0, 1))
        return pd.DataFrame(pred, columns=["target"])


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

    def one_fit_predict(self, train_1, test_1):
        predicted_1 = self._fit_predict(train_1, test_1)
        return predicted_1
