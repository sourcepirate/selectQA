import os
import pandas as pd
from qa.estimators import nlp
from joblib import load


class TextReader(object):
    """ Text Reader """

    def __init__(self, model_path):
        self._model_path = model_path
        self._model = load(model_path)

    def preprocess(self, corpus, query):
        """ preprocess corpus and query """
        qs = []
        for sent in nlp(corpus).sents:
            qs.append({"question": query, "sentence": sent.text})
        return pd.DataFrame(qs)

    def read(self, corpus, query):
        """ reading corpus and answering questions """
        df = self.preprocess(corpus, query)
        return self._model.predict(df)


reader = TextReader(
    os.path.join(os.path.dirname(__file__), "../models/qa.sav")
)