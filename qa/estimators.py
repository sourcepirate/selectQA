import os
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import KeyedVectors

nlp = spacy.load("en")

QUESTION_TOKENS = ["what", "who", "where", "when", "why"]
POS_COUNTS = ["NOUN", "ADV", "VERB", "PROPN"]
LABEL_COUNTS = ["PERSON", "ORG", "GPE", "NORP", "DATE"]


def get_keyed_vectors(filepath: str, binary: bool = True) -> KeyedVectors:
    """ get the keyed vectors from the model file """
    return KeyedVectors.load_word2vec_format(filepath, binary=binary)


def matched(a: str, b: str):
    a_vec = set(a.lower().split(" "))
    b_vec = set(b.lower().split(" "))
    return len(a_vec & b_vec)


class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """ this estimator add extra features like
    question based and answer token based features """

    def __init__(self, glove_path):
        self._glove_path = glove_path
        self.w2v = get_keyed_vectors(glove_path)

    def fit(self, X, y=None):
        return self

    def similarity(self, a: str, b: str) -> float:
        return self.w2v.wmdistance(a, b)

    def apply_ner_features(self, X):
        feature_tokens = []
        for tag in LABEL_COUNTS:
            feature_token = f"num_{tag.lower()}"
            X.loc[:, f"a_{feature_token}"] = X["sentence"].apply(
                lambda x: len([e for e in nlp(x).ents if e.label_ == tag])
            )
            X.loc[:, f"q_{feature_token}"] = X["question"].apply(
                lambda x: len([e for e in nlp(x).ents if e.label_ == tag])
            )
            feature_tokens.append(f"q_{feature_token}")
            feature_tokens.append(f"a_{feature_token}")
        return X, feature_tokens

    def apply_question_tokens(self, X):
        feature_tokens = []
        for token in QUESTION_TOKENS:
            feature_token = f"is_{token.lower()}"
            X.loc[:, feature_token] = X["question"].apply(
                lambda x: token in list(map(lambda i: i.lower(), x.split(" ")))
            )
            feature_tokens.append(feature_token)
        return X, feature_tokens

    def apply_pos(self, X):
        feature_tokens = []
        for tag in POS_COUNTS:
            feature_token = f"{tag}_words"
            X.loc[:, f"a_{feature_token}"] = X["sentence"].apply(
                lambda row: len(
                    [x for x in nlp(row, disable=["ner"]) if x.pos_ == tag]
                )
            )
            X.loc[:, f"q_{feature_token}"] = X["question"].apply(
                lambda row: len(
                    [x for x in nlp(row, disable=["ner"]) if x.pos_ == tag]
                )
            )
            feature_tokens.append(f"q_{feature_token}")
            feature_tokens.append(f"a_{feature_token}")
        return X, feature_tokens

    def transform(self, X, y=None):
        X.loc[:, "similarity"] = X.apply(
            lambda row: self.similarity(row["question"], row["sentence"]),
            axis=1,
        )
        X.loc[:, "matched"] = X.apply(
            lambda row: matched(row["question"], row["sentence"]), axis=1
        )
        X, qt = self.apply_question_tokens(X)
        X, pt = self.apply_pos(X)
        X, nt = self.apply_ner_features(X)
        features = qt + pt + nt + ["similarity", "matched"]
        x_feat = X[features]
        return x_feat
