import spacy
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en")

QUESTION_TOKENS = ["what", "who", "where", "when", "why", "how"]
POS_COUNTS = ["NOUN", "ADV", "VERB", "PROPN"]
LABEL_COUNTS = ["PERSON", "ORG", "GPE", "NORP", "DATE"]


def matched(a: str, b: str):
    a_vec = set(a.lower().strip().split(" "))
    b_vec = set(b.lower().strip().split(" "))
    return len(a_vec & b_vec)


def question_type(question):
    for i, token in enumerate(QUESTION_TOKENS):
        if token in question.lower():
            return i + 1
    return 0


def apply_meta_to_sent(sents):
    pos_tags = []
    ner_tags = []
    tokenizer = nlp(sents)
    for token in tokenizer:
        pos_tags.append(token.pos_)
    for ent in tokenizer.ents:
        ner_tags.append(ent.label_)
    return {"pos": Counter(pos_tags), "ner": Counter(ner_tags)}


class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """this estimator add extra features like
    question based and answer token based features"""

    def fit(self, X, y=None):
        return self

    def apply_question_tokens(self, X):
        X.loc[:, "q_type"] = X["question"].apply(question_type)
        return X, ["q_type"]

    def apply_meta(self, X):
        meta_tags_question = X["question"].apply(apply_meta_to_sent)
        meta_tags_answer = X["sentence"].apply(apply_meta_to_sent)
        feature_tokens = []
        for pos in POS_COUNTS:
            q_token = f"q_{pos.lower()}"
            a_token = f"a_{pos.lower()}"
            X.loc[:, q_token] = meta_tags_question.apply(
                lambda x: x["pos"][pos]
            )
            X.loc[:, a_token] = meta_tags_answer.apply(lambda x: x["pos"][pos])
            feature_tokens.append(q_token)
            feature_tokens.append(a_token)
        for ner in LABEL_COUNTS:
            q_token = f"q_{ner.lower()}"
            a_token = f"a_{ner.lower()}"
            X.loc[:, q_token] = meta_tags_question.apply(
                lambda x: x["ner"][ner]
            )
            X.loc[:, a_token] = meta_tags_answer.apply(lambda x: x["ner"][ner])
            feature_tokens.append(q_token)
            feature_tokens.append(a_token)
        return X, feature_tokens

    def transform(self, X, y=None):
        X.loc[:, "matched"] = X.apply(
            lambda row: matched(row["question"], row["sentence"]), axis=1
        )
        X, qt = self.apply_question_tokens(X)
        X, ft = self.apply_meta(X)
        features = qt + ft + ["matched"]
        x_feat = X[features]
        return x_feat
