from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import NearestCentroid
from qa.estimators import TextFeatureTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump


model_estimator = VotingClassifier(
    estimators=[
        ("svc", SVC()),
        ("logistic", LogisticRegression()),
        ("guass", GaussianNB()),
        ("dt", DecisionTreeClassifier()),
        ("kmeans", NearestCentroid()),
    ],
    voting="hard",
)


def run_pipeline(
    dataset, labels, glove_path, outfile, accuracy_threshold=0.85
):
    """ running trainng pipleine and save the
    model """

    pipeline = Pipeline(
        steps=[
            ("transform", TextFeatureTransformer(glove_path)),
            ("model", model_estimator),
        ]
    )

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels)
    pipeline.fit(train_x, train_y)
    test_pred = pipeline.predict(test_x)
    report = classification_report(test_y, test_pred, output_dict=True)
    if report["accuracy"] >= accuracy_threshold:
        dump(pipeline, outfile)
    return report
