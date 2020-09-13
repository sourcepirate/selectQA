from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from qa.estimators import TextFeatureTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump


model_estimator = VotingClassifier(
    estimators=[
        ("logistic", LogisticRegression(max_iter=1000)),
        ("guass", GaussianNB()),
        ("svc", SVC(gamma='auto'))
    ],
    voting="hard",
)


def run_pipeline(
    dataset, labels, outfile, accuracy_threshold=0.85
):
    """ running trainng pipleine and save the
    model """

    pipeline = Pipeline(
        steps=[
            ("transform", TextFeatureTransformer()),
            ("model", model_estimator),
        ]
    )

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels)
    pipeline.fit(train_x, train_y)
    scores = cross_val_score(pipeline, train_x, train_y, cv=5)

    try:
        import matplotlib.pyplot as plt
        plt.plot(range(len(scores)), scores, 'g-o')
        plt.xlabel("CV number")
        plt.ylabel("Score")
        plt.savefig("cv_score.jpg")
    except ImportError:
        print(scores)
    test_pred = pipeline.predict(test_x)
    report = classification_report(test_y, test_pred, output_dict=True)
    if report["accuracy"] >= accuracy_threshold:
        dump(pipeline, outfile)
    return report
