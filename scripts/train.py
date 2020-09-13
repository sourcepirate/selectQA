import os
import sys
import pandas as pd
from qa.pipeline import run_pipeline

tdf = pd.read_pickle(sys.argv[1])

true_set = tdf[tdf["is_correct"] == True]
false_set = tdf[tdf["is_correct"] == False]


def sample(n):
    return pd.concat([true_set.sample(n), false_set.sample(n)])

df = sample(int(sys.argv[2]))
threshold = float(sys.argv[3])
labels = df["is_correct"]
dataset = df[["question", "sentence"]]

outpath = "{}.lzma".format(
    os.path.join(os.path.dirname(__file__), "../models/qa")
)

print("running the pipeline....")
report = run_pipeline(dataset, labels, outpath, accuracy_threshold=threshold)
print(report)
