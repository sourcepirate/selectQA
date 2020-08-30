import os
import pandas as pd
from qa.pipeline import run_pipeline

tdf = pd.read_pickle("../out/trainer.pkl")

true_set = tdf[tdf["is_correct"] == True]
false_set = tdf[tdf["is_correct"] == False]

df = pd.concat([true_set.loc[:40000], false_set.loc[:40000]])

labels = df["is_correct"]
dataset = df[["question", "sentence"]]

outpath = "{}.sav".format(os.path.join(os.path.dirname(__file__), "../models/qa"))

glovepath = os.path.join(os.path.dirname(__file__), "../out/glove.w2v.bin")

print("running the pipeline....")
report = run_pipeline(dataset, labels, glovepath, outpath)
print(report)
