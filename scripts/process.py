import os
import json
import pandas as pd
from tqdm import tqdm
import spacy
import argparse

nlp = spacy.load("en")


def parse_data(data: dict) -> list:
    """
    Parses the JSON file of Squad dataset by looping through the
    keys and values and returns a list of dictionaries with
    context, query and label triplets being the keys of each dict.
    """
    data = data["data"]
    qa_list = []

    for paragraphs in tqdm(data):

        for para in paragraphs["paragraphs"]:
            context = para["context"]

            for qa in para["qas"]:

                id = qa["id"]
                question = qa["question"]

                for answer in qa["answers"]:
                    qa_dict = {}
                    qa_dict["id"] = id
                    qa_dict["context"] = context
                    qa_dict["question"] = question
                    qa_dict["answer"] = answer["text"]
                    for sent in nlp(context, disable=["ner", "tagger"]).sents:
                        qa_dict["sentence"] = sent.text
                        qa_dict["is_correct"] = answer["text"] in sent.text
                        qa_list.append(qa_dict)

    return pd.DataFrame(qa_list)


def transform(filename: str):

    fp = os.path.join(os.path.dirname(__file__), filename)
    value = json.loads(open(fp).read())
    df = parse_data(value)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transformation step")
    parser.add_argument("--filename", "-f", type=str, default="data")
    parser.add_argument("--out", "-o", type=str, default="out")
    parser.add_argument("--prefix", "-p", type=str, default="prefix")

    args = parser.parse_args()

    df = transform(args.filename)
    df.to_pickle(f"{args.out}/{args.prefix}.pkl")
