## QA

Selection based question answering model. Selects the most appropriate sentence for a question to answer.

## Datasets used to prepare the model

* SQUAD
* TRECQA

## Usage

```python

from qa.reader import reader

context = "context sentense here"
query = "what is this ?"

ans = reader.read(context, query)

```

## Training the model 

```
./download.sh

cd data

# download glove model

cd scripts/

python process.py --filename data/train-v2.0.json --out out/ --prefix train

python train.py ../out/train.pkl

```

