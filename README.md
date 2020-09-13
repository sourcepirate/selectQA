## QA

Selection based question answering model. Selects the most appropriate sentence for a question to answer.
Designed to run on few mb's of memory and very low CPU. 

Approch uses text shallow features, NER and POS tag on question and answer pair.  The result may highly depends on the words used on the phrases. If you need more sophasticated solution try using RNN / Attention model.

## Datasets used to prepare the model

* SQUAD
* TRECQA

## Training the model 

```
./download.sh

cd data

cd scripts/

python process.py --filename data/train-v2.0.json --out out/ --prefix train

python train.py ../out/train.pkl 100000 0.7

```

## Using a pretrained model

``` python
# download the model from the released.

from qa.reader import TextReader

reader = TextReader("path of the downloaded model")
reader.read(context, query)

```

