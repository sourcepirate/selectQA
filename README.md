## QA

Selection based question answering model. Selects the most appropriate sentence for a question to answer.

## Usage

```python

from qa.reader import reader

context = "context sentense here"
query = "what is this ?"

ans = reader.read(context, query)

```
