# RAG Tutorial Run Through ++

## What's Here

### Inspo

Following a rag tutorial to get up to speed ([this one](https://www.youtube.com/watch?v=tcqEUSNCn8I)).


### Adjustments

1. Use DSPy to organise LLM interaction code
2. move chromadb interactions into DSPy module
2. use DSPy optimisation feature to improve performance
    1. using gpt4 to assess similarity between real and predicted answer
3. made a simple set of examples in the [data/testing_data.csv](data/testing_data.csv) file

## Performance on books DB

books found in [data/books/](data/books/).

| Run | Performance |
|----------|----------|
| gpt-3 | 61% |
| gpt-3 optimised | 74% |
| gpt-3 CoT | 64% |
| gpt-3 CoT optimised | 73% |
| gpt-4o CoT | 73% |
| gpt-4o CoT optimised | 82% |

Seems CoT isn't that important here.

But we see prompt optimisation can easily improve our performance, even if the
training data I created isn't that good.



## How to run

### 1. Install deps and have api key ready
```
pip install -r requirements.txt
export OPENAI_API_KEY=<your-key>
```

### 2.Create the files to db
```
python -m create_database
```

### 3. Try out the search
```
python -m main -q "when did the US declare independence?"
```

### 4. Compare optimised versus unoptimised performance
```
# currently set to run with gpt-3
python -m optimise
```