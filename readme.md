# RAG Tutorial Run Through ++

## What's Here

Following a rag tutorial to get up to speed ([this one](https://www.youtube.com/watch?v=tcqEUSNCn8I)).

Adapting it to use DSPy as the interface with the LLM.


## To Try it

### 1. Install deps and have api key ready
```
pip install -r requirements.txt
export OPENAI_API_KEY=<your-key>
```

### 2.load the files to db
```
python create_database.py
```

### 3. try out the search
```
python main.py -q "when did the US declare independence?"
```