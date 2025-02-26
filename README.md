# rag-tutorial-v2
Base on someone else now I forgot but I update code to have recent version of langchain and some other
improvement to make it easier to play around.

# Quick Start

```
python3 -mvenv .venv
. .venv/activate # assume u run bash, otherwise select the proper activate file in there
pip3 install -r requirements.txt
mkdir -p data/txt data/pdf 

```
Now load you PDF into `data/pdf` or structured text into `data/txt`. 

Edit `get_embedding_function.py` set ollama URL and the model u want to run the embedding. Mine one is good but
u have to tell ollama to pull it before.

Run `python3 populate_database.py` to populate.

Read and change code in `query_data.py`; edit as it suits and then run 

```
python3 query_data.py --m 'superdrew100-llama3-abliterated:latest' --q 'your question about your specicic data?'
# without using 
python3 query_data.py -c False --m 'superdrew100-llama3-abliterated:latest' --q 'your generic question?'
```
