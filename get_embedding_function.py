from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(base_url='http://192.168.20.49:11434', model="all-minilm:l6-v2")
    return embeddings
