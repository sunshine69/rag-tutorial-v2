import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, help="Model.")
    parser.add_argument("--c", type=str, help="Context", default='true')
    parser.add_argument("--q", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.q
    model_name = args.m
    enable_context = args.c == 'true'
    query_rag(query_text, model_name=model_name, enable_context=enable_context)

def query_rag(query_text: str, model_name='qwen2.5-coder_7b-instruct-q2_K:custom', enable_context=True):
    # Prepare the DB.
    context_text = ''
    results = []
    if enable_context:
        print('Enabled context')
        PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=4)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    else:
        PROMPT_TEMPLATE = """
Answer the question: {question}
"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)
    model = OllamaLLM(base_url='http://192.168.20.49:11434',model=model_name)
    response_text = model.invoke(prompt)
    if enable_context:
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
    else:
        formatted_response = f"Response: {response_text}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
