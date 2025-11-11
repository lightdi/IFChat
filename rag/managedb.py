import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")
embedding_function = embedding_functions.DefaultEmbeddingFunction
collection = client.get_or_create_collection(name="ifchat_docs", embedding_function=embedding_function)

def add_document(text: str, source: str):
    collection.add(documents=[text], metadatas=[{"source": source}], ids=[source])

def search_similar(query:str):
    return collection.query(query_texts=[query], n_results=3)

