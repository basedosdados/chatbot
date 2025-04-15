from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_chroma_client(url: str, collection: str) -> Chroma:
    if not isinstance(url, str):
        raise TypeError(f"Chroma client URL must be str, got {type(url)}")

    if not isinstance(collection, str):
        raise TypeError(f"Chroma collection name must be str, got {type(collection)}")

    return Chroma(
        collection_name=collection,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=url,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
