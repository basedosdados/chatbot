import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_chroma_vector_store(host: str, port: str | int, collection: str) -> Chroma:
    client = chromadb.HttpClient(
        host=host,
        port=port,
    )

    return Chroma(
        client=client,
        collection_name=collection,
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

def get_chroma_or_none(
    host: str | None,
    port: str | int | None,
    collection: str | None
) -> Chroma | None:
    if host and port and collection:
        return get_chroma_vector_store(host, port, collection)
    return None
