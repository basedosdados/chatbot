import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_chroma_vector_store(host: str, port: str | int, collection: str) -> Chroma:
    """Creates a Chroma vector store from an HTTP client connected to a remote Chroma server

    Args:
        host (str): The Chroma server hostname.
        port (str | int): The Chroma server port.
        collection (str): The name of the Chroma collection.

    Returns:
        Chroma: An instance of a Chroma vector store.
    """
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
    """Returns a Chroma vector store if all parameters are provided, or `None` otherwise.

    Args:
        host (str | None): The Chroma server hostname.
        port (str | int | None): The Chroma server port.
        collection (str | None): The name of the Chroma collection.

    Returns:
        Chroma | None: An instance of a Chroma vector store, or `None`.
    """
    if host and port and collection:
        return get_chroma_vector_store(host, port, collection)
    return None
