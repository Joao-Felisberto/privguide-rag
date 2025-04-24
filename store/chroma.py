from langchain_chroma import Chroma
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from config import VectorStoreConfig
import os

def initialize_vector_store(
    embeddings,
    config: VectorStoreConfig,
    documents: Optional[List[Document]] = None
) -> tuple[Chroma, VectorStoreRetriever]:
    """Initialize and return the Chroma vector store with type-safe config."""
    add_documents = not os.path.exists(config.directory) and documents is not None
    
    vector_store = Chroma(
        collection_name=config.collection_name,
        persist_directory=config.directory,
        embedding_function=embeddings
    )
    
    if add_documents:
        vector_store.add_documents(documents=documents)
    
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": config.retriever_k,
            "filter": config.retriever_filter
        }
    )
    
    return vector_store, retriever