from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OllamaConfig:
    endpoint: str = "http://localhost:11434"
    model: str = "mistral"
    embedding_model: str = "all-minilm"
    stream_response: bool = False
    only_print_prompt: bool = False

@dataclass
class VectorStoreConfig:
    directory: str = "./chroma_db"
    collection_name: str = "privguide_collection"
    retriever_k: int = 5
    retriever_filter: Optional[dict] = None

@dataclass
class DocumentLoaderConfig:
    data_directories: Optional[List[str]] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class EmbeddingCacheConfig:
    directory: str = "./embedding_cache"

@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document_loader: DocumentLoaderConfig = field(default_factory=DocumentLoaderConfig)
    embedding_cache: EmbeddingCacheConfig = field(default_factory=EmbeddingCacheConfig)