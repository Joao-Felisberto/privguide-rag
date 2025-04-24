import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

def load_single_document(file_path: str) -> List[Document]:
    """Load a single document with error handling."""
    try:
        if file_path.endswith(('.json', '.txt')):
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path
            return docs
        return []
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def load_documents_parallel(directories: List[str]) -> List[Document]:
    """Load documents in parallel."""
    documents = []
    file_paths = []
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.json', '.txt')):
                    file_paths.append(os.path.join(root, file))
    
    with ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(load_single_document, path): path for path in file_paths}
        
        for future in as_completed(future_to_path):
            try:
                docs = future.result()
                documents.extend(docs)
            except Exception as e:
                path = future_to_path[future]
                print(f"Error processing {path}: {e}")
    
    return documents