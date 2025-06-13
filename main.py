import os
import time
import yaml
from typing import List, Optional
from argparse import ArgumentParser, BooleanOptionalAction
from config import AppConfig, OllamaConfig, VectorStoreConfig, DocumentLoaderConfig, EmbeddingCacheConfig
from llm.ollama import initialize_llm_components, create_workflow
from store.chroma import initialize_vector_store
from docs.document_loader import load_documents_parallel

def get_config_from_args(arg_list: Optional[List[str]] = None) -> AppConfig:
    parser = ArgumentParser(description="RAG System with Ollama and Chroma")
    
    # General config
    parser.add_argument('--test', action=BooleanOptionalAction,
                       help="Run every query in the tests folder, outputting their results")

    # Ollama config
    parser.add_argument('--prompt', 
                       help="File with the prompt to use, leave empty to interactively ask the user")
    parser.add_argument('--ollama-endpoint', 
                       help="Ollama server endpoint")
    parser.add_argument('--ollama-model', 
                       help="Ollama model to use")
    parser.add_argument('--embedding-model', 
                       help="Embedding model to use")
    parser.add_argument('--stream-response', action=BooleanOptionalAction,
                       help="Do not wait for the response to generate and stream it instead")
    parser.add_argument('--only-print-prompt', action=BooleanOptionalAction,
                       help="Do not send the augmented prompt to the llm and print it instead")
    parser.add_argument('--context-files', nargs='+',
                       help="The files to use as context of all the queries")

    # Vector store config
    parser.add_argument('--vector-dir', 
                       help="Directory for vector store persistence")
    parser.add_argument('--collection-name', 
                       help="Name for the Chroma collection")
    parser.add_argument('--retriever-k', type=int,
                       help="Number of documents to retrieve")
    
    # Document loader config
    parser.add_argument('--data-dirs', nargs='+',
                       help="Directories to load documents from")
    parser.add_argument('--chunk-size', type=int,
                       help="Chunk size for document splitting")
    parser.add_argument('--chunk-overlap', type=int,
                       help="Chunk overlap for document splitting")
    
    # Embedding cache config
    parser.add_argument('--embedding-dir',
                       help="Directory for embedding cache")
    
    args = parser.parse_args(arg_list)
    
    # Create config with defaults
    config = AppConfig()
    
    # Update with provided args
    if args.test:
        config.test = args.test
    if args.prompt:
        config.ollama.prompt = args.prompt
    if args.ollama_endpoint:
        config.ollama.endpoint = args.ollama_endpoint
    if args.ollama_model:
        config.ollama.model = args.ollama_model
    if args.embedding_model:
        config.ollama.embedding_model = args.embedding_model
    if args.stream_response:
        config.ollama.stream_response = args.stream_response
    if args.only_print_prompt:
        config.ollama.only_print_prompt = args.only_print_prompt
    if args.context_files:
        config.ollama.context_files = args.context_files
    
    if args.vector_dir:
        config.vector_store.directory = args.vector_dir
    if args.collection_name:
        config.vector_store.collection_name = args.collection_name
    if args.retriever_k:
        config.vector_store.retriever_k = args.retriever_k
    
    if args.data_dirs:
        config.document_loader.data_directories = args.data_dirs
    if args.chunk_size:
        config.document_loader.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.document_loader.chunk_overlap = args.chunk_overlap
    
    if args.embedding_dir:
        config.embedding_cache.directory = args.embedding_dir
    
    return config

def main(arg_l: Optional[List[str]] = None):
    config = get_config_from_args(arg_l)

    print("Initializing components...")
    llm, embeddings = initialize_llm_components(config.ollama, config.embedding_cache)
    
    add_documents = (
        not os.path.exists(config.vector_store.directory) and 
        config.document_loader.data_directories
    )
    documents = None
    if add_documents:
        print("Loading documents...")
        documents = load_documents_parallel(config.document_loader.data_directories)
        print(f"Loaded {len(documents)} documents")

    print("Creating vector store...")
    start_time = time.time()
    vector_store = initialize_vector_store(
        embeddings, 
        config.vector_store, 
        documents
    )
    print(f"Vector store created in {time.time() - start_time:.2f} seconds")

    print("Creating workflow...")
    graph = create_workflow(vector_store, config, llm, only_print_prompt=config.ollama.only_print_prompt, stream=config.ollama.stream_response)
    print("System ready!")

    if config.test:
        if not os.path.isdir("tests/prompts"):
            print("'tests/prompts' not found")
        if not os.path.isdir("tests/answers"):
            print("'tests/answers' not found")
        import json
        print("Running tests")

        prompts = os.listdir("tests/prompts")
        answers = os.listdir("tests/answers")
        for prompt_f in prompts:
            # TODO: make better caching method
            if prompt_f in answers:
                print(f"Skipping '{prompt_f}'...")
                continue
            print(f"Running test '{prompt_f}'...")
            with open(f"tests/prompts/{prompt_f}") as f:
                prompt = f.read()
            try:
                result = graph.invoke({"question": prompt})
                sources = '\n- '.join([doc.metadata['source'] for doc in result["context"]])
                with open(f"tests/answers/{prompt_f}", "w") as f:
                    f.write(f"{result['answer']}\n{sources}")
            except Exception as e:
                print(f"Error processing your question: {e}")
    else:
        print("Ready to answer questions. Type 'exit' to quit.")
        while True:
            if config.ollama.prompt is not None:
                question = config.ollama.prompt
            else:
                question = input("\nQuestion: ")
            if question.lower() in ('exit', 'q'):
                break
            
            try:
                result = graph.invoke({"question": question})
                # print("\nAnswer:")
                # print(result["answer"])
                if config.ollama.only_print_prompt:
                    print("\nPrompt:")
                    print(result["answer"])
                print("\nSources used:")
                for doc in result["context"]:
                    print(f"- {doc.metadata['source']}")
            except Exception as e:
                print(f"Error processing your question: {e}")

def test():    
    with open("tests/spec.yml") as f:
        metadata = yaml.safe_load(f)

    for test in metadata: 
        config = get_config_from_args(test["args"].split())
        print(config)
        return

        print("Initializing components...")
        llm, embeddings = initialize_llm_components(config.ollama, config.embedding_cache)
        
        add_documents = (
            not os.path.exists(config.vector_store.directory) and 
            config.document_loader.data_directories
        )
        documents = None
        if add_documents:
            print("Loading documents...")
            documents = load_documents_parallel(config.document_loader.data_directories)
            print(f"Loaded {len(documents)} documents")

        print("Creating vector store...")
        start_time = time.time()
        vector_store = initialize_vector_store(
            embeddings, 
            config.vector_store, 
            documents
        )
        print(f"Vector store created in {time.time() - start_time:.2f} seconds")

        print("Creating workflow...")
        graph = create_workflow(vector_store, config, llm, only_print_prompt=config.ollama.only_print_prompt, stream=config.ollama.stream_response)
        print("System ready!")

        answers = os.listdir("tests/answers")
        if test['prompt'] in answers:
            print(f"Skipping '{test['prompt']}'...")
            continue
        
        with open(f"tests/prompts/{test['prompt']}") as f:
            prompt = f.read()
        try:
            result = graph.invoke({"question": prompt})
            sources = ''.join([f"\n- {doc.metadata['source']}" for doc in result["context"]])
            with open(f"tests/answers/{test['prompt']}", "w") as f:
                f.write(f"{result['answer']}\n{sources}")
        except Exception as e:
            print(f"Error processing your question: {e}")


if __name__ == "__main__":
    config = get_config_from_args()

    if config.test:
        test()
    else:
        main()
    # main()
