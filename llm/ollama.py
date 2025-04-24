from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from config import OllamaConfig, EmbeddingCacheConfig

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def initialize_llm_components(ollama_config: OllamaConfig, cache_config: EmbeddingCacheConfig):
    """Initialize Ollama components with type-safe configuration."""
    store = LocalFileStore(cache_config.directory)
    embeddings = OllamaEmbeddings(model=ollama_config.embedding_model)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )
    llm = OllamaLLM(
        base_url=ollama_config.endpoint,
        model=ollama_config.model
    )
    return llm, cached_embeddings

def create_workflow(retriever, llm, prompt_template: str = None):
    """Create and return the LangGraph workflow with type safety."""
    template = prompt_template or """
    You are an expert in JSON schemas. Use the following context to answer the question.  
    Cite the source of each fact you mention by referencing the source filename.  
    Context:  
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def retrieve(state: State):
        print("Retrieving relevant documents...")
        question = state["question"]
        docs = retriever.invoke(question)
        return {"context": docs, "question": question}

    def generate(state: State):
        print("Generating answer...")
        question = state["question"]
        context = state["context"]
        
        context_content = "\n\n---\n\n".join(
            f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" 
            for doc in context
        )
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        answer = chain.invoke({"context": context_content, "question": question})
        
        return {"answer": answer, "context": context}

    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()