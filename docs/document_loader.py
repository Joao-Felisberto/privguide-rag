import os
import json
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

def __get_type_helper(prop: dict) -> str:
    if "type" in prop and prop["type"] == "array":
        if "type" in prop["items"]:
            return f"{prop['items']['type']}[]"
        
        t = prop['items']['$ref'].split('/')[-1]
        return f"{t}[]"
    
    if "$ref" in prop:
        return prop['$ref'].split('/')[-1]

    if "type" in prop:
        return prop["type"]

    # print(prop)

def __get_schema_type(prop: dict) -> str:
    if "anyOf" in prop:
        return "|".join((__get_type_helper(p) for p in prop["anyOf"]))
    return __get_type_helper(prop)

def process_json_schema(data: dict) -> str:
    """
    Convert the big json schema into a more compact representation.
    The object representation will be a list of properties and their subproperties, mapped to the type.

    Ex: `Car(wheels:Wheel[], n_seats:int)`
    """
    props = {
        p: {
            (pp, __get_schema_type(data["definitions"][p]["properties"][pp]))
            for pp in data["definitions"][p]["properties"]
        }
        for p in data["definitions"]
    }

    # return props
    res = ""

    for prop in props:
        res += prop
        sub_props = props[prop]
        res += '(' + ','.join((f"{name}:{_type}" for name, _type in sub_props)) + ')'
        res += '\n'

    return res


def load_single_document(file_path: str) -> List[Document]:
    """Load a single document with error handling."""
    try:
        if file_path.endswith('schema.json'):
            print(f"A schema: '{file_path}'")
            with open(file_path) as f:
                f_data = json.loads(f.read())
            compact_schema = process_json_schema(f_data)
            processed_doc = Document(
                page_content=compact_schema,
                metadata={"source": file_path},
            )
            return [processed_doc]
        else:
            print(f"NOT a schema: '{file_path}'")
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path
            return docs
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
