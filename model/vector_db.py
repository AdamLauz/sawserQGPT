import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
import json
from pathlib import Path


# File paths for query engine and storage persistence
PERSIST_DIR = "../flask/storage"
SETTINGS_FILE = "../flask/storage/settings.json"
ARTICLES_DIR = "../resources"


def save_settings(settings):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)


def get_settings():
    return {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm": None,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "persist_dir": PERSIST_DIR,
        "top_k": 3  # Number of documents to retrieve in response to a query
    }


def build_index():
    # Read documents from 'articles' directory
    documents = SimpleDirectoryReader(ARTICLES_DIR).load_data()
    print(f"Number of documents: {len(documents)}")

    # Store documents into a vector database
    index = VectorStoreIndex.from_documents(documents)
    save_index(index)
    return index


def get_query_engine(settings):
    # set Settings
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings["embed_model"])
    Settings.llm = settings["llm"]
    Settings.chunk_size = settings["chunk_size"]
    Settings.chunk_overlap = settings["chunk_overlap"]

    # save settings file
    save_settings(settings)

    # Load existing index or build a new one if it doesn't exist
    index = load_index()
    if index is None:
        print("Index doesn't exist, building new index from the provided documents")
        index = build_index()

    # Configure retriever to fetch similar documents
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings["top_k"],
    )

    # Assemble query engine with retriever and postprocessor
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    return query_engine


def save_index(index):
    # Save the index to persistent storage
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def load_index():
    # Load the index from persistent storage if it exists
    if not os.path.exists(str(Path(PERSIST_DIR, "docstore.json"))):
        return None
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))


def get_context(query: str, query_engine, top_k):
    # Query the documents using the query engine
    response = query_engine.query(query)

    # Format the response to extract the relevant context
    context = "Context:\n"
    for i in range(top_k):
        context += response.source_nodes[i].text + "\n\n"

    return context


if __name__ == "__main__":
    # get settings
    settings = get_settings()

    # Initialize the query engine
    query_engine = get_query_engine(settings)

    # Get context for a specific query
    context = get_context("Who are the Circassians? Where did they come from?", query_engine, settings["top_k"])
    print(context)
