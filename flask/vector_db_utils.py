import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
import json
import torch.multiprocessing as mp


PERSIST_DIR = "./storage"
SETTINGS_FILE = "./storage/settings.json"


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return settings
    return None


def load_index():
    # Load the index from persistent storage if it exists
    if not os.path.exists(PERSIST_DIR):
        return None
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))


def get_query_engine(settings):
    # set Settings
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings["embed_model"], device='cuda')
    Settings.llm = settings["llm"]
    Settings.chunk_size = settings["chunk_size"]
    Settings.chunk_overlap = settings["chunk_overlap"]

    # Load existing index or build a new one if it doesn't exist
    index = load_index()
    if index is None:
        print("Index doesn't exist.")
        return None

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


def get_context(query: str, query_engine, top_k):
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Query the documents using the query engine
    response = query_engine.query(query)

    # Format the response to extract the relevant context
    context = "Context:\n"
    for i in range(top_k):
        context += response.source_nodes[i].text + "\n\n"

    return context
