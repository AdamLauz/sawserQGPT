import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage

# Number of documents to retrieve in response to a query
TOP_K = 3

# Import any embedding model from HuggingFace hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

# Disable language model (LLM) usage
Settings.llm = None
# Set chunk size and overlap for document processing
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# File paths for query engine and storage persistence
QUERY_ENGINE_FILE = '../flask/query_engine.pkl'
PERSIST_DIR = "../flask/storage"


def build_index():
    # Read documents from 'articles' directory
    documents = SimpleDirectoryReader("articles").load_data()
    print(f"Number of documents: {len(documents)}")

    # Store documents into a vector database
    index = VectorStoreIndex.from_documents(documents)
    save_index(index)
    return index


def get_query_engine():
    # Load existing index or build a new one if it doesn't exist
    index = load_index()
    if index is None:
        print("Index doesn't exist, building new index from the provided documents")
        index = build_index()

    # Configure retriever to fetch similar documents
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
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
    if not os.path.exists(PERSIST_DIR):
        return None
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))


def get_context(query: str, query_engine):
    # Query the documents using the query engine
    response = query_engine.query(query)

    # Format the response to extract the relevant context
    context = "Context:\n"
    for i in range(TOP_K):
        context += response.source_nodes[i].text + "\n\n"

    return context


if __name__ == "__main__":
    # Initialize the query engine
    query_engine = get_query_engine()

    # Get context for a specific query
    context = get_context("Who are the Circassians? Where did they come from?", query_engine)
    print(context)
