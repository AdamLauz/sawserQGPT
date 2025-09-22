"""Modern vector database service with async support."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Disable OpenAI by setting environment variable
os.environ["OPENAI_API_KEY"] = ""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import settings
from app.exceptions import VectorDBError

logger = logging.getLogger(__name__)


class VectorService:
    """Modern vector database service with async support."""
    
    def __init__(self):
        self.query_engine = None
        self.index = None
        self._is_ready = False # This is a flag that indicates if the vector database is ready. Meaning it was initialized successfully.
    
    async def initialize(self) -> None:
        """Initialize the vector database service."""
        try:
            logger.info("Initializing vector database service")
            
            # Set up LlamaIndex settings - explicitly use open source models
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding_model_name,
                device=settings.device
            )
            Settings.chunk_size = settings.chunk_size # Chunk size for the vector database. This is the number of tokens in each chunk.
            Settings.chunk_overlap = settings.chunk_overlap # Chunk overlap for the vector database. This is the number of tokens to overlap between chunks.
            
            # Ensure no OpenAI is used
            Settings.llm = None  # We handle LLM separately in our service
            
            # Load or build index
            self.index = await self._load_or_build_index()
            
            # Create query engine
            self.query_engine = await self._create_query_engine()
            
            self._is_ready = True
            logger.info("Vector database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise VectorDBError(f"Failed to initialize vector database: {e}")
    
    async def _load_or_build_index(self) -> VectorStoreIndex:
        """Load existing index or build a new one."""
        try:
            # Try to load existing index
            if await self._index_exists():
                logger.info("Loading existing vector index")
                return await self._load_index()
            else:
                logger.info("Building new vector index from documents")
                return await self._build_index()
                
        except Exception as e:
            logger.error(f"Error loading/building index: {e}")
            raise VectorDBError(f"Failed to load/build index: {e}")
    
    async def _index_exists(self) -> bool:
        """Check if index exists in storage."""
        persist_path = Path(settings.persist_dir)
        return persist_path.exists() and (persist_path / "docstore.json").exists()
    
    async def _load_index(self) -> VectorStoreIndex:
        """Load index from persistent storage."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=settings.persist_dir) # Create a storage context that knows where to find the vector database.
            return load_index_from_storage(storage_context) # Load the actual index from that storage location.
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return await self._build_index() # If loading fails, build a new index. TODO: Is this the best way to handle this?
    
    async def _build_index(self) -> VectorStoreIndex:
        """Build index from documents."""
        try:
            # Load documents
            documents = SimpleDirectoryReader(settings.resources_dir).load_data() # Load the documents from the resources directory. Documents can be PDFs, txt, docx, etc.
            logger.info(f"Loaded {len(documents)} documents")
            
            if not documents: # If no documents are found, raise an error.
                raise VectorDBError("No documents found in resources directory")
            
            # Build index
            index = VectorStoreIndex.from_documents(documents) # Build the index from the documents.
            
            # Save index
            await self._save_index(index) # Save the index to the persistent storage. Wait for it to complete before returning. (await means "wait for the task to complete")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise VectorDBError(f"Failed to build index: {e}")
    
    async def _save_index(self, index: VectorStoreIndex) -> None:
        """Save index to persistent storage."""
        try:
            index.storage_context.persist(persist_dir=settings.persist_dir) # this should create files such as docstore.json, index.json, etc.
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise VectorDBError(f"Failed to save index: {e}")
    
    async def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create query engine with retriever and postprocessor."""
        try:
            # Create retriever
            retriever = VectorIndexRetriever( # Retriever is a class that allows us to retrieve the most relevant documents from the vector database.
                index=self.index,
                similarity_top_k=settings.top_k # Top-k sampling, the number of documents to retrieve from the vector database. Similarity is calculated using the cosine similarity.
            )
            
            # Create postprocessor
            postprocessor = SimilarityPostprocessor( # Postprocessor is a class that allows us to postprocess the retrieved documents.
                similarity_cutoff=settings.similarity_cutoff # Similarity cutoff for the vector database. This is the similarity score below which documents are considered similar. In other words, a minimum similarity score for the documents to be considered.
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine( # Query engine is a class that allows us to query the vector database.
                retriever=retriever,
                node_postprocessors=[postprocessor]
            )
            
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to create query engine: {e}")
            raise VectorDBError(f"Failed to create query engine: {e}")
    
    async def get_context(self, query: str) -> Tuple[str, List[str], List[float]]:
        """Get relevant context for a query.
        A query is a string that is used to retrieve the most relevant documents from the vector database.
        Query is usually the user's prompt or question."""
        if not self._is_ready: # TODO: I think it would be clearer to use _is_initialized instead of _is_ready.
            await self.initialize() # If the vector database is not ready, initialize it.
        
        try:
            # Query the vector database
            response = self.query_engine.query(query) # Query the vector database. 
            
            # Extract context and metadata
            context_parts = []
            source_texts = []
            similarity_scores = []
            
            for i, node in enumerate(response.source_nodes): # response.source_nodes is a list of nodes that are the most relevant documents from the vector database.
                context_parts.append(node.text)
                source_texts.append(node.text) # why is this the same as context_parts? TODO: Check this.
                similarity_scores.append(node.score)
            
            context = "Context:\n" + "\n\n".join(context_parts) # Context is a string that is the concatenation of the most relevant documents from the vector database.
            
            return context, source_texts, similarity_scores
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            # Return empty context instead of failing
            return "", [], []
    
    async def rebuild_index(self) -> None:
        """Rebuild the vector index from scratch.
        This should be used when the vector database is corrupted or when the documents have changed."""
        try:
            logger.info("Rebuilding vector index")
            
            # Remove existing index
            persist_path = Path(settings.persist_dir)
            if persist_path.exists():
                import shutil # shutil is a module that allows us to delete a directory.
                shutil.rmtree(persist_path)
            
            # Rebuild index
            self.index = await self._build_index()
            self.query_engine = await self._create_query_engine()
            
            logger.info("Vector index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise VectorDBError(f"Failed to rebuild index: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Check if the vector database is ready."""
        return self._is_ready
    
    async def get_stats(self) -> dict:
        """Get vector database statistics."""
        if not self._is_ready:
            return {"status": "not_ready"}
        
        try:
            # Get basic stats
            stats = {
                "status": "ready",
                "index_exists": await self._index_exists(),
                "top_k": settings.top_k,
                "similarity_cutoff": settings.similarity_cutoff,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
