"""Knowledge Graph Service - FastMCP-based service for knowledge graph operations."""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

import networkx as nx
import spacy
from .config import KnowledgeGraphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config
config = KnowledgeGraphConfig()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class KnowledgeGraphService:
    """Knowledge graph service for entity extraction and graph operations."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.kg_dir = Path(config.knowledge_graph_dir)
        self.kg_dir.mkdir(exist_ok=True)
    
    async def extract_entities(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """Extract entities and relations from text."""
        try:
            if not nlp:
                return {"entities": [], "relations": []}
            
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": ent.text,
                    "source": source
                })
            
            # Extract relations using simple patterns
            relations = self._extract_relations(doc, source)
            
            return {"entities": entities, "relations": relations}
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "relations": []}
    
    def _extract_relations(self, doc, source: str) -> List[Dict[str, Any]]:
        """Extract relations using simple patterns."""
        relations = []
        
        # Simple pattern: look for entities that are close to each other
        entities = list(doc.ents)
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                # Check if entities are close in the text
                if abs(ent1.end - ent2.start) < 50:  # Within 50 characters
                    relations.append({
                        "source": ent1.text,
                        "target": ent2.text,
                        "relation": "related",
                        "confidence": 0.5,
                        "source_entity": ent1.text,
                        "target_entity": ent2.text
                    })
        
        return relations
    
    async def add_entities_to_graph(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Add entities to the knowledge graph."""
        try:
            added_count = 0
            for entity in entities:
                # Add entity as node
                node_id = f"{entity['text']}_{entity['label']}"
                self.graph.add_node(
                    node_id,
                    text=entity['text'],
                    label=entity['label'],
                    description=entity.get('description', ''),
                    source=entity.get('source', 'unknown')
                )
                added_count += 1
            
            return {"status": "success", "added_entities": added_count}
            
        except Exception as e:
            logger.error(f"Error adding entities to graph: {e}")
            return {"status": "error", "error": str(e)}
    
    async def add_relations_to_graph(self, relations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Add relations to the knowledge graph."""
        try:
            added_count = 0
            for relation in relations:
                source_id = f"{relation['source']}_{relation.get('label', 'ENTITY')}"
                target_id = f"{relation['target']}_{relation.get('label', 'ENTITY')}"
                
                # Add edge
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation=relation['relation'],
                    confidence=relation.get('confidence', 0.5)
                )
                added_count += 1
            
            return {"status": "success", "added_relations": added_count}
            
        except Exception as e:
            logger.error(f"Error adding relations to graph: {e}")
            return {"status": "error", "error": str(e)}
    
    async def query_graph(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Query the knowledge graph."""
        try:
            # Simple keyword matching for now
            query_lower = query.lower()
            relevant_entities = []
            
            for node, data in self.graph.nodes(data=True):
                if query_lower in data.get('text', '').lower() or query_lower in data.get('description', '').lower():
                    relevant_entities.append({
                        "text": data.get('text', ''),
                        "label": data.get('label', ''),
                        "description": data.get('description', ''),
                        "source": data.get('source', '')
                    })
            
            # Get reasoning path (simple for now)
            reasoning_path = []
            if relevant_entities:
                reasoning_path.append({
                    "step": 1,
                    "action": "keyword_match",
                    "entities": relevant_entities[:max_results]
                })
            
            return {
                "answer": f"Found {len(relevant_entities)} relevant entities",
                "relevant_entities": relevant_entities[:max_results],
                "reasoning_path": reasoning_path
            }
            
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return {"answer": "Error querying knowledge graph", "relevant_entities": [], "reasoning_path": []}
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            return {
                "status": "ready",
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
            }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"status": "error", "error": str(e)}


# Global knowledge graph service instance
kg_service = KnowledgeGraphService()

# Create FastMCP application
mcp = FastMCP("Knowledge Graph Service")


@mcp.tool()
async def extract_entities(
    text: str = Field(..., description="Text to extract entities from"),
    source: str = Field(default="unknown", description="Source of the text")
) -> Dict[str, Any]:
    """Extract entities and relations from text."""
    try:
        result = await kg_service.extract_entities(text, source)
        return result
    except Exception as e:
        logger.error(f"Error in extract_entities: {e}")
        raise


@mcp.tool()
async def add_entities(
    entities: List[Dict[str, Any]] = Field(..., description="Entities to add to the graph")
) -> Dict[str, Any]:
    """Add entities to the knowledge graph."""
    try:
        result = await kg_service.add_entities_to_graph(entities)
        return result
    except Exception as e:
        logger.error(f"Error in add_entities: {e}")
        raise


@mcp.tool()
async def add_relations(
    relations: List[Dict[str, Any]] = Field(..., description="Relations to add to the graph")
) -> Dict[str, Any]:
    """Add relations to the knowledge graph."""
    try:
        result = await kg_service.add_relations_to_graph(relations)
        return result
    except Exception as e:
        logger.error(f"Error in add_relations: {e}")
        raise


@mcp.tool()
async def query_graph(
    query: str = Field(..., description="Query string"),
    max_results: int = Field(default=10, description="Maximum number of results")
) -> Dict[str, Any]:
    """Query the knowledge graph."""
    try:
        result = await kg_service.query_graph(query, max_results)
        return result
    except Exception as e:
        logger.error(f"Error in query_graph: {e}")
        raise


@mcp.tool()
async def get_graph_stats() -> Dict[str, Any]:
    """Get knowledge graph statistics."""
    try:
        stats = await kg_service.get_graph_stats()
        return stats
    except Exception as e:
        logger.error(f"Error in get_graph_stats: {e}")
        raise


@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check for the knowledge graph service."""
    stats = await kg_service.get_graph_stats()
    return {
        "status": "healthy" if stats["status"] == "ready" else "unhealthy",
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run as FastAPI app
    app = mcp.create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True
    )