"""Knowledge Graph MCP Tool using FastMCP."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import spacy
import networkx as nx
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from server.config import settings

logger = logging.getLogger(__name__)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction."""
    text: str = Field(..., description="Text to extract entities from")
    source: str = Field(default="unknown", description="Source of the text")


class EntityExtractionResponse(BaseModel):
    """Response for entity extraction."""
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    relations: List[Dict[str, Any]] = Field(..., description="Extracted relations")


class KnowledgeGraphQueryRequest(BaseModel):
    """Request for knowledge graph query."""
    query: str = Field(..., description="Query to search in knowledge graph")
    max_results: int = Field(default=10, description="Maximum number of results")


class KnowledgeGraphQueryResponse(BaseModel):
    """Response for knowledge graph query."""
    answer: str = Field(..., description="Answer based on knowledge graph")
    relevant_entities: List[Dict[str, Any]] = Field(..., description="Relevant entities")
    reasoning_path: List[Dict[str, Any]] = Field(..., description="Reasoning path through graph")


class KnowledgeGraphTool:
    """Knowledge Graph MCP Tool using FastMCP."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.kg_dir = Path(settings.knowledge_graph_dir)
        self.kg_dir.mkdir(exist_ok=True)
        self.mcp = FastMCP("KnowledgeGraph")
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools."""
        
        @self.mcp.tool()
        async def extract_entities(request: EntityExtractionRequest) -> EntityExtractionResponse:
            """Extract entities and relations from text using spaCy NER."""
            try:
                if not nlp:
                    return EntityExtractionResponse(entities=[], relations=[])
                
                # Process text with spaCy
                doc = nlp(request.text)
                
                # Extract entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": ent.text,
                        "source": request.source
                    })
                
                # Extract relations using simple patterns
                relations = self._extract_relations(doc, request.source)
                
                return EntityExtractionResponse(entities=entities, relations=relations)
                
            except Exception as e:
                logger.error(f"Error extracting entities: {e}")
                return EntityExtractionResponse(entities=[], relations=[])
        
        @self.mcp.tool()
        async def add_entities_to_graph(entities: List[Dict[str, Any]]) -> Dict[str, str]:
            """Add extracted entities to the knowledge graph."""
            try:
                added_count = 0
                for entity in entities:
                    entity_id = f"{entity['text']}_{entity['label']}"
                    
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            text=entity['text'],
                            label=entity['label'],
                            description=entity.get('description', ''),
                            sources=[entity.get('source', 'unknown')]
                        )
                        added_count += 1
                    else:
                        # Add source to existing entity
                        if 'sources' not in self.graph.nodes[entity_id]:
                            self.graph.nodes[entity_id]['sources'] = []
                        if entity.get('source') not in self.graph.nodes[entity_id]['sources']:
                            self.graph.nodes[entity_id]['sources'].append(entity.get('source'))
                
                # Save graph
                await self._save_graph()
                
                return {"status": "success", "added_entities": added_count}
                
            except Exception as e:
                logger.error(f"Error adding entities to graph: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.mcp.tool()
        async def query_knowledge_graph(request: KnowledgeGraphQueryRequest) -> KnowledgeGraphQueryResponse:
            """Query the knowledge graph for relevant information."""
            try:
                # Simple keyword-based search
                query_lower = request.query.lower()
                relevant_entities = []
                
                for node_id, data in self.graph.nodes(data=True):
                    if any(keyword in data['text'].lower() for keyword in query_lower.split()):
                        relevant_entities.append({
                            "text": data['text'],
                            "label": data['label'],
                            "description": data.get('description', ''),
                            "sources": data.get('sources', [])
                        })
                
                # Limit results
                relevant_entities = relevant_entities[:request.max_results]
                
                # Generate simple answer
                if relevant_entities:
                    answer = f"Found {len(relevant_entities)} relevant entities: "
                    answer += ", ".join([f"{ent['text']} ({ent['label']})" for ent in relevant_entities])
                else:
                    answer = "No relevant entities found in the knowledge graph."
                
                return KnowledgeGraphQueryResponse(
                    answer=answer,
                    relevant_entities=relevant_entities,
                    reasoning_path=[]
                )
                
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return KnowledgeGraphQueryResponse(
                    answer="Error querying knowledge graph.",
                    relevant_entities=[],
                    reasoning_path=[]
                )
        
        @self.mcp.tool()
        async def get_graph_stats() -> Dict[str, Any]:
            """Get knowledge graph statistics."""
            try:
                return {
                    "nodes": len(self.graph.nodes),
                    "edges": len(self.graph.edges),
                    "entity_types": list(set(data.get('label', 'UNKNOWN') for _, data in self.graph.nodes(data=True))),
                    "status": "ready"
                }
            except Exception as e:
                logger.error(f"Error getting graph stats: {e}")
                return {"status": "error", "message": str(e)}
    
    def _extract_relations(self, doc, source: str) -> List[Dict[str, Any]]:
        """Extract relations using simple patterns."""
        relations = []
        
        # Simple pattern: entity1 - relation - entity2
        for i, token in enumerate(doc):
            if token.pos_ in ['VERB', 'ADP']:  # Verbs and prepositions often indicate relations
                # Look for entities around this token
                prev_ent = None
                next_ent = None
                
                # Find previous entity
                for j in range(i-1, max(0, i-5), -1):
                    if doc[j].ent_type_:
                        prev_ent = doc[j].text
                        break
                
                # Find next entity
                for j in range(i+1, min(len(doc), i+5)):
                    if doc[j].ent_type_:
                        next_ent = doc[j].text
                        break
                
                if prev_ent and next_ent:
                    relations.append({
                        "subject": prev_ent,
                        "predicate": token.text,
                        "object": next_ent,
                        "confidence": 0.7,
                        "source": source
                    })
        
        return relations
    
    async def _save_graph(self):
        """Save the knowledge graph to disk."""
        try:
            graph_file = self.kg_dir / "knowledge_graph.json"
            graph_data = nx.node_link_data(self.graph)
            
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    async def _load_graph(self):
        """Load the knowledge graph from disk."""
        try:
            graph_file = self.kg_dir / "knowledge_graph.json"
            if graph_file.exists():
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
    
    async def start(self):
        """Start the MCP tool."""
        await self._load_graph()
        await self.mcp.run()


# Global instance
knowledge_graph_tool = KnowledgeGraphTool()
