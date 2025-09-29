# Agent - MCP Tools and RAG Agent

This directory contains the MCP (Model Context Protocol) tools and the RAG agent that orchestrates the knowledge graph operations.

## Structure

```
agent/
├── rag_agent.py              # Main RAG agent (FastMCP with integrated LLM)
├── knowledge_graph_tool.py   # Knowledge graph MCP tool
├── document_processor_tool.py # Document processing MCP tool
├── resources/                # Document storage
├── storage/                  # Vector database storage
└── requirements.txt          # Agent dependencies
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server to use the agent
python ../start_server.py
```

## Components

### RAG Agent
- FastMCP-based agent that orchestrates the entire RAG process
- Uses Knowledge Graph tool internally
- Manages LLM interactions and response generation

### Knowledge Graph Tool
- NetworkX-based MCP tool for knowledge graph operations
- Entity extraction from text
- Knowledge graph queries and reasoning

### Document Processor Tool
- Processes documents from resources directory
- Extracts entities and relationships
- Populates knowledge graph
