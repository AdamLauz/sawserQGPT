"""Document Processing MCP Tool for better PDF and text extraction."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz  # PyMuPDF
import spacy
from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class DocumentProcessingRequest(BaseModel):
    """Request for document processing."""
    file_path: str = Field(..., description="Path to the document file")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")


class DocumentProcessingResponse(BaseModel):
    """Response for document processing."""
    text: str = Field(..., description="Extracted text")
    entities: List[Dict[str, Any]] = Field(default=[], description="Extracted entities")
    metadata: Dict[str, Any] = Field(default={}, description="Document metadata")
    success: bool = Field(..., description="Whether processing was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DocumentProcessorTool:
    """Document Processing MCP Tool using FastMCP."""
    
    def __init__(self):
        self.mcp = FastMCP("DocumentProcessor")
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools."""
        
        @self.mcp.tool()
        async def process_document(request: DocumentProcessingRequest) -> DocumentProcessingResponse:
            """Process a document (PDF, TXT, MD) and extract text and entities."""
            try:
                file_path = Path(request.file_path)
                
                if not file_path.exists():
                    return DocumentProcessingResponse(
                        text="",
                        entities=[],
                        metadata={},
                        success=False,
                        error=f"File not found: {file_path}"
                    )
                
                # Extract text based on file type
                if file_path.suffix.lower() == '.pdf':
                    text, metadata = await self._process_pdf(file_path)
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    text, metadata = await self._process_text(file_path)
                else:
                    return DocumentProcessingResponse(
                        text="",
                        entities=[],
                        metadata={},
                        success=False,
                        error=f"Unsupported file type: {file_path.suffix}"
                    )
                
                # Extract entities if requested
                entities = []
                if request.extract_entities and nlp and text:
                    entities = await self._extract_entities(text, str(file_path.name))
                
                return DocumentProcessingResponse(
                    text=text,
                    entities=entities,
                    metadata=metadata,
                    success=True
                )
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                return DocumentProcessingResponse(
                    text="",
                    entities=[],
                    metadata={},
                    success=False,
                    error=str(e)
                )
        
        @self.mcp.tool()
        async def process_directory(directory_path: str, extract_entities: bool = True) -> Dict[str, Any]:
            """Process all documents in a directory."""
            try:
                dir_path = Path(directory_path)
                if not dir_path.exists():
                    return {"success": False, "error": f"Directory not found: {directory_path}"}
                
                results = []
                supported_extensions = ['.pdf', '.txt', '.md']
                
                for file_path in dir_path.iterdir():
                    if file_path.suffix.lower() in supported_extensions:
                        request = DocumentProcessingRequest(
                            file_path=str(file_path),
                            extract_entities=extract_entities
                        )
                        result = await self.process_document(request)
                        results.append({
                            "file": str(file_path.name),
                            "success": result.success,
                            "text_length": len(result.text),
                            "entities_count": len(result.entities),
                            "error": result.error
                        })
                
                return {
                    "success": True,
                    "processed_files": len(results),
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error processing directory: {e}")
                return {"success": False, "error": str(e)}
    
    async def _process_pdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Process PDF file using PyMuPDF for better text extraction."""
        try:
            doc = fitz.open(file_path)
            text_parts = []
            metadata = {
                "page_count": doc.page_count,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            doc.close()
            full_text = "\n\n".join(text_parts)
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return "", {"error": str(e)}
    
    async def _process_text(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Process text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {
                "file_size": file_path.stat().st_size,
                "encoding": "utf-8"
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return "", {"error": str(e)}
    
    async def _extract_entities(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER."""
        try:
            if not nlp:
                return []
            
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": ent.text,
                    "source": source,
                    "confidence": 1.0  # spaCy doesn't provide confidence scores
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def start(self):
        """Start the MCP tool."""
        await self.mcp.run()


# Global instance
document_processor_tool = DocumentProcessorTool()
