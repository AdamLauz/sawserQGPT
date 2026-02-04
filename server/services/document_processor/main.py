"""Document Processor Service - FastMCP-based service for document processing."""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

import fitz  # PyMuPDF
from .config import DocumentProcessorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config
config = DocumentProcessorConfig()


class DocumentProcessorService:
    """Document processor service for handling document operations."""
    
    def __init__(self):
        self.resources_dir = Path(config.resources_dir)
        self.resources_dir.mkdir(exist_ok=True)
        self.processing_status = {}
    
    async def process_directory(self, directory_path: str, extract_entities: bool = True) -> Dict[str, Any]:
        """Process all documents in a directory."""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                return {"success": False, "error": f"Directory {directory_path} does not exist"}
            
            processed_files = 0
            results = []
            
            # Get all PDF files in directory
            pdf_files = list(directory.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                try:
                    result = await self.process_file(str(pdf_file), extract_entities)
                    results.append(result)
                    if result["success"]:
                        processed_files += 1
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    results.append({
                        "file": str(pdf_file),
                        "success": False,
                        "error": str(e),
                        "entities_count": 0
                    })
            
            return {
                "success": True,
                "processed_files": processed_files,
                "total_files": len(pdf_files),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_file(self, file_path: str, extract_entities: bool = True) -> Dict[str, Any]:
        """Process a single document file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            # Extract text from PDF
            text = await self._extract_text_from_pdf(str(file_path))
            
            # Extract entities if requested
            entities_count = 0
            if extract_entities and text:
                entities_count = await self._extract_entities_from_text(text)
            
            return {
                "file": str(file_path),
                "success": True,
                "text_length": len(text),
                "entities_count": entities_count,
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file": str(file_path),
                "success": False,
                "error": str(e),
                "entities_count": 0
            }
    
    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    async def _extract_entities_from_text(self, text: str) -> int:
        """Extract entities from text (simplified version)."""
        try:
            # Simple entity extraction - count capitalized words
            words = text.split()
            entities = [word for word in words if word[0].isupper() and len(word) > 2]
            return len(set(entities))  # Return unique entities count
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return 0
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "status": "ready",
            "processed_files": len(list(self.resources_dir.glob("*.pdf"))),
            "resources_dir": str(self.resources_dir)
        }


# Global document processor service instance
doc_service = DocumentProcessorService()

# Create FastMCP application
mcp = FastMCP("Document Processor Service")


@mcp.tool()
async def process_directory(
    directory_path: str = Field(..., description="Path to directory to process"),
    extract_entities: bool = Field(default=True, description="Whether to extract entities")
) -> Dict[str, Any]:
    """Process all documents in a directory."""
    try:
        result = await doc_service.process_directory(directory_path, extract_entities)
        return result
    except Exception as e:
        logger.error(f"Error in process_directory: {e}")
        raise


@mcp.tool()
async def process_file(
    file_path: str = Field(..., description="Path to file to process"),
    extract_entities: bool = Field(default=True, description="Whether to extract entities")
) -> Dict[str, Any]:
    """Process a single document file."""
    try:
        result = await doc_service.process_file(file_path, extract_entities)
        return result
    except Exception as e:
        logger.error(f"Error in process_file: {e}")
        raise


@mcp.tool()
async def get_processing_status() -> Dict[str, Any]:
    """Get processing status."""
    try:
        status = await doc_service.get_processing_status()
        return status
    except Exception as e:
        logger.error(f"Error in get_processing_status: {e}")
        raise


@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check for the document processor service."""
    status = await doc_service.get_processing_status()
    return {
        "status": "healthy",
        "processed_files": status["processed_files"],
        "resources_dir": status["resources_dir"]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run as FastAPI app
    app = mcp.create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=True
    )