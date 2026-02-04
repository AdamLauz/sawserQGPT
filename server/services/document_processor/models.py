"""Document Processor Service models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ProcessDirectoryRequest(BaseModel):
    """Request to process directory."""
    directory_path: str = Field(..., description="Path to directory to process")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")


class ProcessDirectoryResponse(BaseModel):
    """Response for directory processing."""
    success: bool = Field(..., description="Whether processing was successful")
    processed_files: int = Field(default=0, description="Number of files processed")
    total_files: int = Field(default=0, description="Total number of files")
    results: List[Dict[str, Any]] = Field(default=[], description="Processing results")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ProcessFileRequest(BaseModel):
    """Request to process single file."""
    file_path: str = Field(..., description="Path to file to process")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")


class ProcessFileResponse(BaseModel):
    """Response for file processing."""
    success: bool = Field(..., description="Whether processing was successful")
    file: str = Field(..., description="File path")
    text_length: int = Field(default=0, description="Length of extracted text")
    entities_count: int = Field(default=0, description="Number of entities extracted")
    text_preview: str = Field(default="", description="Preview of extracted text")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ProcessingStatusResponse(BaseModel):
    """Response for processing status."""
    status: str = Field(..., description="Processing status")
    processed_files: int = Field(..., description="Number of processed files")
    resources_dir: str = Field(..., description="Resources directory path")
