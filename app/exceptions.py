"""Custom exceptions for the application."""


class SawserQGPTError(Exception):
    """Base exception for SawserQ GPT application."""
    pass


class ModelLoadError(SawserQGPTError):
    """Raised when model loading fails."""
    pass


class VectorDBError(SawserQGPTError):
    """Raised when vector database operations fail."""
    pass


class QueryError(SawserQGPTError):
    """Raised when query processing fails."""
    pass


class ConfigurationError(SawserQGPTError):
    """Raised when configuration is invalid."""
    pass
