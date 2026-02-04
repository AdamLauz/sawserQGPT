#!/usr/bin/env python3
"""Start the FastAPI server that bridges HTTP requests to MCP tools."""

import uvicorn
from server.api.config import api_config

if __name__ == "__main__":
    uvicorn.run(
        "server.api.main:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,
        workers=1 if api_config.debug else api_config.workers
    )
