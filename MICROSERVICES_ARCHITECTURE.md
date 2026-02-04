# Microservices Architecture

This document describes the microservices architecture implementation for SawserQ GPT.

## Architecture Overview

The system has been refactored from a monolithic architecture to a microservices architecture with the following services:

### Services

1. **API Gateway** (`server/api/`) - Port 8000
   - FastAPI-based API gateway
   - Handles client requests
   - Routes requests to appropriate services
   - Implements rate limiting and CORS

2. **LLM Service** (`server/services/llm/`) - Port 8004
   - Dedicated service for language model operations
   - Text generation and embeddings
   - Model loading and management
   - GPU/CPU optimization

3. **Knowledge Graph Service** (`server/services/knowledge_graph/`) - Port 8001
   - Entity extraction and relation detection
   - Knowledge graph management
   - Graph querying and reasoning
   - NetworkX-based graph operations

4. **Document Processor Service** (`server/services/document_processor/`) - Port 8002
   - Document processing and text extraction
   - PDF processing with PyMuPDF
   - Entity extraction from documents
   - Batch processing capabilities

5. **RAG Orchestrator Service** (`server/services/rag_orchestrator/`) - Port 8003
   - Orchestrates RAG operations across services
   - Coordinates between LLM, Knowledge Graph, and Document Processor
   - Implements the complete RAG pipeline
   - Service communication and error handling

6. **Client** (`client/`) - Port 8501
   - Streamlit-based user interface
   - Communicates with API Gateway
   - Real-time chat interface

7. **Reverse Proxy** (`server/reverse_proxy/`) - Port 80
   - Nginx-based reverse proxy
   - Production deployment
   - Load balancing and SSL termination

## Service Communication

```
Client (Streamlit) 
    ↓ HTTP
API Gateway (FastAPI)
    ↓ HTTP
RAG Orchestrator
    ↓ HTTP
┌─────────────────┬─────────────────┬─────────────────┐
│   LLM Service   │ Knowledge Graph │ Document Proc.  │
│   (Port 8004)   │   (Port 8001)   │   (Port 8002)   │
└─────────────────┴─────────────────┴─────────────────┘
```

## Benefits of Microservices Architecture

### 1. **Scalability**
- Each service can be scaled independently
- LLM service can be scaled based on GPU availability
- Knowledge Graph service can be scaled based on query load

### 2. **Maintainability**
- Clear separation of concerns
- Independent development and deployment
- Easier testing and debugging

### 3. **Technology Flexibility**
- Each service can use optimal technology stack
- Independent dependency management
- Service-specific optimizations

### 4. **Fault Isolation**
- Service failures don't affect entire system
- Graceful degradation
- Independent health monitoring

### 5. **Development Efficiency**
- Parallel development by different teams
- Independent versioning
- Faster deployment cycles

## Service Dependencies

### API Gateway Dependencies
- RAG Orchestrator Service
- All other services (indirectly)

### RAG Orchestrator Dependencies
- LLM Service
- Knowledge Graph Service
- Document Processor Service

### Service Startup Order
1. Knowledge Graph Service
2. Document Processor Service
3. LLM Service
4. RAG Orchestrator Service
5. API Gateway
6. Client (optional)
7. Reverse Proxy (production only)

## Configuration

Each service has its own configuration file:
- `server/api/config.py` - API Gateway configuration
- `server/services/llm/config.py` - LLM Service configuration
- `server/services/knowledge_graph/config.py` - Knowledge Graph configuration
- `server/services/document_processor/config.py` - Document Processor configuration
- `server/services/rag_orchestrator/config.py` - RAG Orchestrator configuration

## Docker Compose

The system uses Docker Compose for orchestration:

```bash
# Start all services
docker-compose up -d

# Start with client
docker-compose --profile client up -d

# Start production setup
docker-compose --profile production up -d

# Start specific services
docker-compose up -d api-gateway llm-service
```

## Service Health Monitoring

Each service provides health check endpoints:
- `/health` - Basic health status
- Service-specific health endpoints
- Docker health checks
- Comprehensive monitoring

## Development Workflow

### 1. **Local Development**
```bash
# Start individual services
cd server/services/llm
python main.py

cd server/services/knowledge_graph
python main.py

# Or use Docker Compose
docker-compose up -d
```

### 2. **Testing**
```bash
# Test individual services
curl http://localhost:8004/health  # LLM Service
curl http://localhost:8001/health  # Knowledge Graph
curl http://localhost:8002/health  # Document Processor
curl http://localhost:8003/health  # RAG Orchestrator
curl http://localhost:8000/api/v1/health  # API Gateway
```

### 3. **Deployment**
```bash
# Production deployment
docker-compose --profile production up -d
```

## Service Communication Patterns

### 1. **Synchronous HTTP**
- Direct service-to-service communication
- Request-response pattern
- Error handling and retries

### 2. **Service Discovery**
- Environment variables for service URLs
- Docker Compose networking
- Health check integration

### 3. **Error Handling**
- Graceful degradation
- Circuit breaker pattern
- Fallback responses

## Monitoring and Observability

### 1. **Health Checks**
- Service-level health endpoints
- Docker health checks
- Comprehensive status reporting

### 2. **Logging**
- Structured logging per service
- Centralized log aggregation
- Error tracking and debugging

### 3. **Metrics**
- Service performance metrics
- Resource utilization
- Request/response times

## Security Considerations

### 1. **Service Isolation**
- Container-based isolation
- Network segmentation
- Resource limits

### 2. **Authentication**
- Service-to-service authentication
- API key management
- Rate limiting

### 3. **Data Protection**
- Encrypted communication
- Secure configuration
- Access control

## Future Enhancements

### 1. **Service Mesh**
- Istio or Linkerd integration
- Advanced traffic management
- Security policies

### 2. **Message Queues**
- Asynchronous communication
- Event-driven architecture
- Better scalability

### 3. **API Gateway Features**
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation

### 4. **Monitoring Stack**
- Prometheus metrics
- Grafana dashboards
- Alerting and notifications

## Migration from Monolithic Architecture

The migration from monolithic to microservices involved:

1. **Service Extraction**
   - Extracted LLM functionality to dedicated service
   - Separated Knowledge Graph operations
   - Isolated Document Processing

2. **API Gateway Implementation**
   - Centralized request handling
   - Service routing
   - Cross-cutting concerns

3. **Service Communication**
   - HTTP-based communication
   - Service discovery
   - Error handling

4. **Configuration Management**
   - Service-specific configuration
   - Environment-based settings
   - Docker Compose orchestration

This microservices architecture provides a solid foundation for scalable, maintainable, and robust AI-powered applications.
