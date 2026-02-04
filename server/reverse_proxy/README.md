# Docker - Container Configuration

This directory contains Docker configuration files for containerized deployment.

## Structure

```
docker/
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile           # Server Dockerfile (moved to server/)
└── nginx/               # Nginx configuration
    └── nginx.conf       # Nginx reverse proxy config
```

## Usage

```bash
# Build and run with Docker Compose
docker-compose up --build

# Production deployment
docker-compose --profile production up -d
```

## Services

- **sawserq-gpt**: FastAPI server container
- **nginx**: Reverse proxy (production profile)
