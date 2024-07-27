
# Sawser Q GPT

Sawser Q GPT is a question-answering service built using a Retrieval-Augmented Generation (RAG) architecture. This service allows users to ask questions, which are then enhanced with contextual information using an embedding model and a vector database.

## Technologies Used

- **Python**: The core programming language for the service.
- **Flask**: A lightweight WSGI web application framework for the backend.
- **uWSGI**: A deployment server for serving Flask applications.
- **NGiNX**: A web server that acts as a reverse proxy.
- **llama-index**: Used to build and manage the vector database.
- **BAAI/bge-small-en-v1.5**: The embedding model used to augment questions.
- **TheBloke/Mistral-7B-Instruct-v0.2-GPTQ**: The language model used to answer questions.
- **Docker Compose**: Simplifies the deployment process, for example, using AWS EC2.

## Overview

The service operates by creating an index for the vector database using the llama-index library with the BAAI/bge-small-en-v1.5 embedding model. This index is built using a couple of Circassian-related history articles. When a user asks a question, it is enhanced with contextual information from the vector database before being passed to the LLM model, TheBloke/Mistral-7B-Instruct-v0.2-GPTQ, to generate the answer.

## Deployment

The service can be easily deployed using Docker Compose. Below is an example of how to deploy it on AWS EC2:

1. Clone the repository.
2. Navigate to the project directory.
3. Run `docker compose up --build` to build and start the service.

---

Feel free to adjust any details or add additional information as needed.
