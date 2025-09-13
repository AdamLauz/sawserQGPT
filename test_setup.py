#!/usr/bin/env python3
"""Test script to verify the modernized setup."""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.config import settings
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService


async def test_services():
    """Test the modernized services."""
    print("🧪 Testing SawserQ GPT 2.0 Setup")
    print("=" * 50)
    
    # Test configuration
    print(f"✅ Configuration loaded")
    print(f"   - App: {settings.app_name} v{settings.app_version}")
    print(f"   - LLM Model: {settings.llm_model_name}")
    print(f"   - Embedding Model: {settings.embedding_model_name}")
    print(f"   - Device: {settings.device}")
    print(f"   - Debug: {settings.debug}")
    
    # Test LLM service
    print(f"\n🤖 Testing LLM Service...")
    try:
        llm_service = LLMService()
        await llm_service.load_model()
        print(f"✅ LLM service loaded successfully")
        
        # Test generation
        response = await llm_service.generate_response("Hello, how are you?")
        print(f"✅ LLM generation test: {response[:50]}...")
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False
    
    # Test Vector service
    print(f"\n🔍 Testing Vector Service...")
    try:
        vector_service = VectorService()
        await vector_service.initialize()
        print(f"✅ Vector service initialized successfully")
        
        # Test context retrieval
        context, sources, scores = await vector_service.get_context("test query")
        print(f"✅ Vector service test: {len(sources)} sources found")
        
    except Exception as e:
        print(f"❌ Vector service test failed: {e}")
        return False
    
    # Test RAG service
    print(f"\n🧠 Testing RAG Service...")
    try:
        rag_service = RAGService(llm_service, vector_service)
        
        # Test query
        response, context_used, context_info = await rag_service.query("What is this about?")
        print(f"✅ RAG service test: {response[:50]}...")
        print(f"   - Context used: {context_used}")
        print(f"   - Sources: {context_info.total_sources}")
        
    except Exception as e:
        print(f"❌ RAG service test failed: {e}")
        return False
    
    # Test health status
    print(f"\n🏥 Testing Health Status...")
    try:
        health = await rag_service.get_health_status()
        print(f"✅ Health check: {health}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! The modernized setup is working correctly.")
    return True


async def main():
    """Main test function."""
    try:
        success = await test_services()
        if success:
            print(f"\n🚀 Ready to start the server with: python start_server.py")
            print(f"🌐 Ready to start the client with: streamlit run client_streamlit.py")
        else:
            print(f"\n❌ Setup test failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
