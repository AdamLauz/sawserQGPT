"""Modern Streamlit client for SawserQ GPT FastAPI backend."""

import asyncio
import json
import streamlit as st
import httpx
from typing import AsyncGenerator

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update with your server URL
API_ENDPOINTS = {
    "query": f"{API_BASE_URL}/api/v1/query",
    "query_stream": f"{API_BASE_URL}/api/v1/query/stream",
    "health": f"{API_BASE_URL}/api/v1/health"
}

# Page configuration
st.set_page_config(
    page_title="SawserQ GPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy {
        background-color: #4caf50;
    }
    .status-unhealthy {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)


class SawserQGPTClient:
    """Client for interacting with SawserQ GPT API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.endpoints = {
            "query": f"{base_url}/api/v1/query",
            "query_stream": f"{base_url}/api/v1/query/stream",
            "health": f"{base_url}/api/v1/health"
        }
    
    async def check_health(self) -> dict:
        """Check API health status."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.endpoints["health"])
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    async def query(self, query: str, max_tokens: int = None, temperature: float = None) -> dict:
        """Send a query to the API."""
        try:
            payload = {"query": query}
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature:
                payload["temperature"] = temperature
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoints["query"],
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def query_stream(self, query: str, max_tokens: int = None, temperature: float = None) -> AsyncGenerator[str, None]:
        """Stream a query to the API."""
        try:
            payload = {"query": query}
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature:
                payload["temperature"] = temperature
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    self.endpoints["query_stream"],
                    json=payload
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"


def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>SawserQ GPT:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)


def display_health_status(health_data: dict):
    """Display health status in sidebar."""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        if "error" in health_data:
            st.error(f"❌ Connection Error: {health_data['error']}")
            return
        
        status = health_data.get("status", "unknown")
        model_loaded = health_data.get("model_loaded", False)
        vector_ready = health_data.get("vector_db_ready", False)
        
        # Overall status
        if status == "healthy":
            st.success("✅ System Healthy")
        else:
            st.error("❌ System Unhealthy")
        
        # Model status
        if model_loaded:
            st.success("✅ LLM Model Loaded")
        else:
            st.warning("⚠️ LLM Model Not Loaded")
        
        # Vector DB status
        if vector_ready:
            st.success("✅ Vector DB Ready")
        else:
            st.warning("⚠️ Vector DB Not Ready")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🤖 SawserQ GPT</h1>', unsafe_allow_html=True)
    st.markdown("**A modern RAG-powered chatbot for Circassian history and culture**")
    
    # Initialize client
    client = SawserQGPTClient()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # API URL configuration
        api_url = st.text_input(
            "API URL", 
            value=API_BASE_URL,
            help="URL of the SawserQ GPT API server"
        )
        
        if api_url != API_BASE_URL:
            client = SawserQGPTClient(api_url)
        
        # Generation parameters
        st.markdown("#### Generation Parameters")
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=50, 
            max_value=1000, 
            value=512,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.7,
            step=0.1,
            help="Controls randomness in generation"
        )
        
        # Health check button
        if st.button("🔍 Check System Health"):
            with st.spinner("Checking system health..."):
                health_data = asyncio.run(client.check_health())
                display_health_status(health_data)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about Circassian history and culture..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use streaming for better UX
                response_container = st.empty()
                full_response = ""
                
                try:
                    async def generate_response():
                        nonlocal full_response
                        async for chunk in client.query_stream(
                            prompt, 
                            max_tokens=max_tokens,
                            temperature=temperature
                        ):
                            full_response += chunk
                            response_container.markdown(full_response)
                    
                    asyncio.run(generate_response())
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using FastAPI, Streamlit, and modern RAG techniques | "
        f"API: {client.base_url}"
    )


if __name__ == "__main__":
    main()