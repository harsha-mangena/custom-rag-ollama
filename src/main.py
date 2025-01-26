# main.py

import streamlit as st
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from config.settings import Settings
from core.rag import EnhancedRAG
from ui.pages import Pages
from ui.components import UIComponents
from utils.ollama_utils import check_ollama_status, get_available_models

async def main():
    """Main application entry point."""
    # Setup
    setup_logging()
    
    # Initialize UI
    st.set_page_config(
        page_title="Enhanced RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        with st.expander("API Settings"):
            hf_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Enter your Hugging Face API token"
            )
        
        # Model Selection
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "bge-small",
                "bge-base",
                "minilm",
                "huggingface"
            ]
        )
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=128,
                max_value=2048,
                value=512,
                step=64
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=256,
                value=50,
                step=10
            )
    
    # Check Ollama status
    ollama_available = await check_ollama_status()
    if not ollama_available:
        st.error("⚠️ Ollama service is not accessible. Please make sure it's running.")
        st.info("To start Ollama, run: `ollama serve` in your terminal")
        return
    
    if 'rag_system' not in st.session_state:
        try:
            rag_system = EnhancedRAG(
                hf_token=hf_token if hf_token else None,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            await rag_system.initialize()
            st.session_state.rag_system = rag_system
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            return
    
    # Main content
    tab1, tab2, tab3 = st.tabs([
        "Document Processing",
        "Query Interface",
        "Benchmarking"
    ])
    
    # Document Processing Tab
    with tab1:
        await Pages.document_processing_page(st.session_state.rag_system)
    
    # Query Interface Tab
    with tab2:
        await Pages.query_interface_page(st.session_state.rag_system)
    
    # Benchmarking Tab
    with tab3:
        await Pages.benchmarking_page(st.session_state.rag_system)

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
            ),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    asyncio.run(main())