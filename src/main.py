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
        model = st.selectbox(
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
    
    # Get available models
    available_models = await get_available_models()
    if not available_models:
        st.warning("No models found. Please pull some models using 'ollama pull model_name'")
        st.info("Example: `ollama pull llama2`")
        return
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        try:
            rag_system = EnhancedRAG(
                hf_token=hf_token,
                embedding_model=model,
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
        uploaded_files = Pages.document_processing_page()
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                        success = await st.session_state.rag_system.process_document(
                            content=file.read(),
                            filename=file.name,
                            file_type=file.name.split('.')[-1].lower(),
                            progress_callback=lambda p: progress_bar.progress(p)
                        )
                        
                        if success:
                            st.success(f"Successfully processed {file.name}")
                        else:
                            st.error(f"Failed to process {file.name}")
                            
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
    
    # Query Interface Tab
    with tab2:
        # Create form for query input
        with st.form("query_form"):
            query = st.text_area(
                "Enter your question",
                help="Enter your question here. The system will search for relevant information."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_results = st.number_input(
                    "Number of results",
                    min_value=1,
                    max_value=10,
                    value=5
                )
            
            with col2:
                model = st.selectbox(
                    "LLM Model",
                    options=available_models,
                    help="Select the model to generate the answer"
                )
            
            submitted = st.form_submit_button("Search", type="primary")
        
        if submitted and query:
            with st.spinner("Searching and generating response..."):
                try:
                    # Get results and generated response
                    results, metrics, response = await st.session_state.rag_system.query(
                        query_text=query,
                        n_results=n_results,
                        ollama_model=model
                    )
                    
                    # Display results
                    Pages.display_results(results, metrics, response)
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.error("Please make sure the Ollama service is running and accessible.")
    
    # Benchmarking Tab
    with tab3:
        benchmark_params = Pages.benchmarking_page()
        
        if benchmark_params["submitted"]:
            if not benchmark_params["queries"] or not benchmark_params["models"]:
                st.warning("Please provide test queries and select models.")
            else:
                with st.spinner("Running benchmark..."):
                    try:
                        results_df = await st.session_state.rag_system.run_benchmark(
                            queries=benchmark_params["queries"],
                            llm_models=benchmark_params["models"],
                            n_results=benchmark_params["n_results"],
                            similarity_threshold=benchmark_params["threshold"]
                        )
                        
                        if not results_df.empty:
                            Pages.display_benchmark_results(results_df)
                        else:
                            st.warning("No benchmark results generated.")
                            
                    except Exception as e:
                        st.error(f"Benchmark failed: {str(e)}")

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