# pages.py

import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from core.rag import EnhancedRAG
from utils.ollama_utils import check_ollama_status, get_available_models

class Pages:
    @staticmethod
    async def document_processing_page(rag_system: EnhancedRAG):
        st.header("Document Processing")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["txt", "pdf", "docx", "md"],
            accept_multiple_files=True,
            help="Select documents to process"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                ### Supported File Types
                - PDF Documents (.pdf)
                - Word Documents (.docx)
                - Text Files (.txt)
                - Markdown Files (.md)
            """)
        
        with col2:
            st.markdown("""
                ### Processing Features
                - Automatic text extraction
                - Smart chunking
                - Vector embeddings
                - Metadata enrichment
            """)
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                        success = await rag_system.process_document(
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

    @staticmethod
    async def query_interface_page(rag_system: EnhancedRAG):
        st.header("Query Interface")
        
        # Check Ollama status and get available models
        ollama_status = await check_ollama_status()
        if not ollama_status:
            st.error("⚠️ Ollama service is not accessible. Please make sure it's running.")
            st.info("To start Ollama, run: `ollama serve` in your terminal")
            return

        # Get available models
        models = await get_available_models()
        if not models:
            st.warning("No models found. Please pull some models using 'ollama pull model_name'")
            st.info("Example: `ollama pull llama2`")
            return
        
        with st.form("query_form"):
            query = st.text_area(
                "Enter your question",
                help="Ask any question - the system will search both local documents and the web."
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
                    options=models,
                    help="Select the model to generate the answer"
                )
            
            submitted = st.form_submit_button("Search", type="primary")
        
        if submitted and query:
            with st.spinner("Searching and generating response..."):
                try:
                    # Use the optimized query method from RAG
                    response, metrics = await rag_system.optimized_query(
                        query_text=query,
                        n_results=n_results,
                        ollama_model=model,
                        has_documents=await rag_system.db.get_document_count() > 0
                    )
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Search Time", f"{metrics.get('search_time', 0):.3f}s")
                    col2.metric("Embedding Time", f"{metrics.get('embedding_time', 0):.3f}s")
                    col3.metric("LLM Time", f"{metrics.get('llm_time', 0):.3f}s")
                    col4.metric("Total Time", f"{metrics.get('total_time', 0):.3f}s")
                    
                    # Display formatted response
                    st.markdown("### Generated Answer")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.error("Please ensure all services are accessible.")

    @staticmethod
    async def benchmarking_page(rag_system: EnhancedRAG):
        st.header("System Benchmarking")
        
        with st.form("benchmark_form"):
            # Test queries
            queries = st.text_area(
                "Test Queries (one per line)",
                "What is RAG?\nHow do embeddings work?\nExplain vector similarity.",
                help="Enter test queries, one per line"
            ).split('\n')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get available models for selection
                available_models = await get_available_models()
                selected_models = st.multiselect(
                    "Select Models to Test",
                    options=available_models,
                    default=[available_models[0]] if available_models else []
                )
            
            with col2:
                # Test parameters
                n_results = st.number_input(
                    "Results per Query",
                    min_value=1,
                    max_value=10,
                    value=5
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            
            submitted = st.form_submit_button("Run Benchmark")
            
        if submitted:
            if not queries or not selected_models:
                st.warning("Please provide test queries and select models.")
            else:
                with st.spinner("Running benchmark..."):
                    try:
                        results_df = await rag_system.run_benchmark(
                            queries=queries,
                            llm_models=selected_models,
                            n_results=n_results,
                            similarity_threshold=similarity_threshold
                        )
                        
                        if not results_df.empty:
                            Pages.display_benchmark_results(results_df)
                        else:
                            st.warning("No benchmark results generated.")
                            
                    except Exception as e:
                        st.error(f"Benchmark failed: {str(e)}")

    @staticmethod
    def display_benchmark_results(results_df: pd.DataFrame):
        """Display benchmark results."""
        st.subheader("Benchmark Results")
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        st.dataframe(results_df.describe())
        
        # Create visualization
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(
            go.Box(
                name="Total Time",
                y=results_df["total_time"],
                boxpoints="all"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Performance Distribution",
            yaxis_title="Time (seconds)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        with st.expander("View Raw Data"):
            st.dataframe(results_df)
        
        # Export option
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )