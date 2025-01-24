import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

class Pages:
    @staticmethod
    def document_processing_page():
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
        
        return uploaded_files

    @staticmethod
    async def query_interface_page():
        st.header("Query Interface")
        
        # Check Ollama status
        ollama_status = await check_ollama_status()
        if not ollama_status:
            st.error("⚠️ Ollama service is not accessible. Please make sure it's running.")
            st.info("To start Ollama, run: `ollama serve` in your terminal")
            return None
        
        # Get available models
        available_models = await get_available_models()
        if not available_models:
            st.warning("No models found. Please pull some models using 'ollama pull model_name'")
            st.info("Example: `ollama pull llama2`")
            return None
        
        # Query input
        query = st.text_area(
            "Enter your question",
            help="Enter your question here. The system will search for relevant information."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_button = st.button("Search", type="primary", disabled=not available_models)
        
        with col2:
            n_results = st.number_input(
                "Number of results",
                min_value=1,
                max_value=10,
                value=5
            )
        
        with col3:
            model = st.selectbox(
                "LLM Model",
                options=available_models,
                help="Select the model to generate the answer",
                disabled=not available_models
            ) if available_models else None
        
        return {
            "query": query,
            "search": search_button,
            "n_results": n_results,
            "model": model,
            "available": bool(available_models)
        }

    @staticmethod
    def benchmarking_page():
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
                # Model selection
                models = ["llama2", "mistral", "codellama"]  # Example models
                selected_models = st.multiselect(
                    "Select Models to Test",
                    models,
                    default=[models[0]]
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
            
        return {
            "queries": queries,
            "models": selected_models if submitted else [],
            "n_results": n_results,
            "threshold": similarity_threshold,
            "submitted": submitted
        }

    @staticmethod
    def display_results(results: List[Dict[str, Any]], metrics: Dict[str, float], response: str):
        """Display search results and metrics."""
        if response:
            st.markdown("### 📝 Generated Answer")
            st.markdown(response)
            st.markdown("---")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Search Time", f"{metrics.get('search_time', 0):.3f}s")
        col2.metric("Embedding Time", f"{metrics.get('embedding_time', 0):.3f}s")
        col3.metric("LLM Time", f"{metrics.get('llm_time', 0):.3f}s")
        col4.metric("Total Time", f"{metrics.get('total_time', 0):.3f}s")
        
        # Display results
        if results:
            st.markdown(f"### 📚 Found {len(results)} relevant passages")
            for idx, result in enumerate(results, 1):
                with st.expander(
                    f"Source {idx}: {result['source']} (Similarity: {result['similarity']:.3f})"
                ):
                    st.markdown("**Content:**")
                    st.markdown(result['content'])
                    
                    if result.get('highlight'):
                        st.markdown("**Relevant Excerpt:**")
                        st.markdown(f"...{result['highlight']}...")
                    
                    st.markdown("**Metadata:**")
                    st.json(result.get('metadata', {}))
        else:
            st.warning("No relevant documents found.")

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