import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Tuple

class UIComponents:
    """Collection of reusable UI components for the RAG application."""

    @staticmethod
    def setup_page():
        """Configure initial page settings."""
        st.set_page_config(
            page_title="Enhanced RAG System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        st.markdown("""
            <style>
                .main {
                    padding: 2rem;
                }
                .stButton button {
                    width: 100%;
                }
                .status-box {
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
                .metrics-container {
                    display: flex;
                    justify-content: space-between;
                    gap: 1rem;
                }
                .header-box {
                    background-color: #f0f2f6;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    margin-bottom: 2rem;
                }
                .file-uploader {
                    border: 2px dashed #ccc;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    text-align: center;
                }
                .results-container {
                    margin-top: 2rem;
                }
                .info-box {
                    background-color: #e1f5fe;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def header():
        """Display application header."""
        st.markdown(
            """
            <div class='header-box'>
                <h1>🔍 Enhanced RAG System</h1>
                <p>A powerful retrieval-augmented generation system for document analysis and querying.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def file_uploader(accepted_types: List[str]) -> Dict[str, Any]:
        """Create a styled file uploader component."""
        st.markdown("<div class='file-uploader'>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=accepted_types,
            accept_multiple_files=True,
            help="Select one or more files to process"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_files:
            files_info = {
                "count": len(uploaded_files),
                "types": set(f.name.split('.')[-1].lower() for f in uploaded_files),
                "files": uploaded_files
            }
            
            st.markdown(
                f"""
                <div class='info-box'>
                    📁 Selected {files_info['count']} files<br>
                    📊 Types: {', '.join(files_info['types'])}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            return files_info
        return {}

    @staticmethod
    def progress_tracker(total_steps: int):
        """Create a progress tracking component."""
        progress = st.progress(0)
        status = st.empty()
        
        def update(step: int, message: str = ""):
            progress.progress(step / total_steps)
            if message:
                status.markdown(f"🔄 {message}")
        
        def complete(message: str = "✅ Processing complete!"):
            progress.progress(1.0)
            status.markdown(message)
        
        return update, complete

    @staticmethod
    def query_interface():
        """Create the query input interface."""
        st.markdown("<div class='query-section'>", unsafe_allow_html=True)
        
        query = st.text_area(
            "Enter your question",
            placeholder="Type your question here...",
            help="Enter your question to search through the processed documents",
            height=100
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_button = st.button(
                "🔍 Search",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            n_results = st.number_input(
                "Number of results",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col3:
            threshold = st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        return {
            "query": query,
            "search": search_button,
            "n_results": n_results,
            "threshold": threshold
        }

    @staticmethod
    def display_results(results: List[Dict[str, Any]], metrics: Dict[str, float], response: str):
        """Display search results, metrics, and generated response."""
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        
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
        
        st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def benchmark_interface() -> Dict[str, Any]:
        """Create the benchmarking interface."""
        st.markdown("<div class='benchmark-section'>", unsafe_allow_html=True)
        
        with st.form("benchmark_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                queries = st.text_area(
                    "Test Queries (one per line)",
                    value="What is RAG?\nHow do embeddings work?\nExplain vector similarity.",
                    height=150
                ).split('\n')
                
                models = st.multiselect(
                    "Select Models to Test",
                    ["llama2", "mistral", "codellama"],
                    default=["llama2"]
                )
            
            with col2:
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
                
                runs = st.number_input(
                    "Number of Runs",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            submitted = st.form_submit_button("🚀 Run Benchmark")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        return {
            "queries": queries,
            "models": models,
            "n_results": n_results,
            "threshold": similarity_threshold,
            "runs": runs,
            "submitted": submitted
        }

    @staticmethod
    def display_benchmark_results(results_df: pd.DataFrame):
        """Display benchmark results with visualizations."""
        st.markdown("<div class='benchmark-results'>", unsafe_allow_html=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Total Processing Time by Model',
                'Time Components Distribution',
                'Average Similarity Scores',
                'Retrieved Documents Distribution'
            )
        )
        
        # Plot 1: Box plot of total time by model
        fig.add_trace(
            go.Box(
                x=results_df['model_name'],
                y=results_df['total_time'],
                name='Total Time'
            ),
            row=1, col=1
        )
        
        # Plot 2: Time components distribution
        for component in ['embedding_time', 'query_time', 'response_time']:
            fig.add_trace(
                go.Box(
                    y=results_df[component],
                    name=component.replace('_', ' ').title()
                ),
                row=1, col=2
            )
        
        # Plot 3: Similarity scores distribution
        fig.add_trace(
            go.Histogram(
                x=results_df['avg_similarity'],
                nbinsx=20,
                name='Similarity Distribution'
            ),
            row=2, col=1
        )
        
        # Plot 4: Retrieved documents distribution
        doc_counts = results_df['retrieval_count'].value_counts()
        fig.add_trace(
            go.Bar(
                x=doc_counts.index,
                y=doc_counts.values,
                name='Document Count Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(results_df.describe())
        
        st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def configuration_sidebar() -> Dict[str, Any]:
        """Create the configuration sidebar."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Configuration
            with st.expander("API Settings"):
                hf_token = st.text_input(
                    "Hugging Face API Token",
                    type="password",
                    help="Enter your Hugging Face API token for enhanced embeddings"
                )
            
            # Model Selection
            model = st.selectbox(
                "Embedding Model",
                options=[
                    "bge-small",
                    "bge-base",
                    "minilm",
                    "huggingface"
                ],
                help="Select the embedding model to use"
            )
            
            # Advanced Settings
            with st.expander("Advanced Settings"):
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=128,
                    max_value=2048,
                    value=512,
                    step=64,
                    help="Size of text chunks for processing"
                )
                
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=256,
                    value=50,
                    step=10,
                    help="Overlap between consecutive chunks"
                )
            
            # System Info
            with st.expander("System Info"):
                st.markdown("""
                    - Version: 1.0.0
                    - Status: Active
                    - [Documentation](https://github.com/yourusername/rag-system)
                    - [Report Issues](https://github.com/yourusername/rag-system/issues)
                """)
        
        return {
            "hf_token": hf_token,
            "model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }