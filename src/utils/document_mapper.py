# src/utils/document_mapper.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentMapper:
    """Creates and manages document mind maps."""
    
    def __init__(self):
        """Initialize document mapper."""
        self.graph = nx.Graph()
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            max_df=0.95,
            min_df=2
        )
    
    def create_mind_map(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.5
    ) -> nx.Graph:
        """
        Create mind map between documents based on content similarity.
        
        Args:
            documents: List of document dictionaries
            similarity_threshold: Minimum similarity for connection
            
        Returns:
            NetworkX graph representing document relationships
        """
        try:
            # Clear existing graph
            self.graph.clear()
            
            if not documents:
                self.logger.warning("No documents provided for mind mapping")
                return self.graph
            
            # Extract text content and prepare document mapping
            texts = []
            doc_mapping = {}
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '').strip()
                if content:
                    texts.append(content)
                    doc_mapping[i] = doc
            
            if not texts:
                self.logger.warning("No valid text content found in documents")
                return self.graph
            
            # Calculate TF-IDF matrix
            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
            except Exception as e:
                self.logger.error(f"TF-IDF calculation failed: {e}")
                return self.graph
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Add nodes and edges
            for i in range(len(texts)):
                doc = doc_mapping[i]
                self.graph.add_node(
                    doc['source'],
                    content=doc['content'][:200],  # Store preview
                    type=doc['file_type'],
                    metadata=doc.get('metadata', {})
                )
                
                # Add edges for similar documents
                for j in range(i + 1, len(texts)):
                    if similarities[i, j] >= similarity_threshold:
                        self.graph.add_edge(
                            doc_mapping[i]['source'],
                            doc_mapping[j]['source'],
                            weight=float(similarities[i, j])
                        )
            
            return self.graph
            
        except Exception as e:
            self.logger.error(f"Mind map creation failed: {e}")
            return self.graph
    
    def visualize_mind_map(
        self,
        output_path: str = "mind_map.png",
        figsize: tuple = (12, 8)
    ):
        """Generate mind map visualization."""
        try:
            if not self.graph.nodes():
                self.logger.warning("No nodes in graph to visualize")
                return
            
            plt.figure(figsize=figsize)
            
            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Draw nodes
            node_colors = [
                'lightblue' if self.graph.nodes[node].get('type') in ['txt', 'pdf', 'docx'] 
                else 'lightgreen' for node in self.graph.nodes()
            ]
            
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                node_color=node_colors,
                node_size=1000,
                alpha=0.7
            )
            
            # Draw edges with varying thickness based on weight
            edge_weights = [
                self.graph[u][v]['weight'] * 2
                for u, v in self.graph.edges()
            ]
            
            nx.draw_networkx_edges(
                self.graph,
                pos,
                width=edge_weights,
                alpha=0.5,
                edge_color='gray'
            )
            
            # Add labels
            labels = {
                node: f"{node}\n{data['type']}"
                for node, data in self.graph.nodes(data=True)
            }
            
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels,
                font_size=8
            )
            
            plt.title("Document Mind Map")
            plt.axis('off')
            
            # Save with high DPI
            plt.savefig(
                output_path,
                bbox_inches='tight',
                dpi=300,
                format='png'
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Mind map visualization failed: {e}")
            raise