# Custom RAG System with Ollama

A flexible Retrieval-Augmented Generation (RAG) system built with Ollama, designed for efficient document processing and contextual query responses.

## Features

- 📄 Multi-format document support (PDF, DOCX, Markdown, TXT)
- 🔍 Advanced vector similarity search
- 🤖 Integration with Ollama's language models
- 📊 Built-in benchmarking capabilities
- 🎯 High-precision context retrieval

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running
- Required Python packages (specified in `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/custom-rag-ollama.git

# Navigate to project directory
cd custom-rag-ollama

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Document Upload

- Access the Document Upload section in the UI
- Supported file formats:
  - PDF
  - DOCX
  - Markdown
  - Plain text
- Documents are automatically processed and chunked for optimal retrieval

### 2. Query Interface

- Navigate to the Query tab
- Input your question
- Configure parameters:
  - Number of results to retrieve
  - Ollama model selection
- View results with:
  - Generated response
  - Source document metadata
  - Similarity scores

### 3. Benchmarking

Run performance tests to evaluate:
- Embedding generation speed
- Search efficiency
- Model response quality
- Overall system performance

## System Architecture

### Core Components

1. **Document Processing Pipeline**
   - Text extraction from multiple formats
   - Intelligent document chunking
   - Metadata preservation

2. **Embedding Engine**
   - Vector embedding generation
   - Optimization for search performance
   - Model-agnostic design

3. **Vector Search System**
   - High-performance similarity matching
   - Configurable search parameters
   - Result ranking optimization

4. **Response Generation**
   - Integration with Ollama LLMs
   - Context-aware response synthesis
   - Source attribution

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ollama team for their excellent LLM framework
- Contributors and community members

## Contact

For questions and support, please open an issue in the GitHub repository.

---
Made with ❤️ by [Your Name/Organization]
