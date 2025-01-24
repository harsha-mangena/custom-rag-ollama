Welcome to the Custom Retrieval-Augmented Generation (RAG) System, a comprehensive solution that combines document retrieval and large language models (LLMs) to deliver accurate, context-aware responses. This project integrates Ollama, a state-of-the-art LLM service, to provide advanced language capabilities.

Features

1. Document Processing & Chunking

Supports various file formats: PDFs, DOCX, Markdown, and plain text.

Processes documents into manageable chunks (default: 512 tokens) with configurable overlap.

2. Embedding Generation

Utilizes embedding models like SentenceTransformer (local) or Hugging Face API.

Supports quantized models for optimized performance.

3. Vector Search

Implements vector similarity search using frameworks like FAISS.

Retrieves the most relevant chunks for queries.

4. Response Generation

Leverages Ollama's LLMs to generate contextual responses based on retrieved chunks.

Configurable prompts for flexible usage.

5. Performance Monitoring

Tracks key metrics such as embedding time, search time, and total query time.

Installation

Clone the Repository:

git clone https://github.com/harsha-mangena/custom-rag-ollama.git
cd custom-rag-ollama

Set Up a Virtual Environment:

python3 -m venv venv
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Configure Environment Variables:

Create a .env file:

HF_API_TOKEN=<your_huggingface_api_token>
DATABASE_PATH=./data/rag.db

Start the Application:

python app.py

