# Complete RAG System

A modular Retrieval Augmented Generation (RAG) system for document analysis and question answering.

## Features

- **Modular Architecture**: Interchangeable components for document processing, embeddings, vector storage, and language models
- **Multiple Document Processors**: 
  - **Docling Processor**: Local PDF processing with page information extraction
  - **Gemini Flash 2.5 Processor**: Cloud-based OCR with table and chart recognition from PDF URLs
- **Multiple Embedding Providers**: Support for Nomic AI and Cohere embeddings
- **Semantic Search**: High-quality embeddings for semantic similarity search
- **Vector Storage**: Efficient similarity search with ChromaDB
- **Language Model Integration**: Response generation with Google Gemini
- **CLI Interface**: Command-line tools for indexing and querying
- **API Ready**: Designed for easy integration into web APIs

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd complete_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

**Option A: Using export commands**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"  # For Gemini Flash processor
export COHERE_API_KEY="your-cohere-api-key-here"  # Optional, only if using Cohere
```

**Option B: Using a .env file (recommended)**
```bash
# Copy the template and edit it
cp .env.template .env
# Edit .env file with your actual API keys
```

Your `.env` file should look like:
```
GEMINI_API_KEY=your_actual_gemini_api_key
GOOGLE_API_KEY=your_actual_google_api_key
COHERE_API_KEY=your_actual_cohere_api_key
```

## Document Processors

This system supports two document processing approaches:

### 1. Docling Processor (Local Processing)
- Processes local PDF files
- Uses the Docling library for PDF parsing
- Best for: Local documents, privacy-sensitive content

### 2. Gemini Flash 2.5 Processor (Cloud OCR)
- Processes PDF documents from URLs
- Uses Google's Gemini Flash 2.5 for advanced OCR
- Includes table extraction and chart descriptions
- Returns page-wise verbatim content in markdown format
- Best for: Public documents, complex layouts with tables/charts

#### Using Gemini Flash Processor

```python
from src.document_processing import GeminiFlashProcessor

# Initialize processor
processor = GeminiFlashProcessor()

# Process a PDF from URL
doc_url = "https://example.com/document.pdf"
chunks = processor.process_document(doc_url)

# Save raw OCR result
raw_result = processor.save_raw_ocr_result(doc_url, "ocr_result.json")
```

#### Via Factory with Configuration

```python
from src.factory import RAGFactory
from config import RAGConfig

# Use Gemini processor configuration
config = RAGConfig.from_file("config/gemini_config.json")
pipeline = RAGFactory.create_pipeline(config)

# Process document URL
result = pipeline.process_and_index_document("https://example.com/document.pdf")
```

## Quick Start

### Basic Usage

```python
from src.factory import RAGFactory
from config import RAGConfig

# Create configuration - API keys loaded automatically from environment
config = RAGConfig.default()

# Create pipeline
pipeline = RAGFactory.create_pipeline(config)

# Index a document
result = pipeline.index_document("path/to/document.pdf")

# Query the system
response = pipeline.query("What is the coverage limit?")
print(response["answer"])
```

### Using Cohere Embeddings

```python
from src.factory import RAGFactory
from config import RAGConfig

# Create configuration with Cohere embeddings
config = RAGConfig.default()
config.embedding.provider = "cohere"
config.embedding.model_name = "embed-english-v3.0"
# API keys loaded automatically from environment variables

# Create pipeline
pipeline = RAGFactory.create_pipeline(config)
# ... rest is the same
```

### CLI Usage

Index a document:
```bash
python examples/cli.py index document.pdf
```

Query in interactive mode:
```bash
python examples/cli.py query --interactive
```

Single query:
```bash
python examples/cli.py query --query "What are the policy terms?"
```

## Architecture

### Components

- **Document Processors**: 
  - **Docling Processor**: Local PDF processing with page information extraction
  - **Gemini Flash Processor**: Cloud-based OCR with advanced table/chart recognition
- **Embedding Providers**: 
  - **Nomic AI**: Local embeddings using nomic-embed-text-v1.5 (default)
  - **Cohere**: API-based embeddings with embed-english-v3.0
- **Vector Store**: Stores and retrieves document embeddings using ChromaDB
- **Language Model**: Generates responses based on retrieved context using Google Gemini
- **RAG Pipeline**: Orchestrates the complete workflow

### Directory Structure

```
complete_rag/
├── src/
│   ├── document_processing/    # PDF processing and chunking
│   ├── embeddings/            # Text embedding generation
│   ├── vector_store/          # Vector storage and retrieval
│   ├── llm/                   # Language model integration
│   ├── rag/                   # Main RAG pipeline
│   ├── interfaces.py          # Abstract base classes
│   └── factory.py             # Component factory
├── config/                    # Configuration management
├── examples/                  # Usage examples and CLI
└── requirements.txt           # Dependencies
```

## Configuration

The system uses configuration classes for easy customization:

```python
from config import RAGConfig, DocumentProcessingConfig

config = RAGConfig.default()
config.document_processing.chunk_size = 1000
config.embedding.provider = "cohere"  # or "nomic"
config.embedding.batch_size = 64
config.vector_store.collection_name = "my_documents"
```

## Testing

### Quick Test for Gemini Flash Processor

To test the new Gemini Flash 2.5 processor:

```bash
# Set your API key
export GOOGLE_API_KEY="your-api-key"

# Run the test script
python test_gemini_processor.py
```

This will:
- Test basic OCR functionality
- Process a sample PDF document
- Generate both chunked and raw OCR results
- Save results to JSON files for inspection

### Running Examples

```bash
# Test Gemini processor with various examples
python examples/gemini_processor_example.py

# Standard CLI usage
python examples/cli.py index document.pdf
python examples/cli.py query --interactive
```

## Extending the System

### Adding New Components

1. Implement the appropriate interface (e.g., `EmbeddingProvider`)
2. Add factory methods in `RAGFactory`
3. Update configuration classes if needed

### Custom Embedding Providers

```python
from src.interfaces import EmbeddingProvider

class OpenAIEmbeddings(EmbeddingProvider):
    def get_embeddings(self, texts):
        # OpenAI API implementation
        pass
```

### Custom Language Models

```python
from src.interfaces import LanguageModel

class CustomLLM(LanguageModel):
    def generate_response(self, prompt: str) -> str:
        # Your implementation here
        pass
```

## API Integration

The system is designed for easy API integration:

```python
from flask import Flask, request, jsonify
from src.factory import RAGFactory

app = Flask(__name__)
pipeline = RAGFactory.create_pipeline(config)

@app.route('/index', methods=['POST'])
def index_document():
    document_url = request.json['document_url']
    # Download and process document
    result = pipeline.index_document(document_path)
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query_documents():
    query = request.json['query']
    result = pipeline.query(query)
    return jsonify(result)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]
