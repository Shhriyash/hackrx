# HackRX API Documentation

This FastAPI application provides a `/hackrx/run` endpoint for processing PDF documents and answering multiple questions using a RAG (Retrieval Augmented Generation) pipeline.

## Features

- **Document Processing**: Uses Gemini Flash 2.5 for OCR and text extraction from PDF URLs
- **Embeddings**: Uses Cohere embeddings for semantic search
- **Vector Storage**: ChromaDB for efficient similarity search
- **Question Answering**: Gemini Pro 2.5 for generating comprehensive answers
- **Batch Processing**: Answers multiple questions in a single API call
- **Authentication**: Bearer token authentication

## Setup

### 1. Environment Variables

Set the required API keys:

```bash
set GOOGLE_API_KEY=your_google_api_key
set COHERE_API_KEY=your_cohere_api_key
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python hackrx_api.py
```

Or use the startup script:

```bash
python start_hackrx_api.py
```

The server will be available at `http://localhost:8000`

## API Endpoints

### POST /hackrx/run

Process a PDF document and answer multiple questions.

**Headers:**
```
Authorization: Bearer a6d040b213c56a698bfba272cfdc432ab9087dc2a669861d58b0d59eab025306
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "Question 1?",
        "Question 2?",
        "Question 3?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "Answer to question 1",
        "Answer to question 2", 
        "Answer to question 3"
    ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "message": "HackRX API is running"
}
```

### GET /

API information and available endpoints.

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer a6d040b213c56a698bfba272cfdc432ab9087dc2a669861d58b0d59eab025306" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": [
      "What is the main topic?",
      "What are the key findings?"
    ]
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer a6d040b213c56a698bfba272cfdc432ab9087dc2a669861d58b0d59eab025306",
    "Content-Type": "application/json"
}

data = {
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic?",
        "What are the key findings?"
    ]
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["answers"])
```

## Testing

### Run the test suite:

```bash
python test_hackrx_api.py
```

### Run the example:

```bash
python example_hackrx_usage.py
```

## Architecture

The API uses the following RAG pipeline:

1. **Document Processing**: Downloads and processes PDF with Gemini Flash 2.5
2. **Chunking**: Splits text into manageable chunks with overlap
3. **Embedding**: Generates embeddings using Cohere's embed-english-v3.0
4. **Indexing**: Stores embeddings in ChromaDB vector store
5. **Retrieval**: For each question, retrieves top 7 relevant chunks
6. **Generation**: Combines all contexts and generates answers using Gemini Pro 2.5

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `401`: Invalid or missing authentication token
- `422`: Invalid request format
- `500`: Internal server error (processing failed)

## Performance Notes

- Document processing time depends on PDF size and complexity
- Typical response time: 30-60 seconds for a standard policy document
- The API processes all questions in a single LLM call for efficiency
- Vector store is in-memory for this demo (can be persisted if needed)

## Limitations

- Maximum document size: Limited by Gemini Flash processing capabilities
- Context window: Limited by Gemini Pro token limits
- Rate limits: Subject to Google AI and Cohere API rate limits

## Security

- Bearer token authentication required
- API keys should be kept secure
- In production, consider implementing additional security measures
