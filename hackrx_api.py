"""
FastAPI application for HackRX document processing and question answering.

This application processes PDF documents from URLs and answers multiple questions
using a RAG pipeline with Gemini Flash and Cohere embeddings.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, HttpUrl
import uvicorn

# from src.document_processing import GeminiFlashProcessor
from src.document_processing import SimpleTextProcessor
from src.embeddings import CohereEmbeddingProvider
from src.vector_store import FAISSVectorStore
from src.llm import GeminiLanguageModel

from dotenv import load_dotenv
load_dotenv(override=True)  # Load environment variables from .env file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hackrx_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create audit directory if it doesn't exist
os.makedirs('audit_logs', exist_ok=True)

# Pydantic models
class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class HackRXResponse(BaseModel):
    answers: List[str]


# FastAPI app
app = FastAPI(
    title="HackRX Document Processing API",
    description="Process documents and answer questions using RAG pipeline",
    version="1.0.0"
)


# Authentication dependency
async def verify_auth_token(authorization: str = Header(...)):
    """Verify the Bearer token."""
    expected_token = "a6d040b213c56a698bfba272cfdc432ab9087dc2a669861d58b0d59eab025306"
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.replace("Bearer ", "")
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return token


class RAGPipeline:
    """Simple RAG pipeline for document processing and question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        # Initialize components
        # self.document_processor = GeminiFlashProcessor(
        #     api_key=os.getenv("GOOGLE_API_KEY"),
        #     chunk_size=800,
        #     chunk_overlap=100
        # )
        self.document_processor = SimpleTextProcessor(chunk_size=800, chunk_overlap=100)


        self.embedding_provider = CohereEmbeddingProvider(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-english-v3.0",
            input_type="search_document"
        )
        
        self.vector_store = FAISSVectorStore(
            dimension=1024,  # Cohere embed-english-v3.0 produces 1024-dimensional embeddings
            persist_directory=None  # In-memory for this demo
        )
        
        self.language_model = GeminiLanguageModel(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="gemini-2.5-flash-lite" 
        )
    
    def process_document(self, document_url: str) -> List[Dict[str, Any]]:
        """Process a document and return chunks."""
        start_time = time.time()
        logger.info(f"Starting document processing for URL: {document_url}")
        
        chunks = self.document_processor.process_document(document_url)
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f} seconds. Created {len(chunks)} chunks")
        
        # Store processing info for audit
        self._last_doc_processing_info = {
            "document_url": document_url,
            "chunk_count": len(chunks),
            "processing_time": processing_time,
            "chunks_preview": [
                {
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "page_number": chunk.get("page_number", "unknown"),
                    "content_length": len(chunk.get("content", "")),
                    "content_preview": chunk.get("content", "")
                }
                for chunk in chunks
            ]
        }
        
        return chunks
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Index document chunks in the vector store."""
        start_time = time.time()
        logger.info(f"Starting indexing of {len(chunks)} chunks")
        
        # Extract text content
        texts = [chunk["content"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Generate embeddings
        embed_start = time.time()
        embeddings = self.embedding_provider.get_embeddings(texts)
        embed_time = time.time() - embed_start
        logger.info(f"Generated embeddings in {embed_time:.2f} seconds")
        
        # Store in vector database
        store_start = time.time()
        self.vector_store.add_documents(texts, embeddings, chunks, ids)
        store_time = time.time() - store_start
        logger.info(f"Stored embeddings in vector store in {store_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total indexing completed in {total_time:.2f} seconds")
    
    def retrieve_context(self, query: str, top_k: int = 7) -> List[str]:
        """Retrieve relevant context for a query."""
        start_time = time.time()
        
        # Generate query embedding
        query_embed_start = time.time()
        query_embedding = self.embedding_provider.get_embeddings([query])[0]
        query_embed_time = time.time() - query_embed_start
        
        # Search vector store
        search_start = time.time()
        results = self.vector_store.search(query_embedding, top_k)
        search_time = time.time() - search_start
        
        # Extract text content
        contexts = []
        for result in results:
            if "content" in result:
                contexts.append(result["content"])
            elif "text" in result:
                contexts.append(result["text"])
        
        total_time = time.time() - start_time
        logger.info(f"Retrieved {len(contexts)} contexts in {total_time:.2f}s (embed: {query_embed_time:.2f}s, search: {search_time:.2f}s)")
        
        return contexts
    
    def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer multiple questions using retrieved context."""
        start_time = time.time()
        logger.info(f"Starting to answer {len(questions)} questions")
        
        # Collect all contexts for all questions
        all_contexts = []
        question_contexts = {}
        
        context_start = time.time()
        for i, question in enumerate(questions):
            contexts = self.retrieve_context(question, top_k=7)
            question_contexts[i] = contexts
            all_contexts.extend(contexts)
        context_time = time.time() - context_start
        
        # Remove duplicates while preserving order
        unique_contexts = []
        seen = set()
        for context in all_contexts:
            if context not in seen:
                unique_contexts.append(context)
                seen.add(context)
        
        logger.info(f"Retrieved contexts in {context_time:.2f} seconds. Unique contexts: {len(unique_contexts)}")
        
        # Limit total context to avoid token limits
        
        # Create prompt for all questions
        context_text = "\n\n".join(unique_contexts)
        
        prompt = f"""
Based on the following document context, answer the questions below. Return your response as a JSON object with a "dummy_key" field containing an array of answers in the exact order of the questions.

CONTEXT:
{context_text}

QUESTIONS:
{json.dumps(questions, indent=2)}

Please provide accurate, specific answers based only on the information in the context. If information is not available in the context, state that clearly.

Return the response in this exact JSON format:
{{
    "dummy_key": [
        "Answer to question 1",
        "Answer to question 2",
        ...
    ]
}}
"""
        
        # Get response from language model
        llm_start = time.time()
        response = self.language_model.generate_response(prompt)
        llm_time = time.time() - llm_start
        logger.info(f"LLM response generated in {llm_time:.2f} seconds")
        
        # Parse JSON response
        parse_start = time.time()
        try:
            # Clean response if it contains markdown code blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            parsed_response = json.loads(cleaned_response.strip())
            
            if "dummy_key" in parsed_response:
                answers = parsed_response["dummy_key"]
            else:
                # Fallback: try to extract answers from response
                answers = [f"Unable to parse answer for question {i+1}" for i in range(len(questions))]
                
        except json.JSONDecodeError:
            # Fallback: return error messages
            answers = [f"Error parsing response for question {i+1}" for i in range(len(questions))]
        
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time
        
        logger.info(f"Question answering completed in {total_time:.2f} seconds (context: {context_time:.2f}s, LLM: {llm_time:.2f}s, parse: {parse_time:.2f}s)")
        
        # Store debugging information for audit
        self._last_qa_debug_info = {
            "unique_contexts": unique_contexts,
            "context_count": len(unique_contexts),
            "total_context_length": len(context_text),
            "prompt": prompt,
            "raw_llm_response": response,
            "cleaned_response": cleaned_response,
            "parsed_successfully": isinstance(answers, list) and len(answers) > 0,
            "timing": {
                "context_retrieval": context_time,
                "llm_generation": llm_time,
                "response_parsing": parse_time,
                "total": total_time
            }
        }
        
        return answers


# Global pipeline instance
pipeline = RAGPipeline()


def save_audit_log(request_data: dict, response_data: dict, timing_info: dict, request_id: str):
    """Save request, response, and timing information for auditing."""
    audit_entry = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
        "timing": timing_info
    }
    
    # Save to file
    audit_filename = f"audit_logs/request_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(audit_filename, 'w', encoding='utf-8') as f:
            json.dump(audit_entry, f, indent=2, ensure_ascii=False)
        logger.info(f"Audit log saved to {audit_filename}")
    except Exception as e:
        logger.error(f"Failed to save audit log: {e}")


@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    token: str = Depends(verify_auth_token)
):
    """
    Process a document and answer questions.
    
    Args:
        request: The request containing document URL and questions
        token: Authentication token (verified by dependency)
    
    Returns:
        Response containing answers to the questions
    """
    # Generate request ID for tracking
    request_id = f"{int(time.time())}_{hash(str(request.documents))}"
    start_time = time.time()
    
    logger.info(f"[{request_id}] Starting request processing")
    
    try:
        # Convert URL to string
        document_url = str(request.documents)
        
        # Process document
        doc_start = time.time()
        logger.info(f"[{request_id}] Processing document: {document_url}")
        chunks = pipeline.process_document(document_url)
        doc_time = time.time() - doc_start
        logger.info(f"[{request_id}] Created {len(chunks)} chunks in {doc_time:.2f} seconds")
        
        # Index chunks
        index_start = time.time()
        logger.info(f"[{request_id}] Indexing chunks...")
        pipeline.index_chunks(chunks)
        index_time = time.time() - index_start
        logger.info(f"[{request_id}] Indexing completed in {index_time:.2f} seconds")
        
        # Answer questions
        qa_start = time.time()
        logger.info(f"[{request_id}] Answering {len(request.questions)} questions...")
        answers = pipeline.answer_questions(request.questions)
        qa_time = time.time() - qa_start
        
        total_time = time.time() - start_time
        
        logger.info(f"[{request_id}] Request completed in {total_time:.2f} seconds")
        logger.info(f"[{request_id}] Timing breakdown - Doc: {doc_time:.2f}s, Index: {index_time:.2f}s, QA: {qa_time:.2f}s")
        
        # Get QA debug info for audit
        qa_debug_info = getattr(pipeline, '_last_qa_debug_info', {})
        doc_debug_info = getattr(pipeline, '_last_doc_processing_info', {})
        
        # Save audit log
        timing_info = {
            'total_time': total_time,
            'document_processing_time': doc_time,
            'indexing_time': index_time,
            'question_answering_time': qa_time,
            'qa_breakdown': qa_debug_info.get('timing', {})
        }
        
        request_data = {
            "document_url": document_url,
            "questions": request.questions,
            "num_questions": len(request.questions)
        }
        
        response_data = {
            "answers": answers,
            "num_answers": len(answers),
            "document_processing_debug": {
                "chunk_count": doc_debug_info.get('chunk_count', 0),
                "processing_time": doc_debug_info.get('processing_time', 0),
                "chunks_preview": doc_debug_info.get('chunks_preview', [])
            },
            "qa_debug": {
                "context_count": qa_debug_info.get('context_count', 0),
                "total_context_length": qa_debug_info.get('total_context_length', 0),
                "parsed_successfully": qa_debug_info.get('parsed_successfully', False),
                "prompt_preview": qa_debug_info.get('prompt', '')[:500] + "..." if qa_debug_info.get('prompt') else "",
                "llm_response_preview": qa_debug_info.get('raw_llm_response', '')[:500] + "..." if qa_debug_info.get('raw_llm_response') else "",
                "retrieved_contexts": [
                    {
                        "index": i,
                        "preview": context[:200] + "..." if len(context) > 200 else context,
                        "length": len(context)
                    }
                    for i, context in enumerate(qa_debug_info.get('unique_contexts', [])[:10])  # Limit to first 10 contexts
                ]
            }
        }
        
        save_audit_log(request_data, response_data, timing_info, request_id)
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"[{request_id}] Error processing request after {error_time:.2f} seconds: {e}")
        
        # Save error audit log
        timing_info = {'error_time': error_time}
        request_data = {
            "document_url": str(request.documents),
            "questions": request.questions,
            "num_questions": len(request.questions)
        }
        error_data = {
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        save_audit_log(request_data, error_data, timing_info, request_id)
        
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "HackRX API is running"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HackRX Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /hackrx/run": "Process document and answer questions",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }


if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["GOOGLE_API_KEY", "COHERE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"  {var}=your_api_key")
        exit(1)
    
    print("Starting HackRX API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
