"""
FastAPI application for HackRX document processing and question answering.

This application processes PDF documents from URLs and answers multiple questions
using a RAG pipeline with Gemini Flash and dual embedding providers (Cohere/Google).
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, HttpUrl

# Import security middleware
import sys
sys.path.append('g:/dem/hackrx')
from security_middleware import security_middleware, record_auth_failure, get_security_stats
import uvicorn

# from src.document_processing import GeminiFlashProcessor
from src.document_processing import SimpleTextProcessor
from src.embeddings import get_embedding_provider
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
    description="Process documents and answer questions using RAG pipeline with security features",
    version="1.0.0"
)

# Add security middleware
app.middleware("http")(security_middleware)


# Authentication dependency
async def verify_auth_token(request: Request, authorization: str = Header(...)):
    """Verify the Bearer token."""
    expected_token = "a6d040b213c56a698bfba272cfdc432ab9087dc2a669861d58b0d59eab025306"
    
    # Get client IP for logging
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    
    if not authorization.startswith("Bearer "):
        record_auth_failure(client_ip)
        logger.warning(f"Invalid auth header format from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.replace("Bearer ", "")
    if token != expected_token:
        record_auth_failure(client_ip)
        logger.warning(f" Invalid token from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    logger.info(f" Successful authentication from {client_ip}")
    return token


class RAGPipeline:
    """Simple RAG pipeline for document processing and question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        # Initialize components
        # Use optimized document processor for better performance
        # self.document_processor = GeminiFlashProcessor(
        #     api_key=os.getenv("GOOGLE_API_KEY"),
        #     chunk_size=800,
        #     chunk_overlap=100
        # )
        # OPTIMIZATION: Balanced chunk size for optimal performance and context
        self.document_processor = SimpleTextProcessor(
            chunk_size=2048,  
            chunk_overlap=204 
        )

        # Manual provider selection - change "cohere" to "google" as needed
        provider_type = os.getenv("EMBEDDING_PROVIDER", "google") 
        self.embedding_provider = get_embedding_provider(provider_type)
        
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_provider.dimension,  # Dynamic dimension based on provider
            persist_directory=None  # In-memory for this demo
        )
        
        self.language_model = GeminiLanguageModel(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="gemini-2.5-pro" 
        )
        
        # Configuration for parallel processing
        self.enable_parallel_processing = os.getenv("ENABLE_PARALLEL_QA", "true").lower() == "true"
        self.max_context_workers = int(os.getenv("MAX_CONTEXT_WORKERS", "10"))
        self.max_llm_workers = int(os.getenv("MAX_LLM_WORKERS", "8"))
        
        logger.info(f"Parallel processing: {self.enable_parallel_processing}")
        if self.enable_parallel_processing:
            logger.info(f"Max workers - Context: {self.max_context_workers}, LLM: {self.max_llm_workers}")
        
        # For neighbor chunk lookup
        self.chunk_lookup = {}  # {sequence_index: chunk_data}
        self.total_chunks = 0   # Total number of chunks for boundary checks
    
    def process_document(self, document_url: str) -> List[Dict[str, Any]]:
        """Process a document and return chunks."""
        start_time = time.time()
        logger.info(f"Starting document processing for URL: {document_url}")
        
        chunks = self.document_processor.process_document(document_url)
        
        # Add sequence_index to each chunk for neighbor retrieval
        for i, chunk in enumerate(chunks):
            chunk["sequence_index"] = i
        
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
                    "sequence_index": chunk.get("sequence_index", "unknown"),
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
        
        # Store total chunks and create lookup for neighbor retrieval
        self.total_chunks = len(chunks)
        self.chunk_lookup = {chunk["sequence_index"]: chunk for chunk in chunks}
        
        # Extract text content
        texts = [chunk["content"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Generate embeddings with parallel processing
        embed_start = time.time()
        logger.info(f"Starting parallel embedding generation for {len(texts)} chunks using {self.embedding_provider.provider_name}")
        embeddings = self.embedding_provider.get_embeddings(texts)
        embed_time = time.time() - embed_start
        
        processing_rate = len(texts) / embed_time if embed_time > 0 else 0
        logger.info(f"Parallel embedding generation completed in {embed_time:.2f}s ({processing_rate:.1f} chunks/sec)")
        
        # Store in vector database
        store_start = time.time()
        self.vector_store.add_documents(texts, embeddings, chunks, ids)
        store_time = time.time() - store_start
        logger.info(f"Stored embeddings in vector store in {store_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total indexing completed in {total_time:.2f} seconds")
        logger.info(f"Created chunk lookup for {self.total_chunks} chunks with neighbor retrieval capability")
    
    def retrieve_context(self, query: str, top_k: int = 7) -> List[str]:
        """Retrieve relevant context for a query with neighboring chunks."""
        start_time = time.time()
        
        # Generate query embedding using optimized method
        query_embed_start = time.time()
        # Use get_query_embedding for RETRIEVAL_QUERY task type optimization
        if hasattr(self.embedding_provider, 'get_query_embedding'):
            query_embedding = self.embedding_provider.get_query_embedding(query)
        else:
            # Fallback: use get_embeddings with RETRIEVAL_QUERY task type
            if hasattr(self.embedding_provider, 'get_embeddings'):
                # Check if get_embeddings supports task_type parameter
                try:
                    query_embedding = self.embedding_provider.get_embeddings([query], task_type="RETRIEVAL_QUERY")[0]
                except TypeError:
                    # Fallback for providers without task_type parameter
                    query_embedding = self.embedding_provider.get_embeddings([query])[0]
            else:
                query_embedding = self.embedding_provider.get_embeddings([query])[0]
        query_embed_time = time.time() - query_embed_start
        
        # Search vector store for top-k chunks
        search_start = time.time()
        results = self.vector_store.search(query_embedding, top_k)
        search_time = time.time() - search_start
        
        # Expand with neighboring chunks
        expand_start = time.time()
        expanded_indices = set()
        
        for result in results:
            seq_idx = int(result["metadata"]["chunk_id"][6:])
            
            # Add current chunk
            expanded_indices.add(seq_idx)
            
            # Add previous chunk (seq_idx - 1) if exists
            if seq_idx - 1 >= 0:
                expanded_indices.add(seq_idx - 1)
                
            # Add next chunk (seq_idx + 1) if exists  
            if seq_idx + 1 < self.total_chunks:
                expanded_indices.add(seq_idx + 1)
        
        # Sort by document order (sequence_index)
        sorted_indices = sorted(expanded_indices)
        
        # Get content in document order
        contexts = []
        for idx in sorted_indices:
            if idx in self.chunk_lookup:
                contexts.append(self.chunk_lookup[idx]["content"])
        
        expand_time = time.time() - expand_start
        total_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(contexts)} expanded contexts (from {len(results)} top-k) in {total_time:.2f}s")
        logger.info(f"Timing breakdown - embed: {query_embed_time:.2f}s, search: {search_time:.2f}s, expand: {expand_time:.2f}s")
        logger.info(f"Expanded from indices {[r.get('sequence_index', 'unknown') for r in results]} to {sorted_indices}")
        
        return contexts
    
    def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer multiple questions using parallel processing with sequential fallback."""
        start_time = time.time()
        logger.info(f"Starting to answer {len(questions)} questions")
        
        # Use parallel processing if enabled and multiple questions, otherwise use sequential
        if self.enable_parallel_processing and len(questions) > 1:
            try:
                return self._answer_questions_parallel(questions, start_time)
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
                return self._answer_questions_sequential(questions, start_time)
        else:
            logger.info(f"Using sequential processing (parallel disabled or single question)")
            return self._answer_questions_sequential(questions, start_time)
    
    def _answer_questions_parallel(self, questions: List[str], start_time: float) -> List[str]:
        """
        PARALLEL PROCESSING: Answer questions concurrently for maximum speed.
        Optimized for 10-20 questions with rate limiting considerations.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Using PARALLEL processing for {len(questions)} questions")
        
        # STEP 1: Batch generate all query embeddings (already optimized)
        embed_start = time.time()
        if hasattr(self.embedding_provider, 'get_query_embeddings'):
            query_embeddings = self.embedding_provider.get_query_embeddings(questions)
        elif hasattr(self.embedding_provider, 'get_query_embedding'):
            query_embeddings = []
            for question in questions:
                query_embeddings.append(self.embedding_provider.get_query_embedding(question))
        else:
            query_embeddings = self.embedding_provider.get_embeddings(questions)
        embed_time = time.time() - embed_start
        logger.info(f"Generated {len(query_embeddings)} query embeddings in {embed_time:.2f}s (parallel)")
        
        # STEP 2: Parallel context retrieval for all questions
        context_start = time.time()
        question_contexts = {}
        
        def retrieve_context_for_question(question_idx, query_embedding):
            """Retrieve context for a single question."""
            results = self.vector_store.search(query_embedding, top_k=7)
            
            # Expand with neighboring chunks
            expanded_indices = set()
            for result in results:
                seq_idx = int(result["metadata"]["chunk_id"][6:])
                expanded_indices.add(seq_idx)
                if seq_idx - 1 >= 0:
                    expanded_indices.add(seq_idx - 1)
                if seq_idx + 1 < self.total_chunks:
                    expanded_indices.add(seq_idx + 1)
            
            # Get contexts in document order
            sorted_indices = sorted(expanded_indices)
            contexts = []
            for idx in sorted_indices:
                if idx in self.chunk_lookup:
                    contexts.append(self.chunk_lookup[idx]["content"])
            
            return question_idx, contexts
        
        # Execute context retrieval in parallel (configurable max workers)
        max_workers = min(len(questions), self.max_context_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_question = {
                executor.submit(retrieve_context_for_question, i, emb): i 
                for i, emb in enumerate(query_embeddings)
            }
            
            for future in as_completed(future_to_question):
                question_idx, contexts = future.result()
                question_contexts[question_idx] = contexts
        
        context_time = time.time() - context_start
        logger.info(f"Parallel context retrieval completed in {context_time:.2f}s")
        
        # STEP 3: Parallel LLM calls for individual questions
        llm_start = time.time()
        
        def answer_single_question(question_idx, question, contexts):
            """Generate answer for a single question with its context."""
            context_text = "\n\n".join(contexts)
            
            prompt = f"""Based on the following document context, answer the question below.

CONTEXT:
{context_text}

QUESTION: {question}

Instructions:
1. Provide accurate, specific answers based only on the information in the context
2. Include relevant excerpts or quotes from the context that support your answer
3. Reference specific parts by including phrases like "According to the document..." when applicable
4. If information is not available in the context, state that clearly

Answer:"""
            
            response = self.language_model.generate_response(prompt)
            return question_idx, response.strip()
        
        # Execute LLM calls in parallel (configurable max workers to respect rate limits)
        max_llm_workers = min(len(questions), self.max_llm_workers)
        answers_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_llm_workers) as executor:
            future_to_question = {
                executor.submit(answer_single_question, i, q, question_contexts[i]): i 
                for i, q in enumerate(questions)
            }
            
            for future in as_completed(future_to_question):
                question_idx, answer = future.result()
                answers_dict[question_idx] = answer
        
        llm_time = time.time() - llm_start
        
        # Ensure answers are in correct order
        answers = [answers_dict[i] for i in range(len(questions))]
        
        total_time = time.time() - start_time
        logger.info(f"PARALLEL processing completed in {total_time:.2f}s (embed: {embed_time:.2f}s, context: {context_time:.2f}s, LLM: {llm_time:.2f}s)")
        logger.info(f"Speed improvement: ~{max_llm_workers}x for LLM calls, ~{max_workers}x for context retrieval")
        
        # Store debugging information for audit
        all_contexts = []
        for contexts in question_contexts.values():
            all_contexts.extend(contexts)
        
        unique_contexts = list(dict.fromkeys(all_contexts))  # Remove duplicates, preserve order
        
        self._last_qa_debug_info = {
            "processing_mode": "parallel",
            "parallel_workers": {"context": max_workers, "llm": max_llm_workers},
            "unique_contexts": unique_contexts,
            "context_count": len(unique_contexts),
            "total_context_length": sum(len(ctx) for ctx in unique_contexts),
            "individual_question_contexts": question_contexts,
            "timing": {
                "embed_generation": embed_time,
                "context_retrieval": context_time,
                "llm_generation": llm_time,
                "total": total_time
            }
        }
        
        return answers
    
    def _answer_questions_sequential(self, questions: List[str], start_time: float) -> List[str]:
        """
        SEQUENTIAL PROCESSING: Original implementation as fallback.
        Reliable but slower for multiple questions.
        """
        logger.info(f"Using SEQUENTIAL processing for {len(questions)} questions")
        
        # OPTIMIZATION 1: Batch generate all query embeddings at once with parallel processing
        embed_start = time.time()
        if hasattr(self.embedding_provider, 'get_query_embeddings'):
            # Use parallel query embedding processing (MAJOR OPTIMIZATION)
            query_embeddings = self.embedding_provider.get_query_embeddings(questions)
        elif hasattr(self.embedding_provider, 'get_query_embedding'):
            # Fallback: Use single query method in loop (slower)
            query_embeddings = []
            for question in questions:
                query_embeddings.append(self.embedding_provider.get_query_embedding(question))
        else:
            # Fallback: Batch process all questions with RETRIEVAL_DOCUMENT task type
            query_embeddings = self.embedding_provider.get_embeddings(questions)
        embed_time = time.time() - embed_start
        logger.info(f"Generated {len(query_embeddings)} query embeddings in {embed_time:.2f} seconds (parallel processing)")
        
        # OPTIMIZATION 2: Retrieve contexts for all questions efficiently
        context_start = time.time()
        all_contexts = []
        question_contexts = {}
        
        for i, (question, query_embedding) in enumerate(zip(questions, query_embeddings)):
            # Use pre-computed embedding for vector search
            search_start = time.time()
            results = self.vector_store.search(query_embedding, top_k=7)
            
            # Expand with neighboring chunks (same logic as before)
            expanded_indices = set()
            for result in results:
                seq_idx = int(result["metadata"]["chunk_id"][6:])
                expanded_indices.add(seq_idx)
                if seq_idx - 1 >= 0:
                    expanded_indices.add(seq_idx - 1)
                if seq_idx + 1 < self.total_chunks:
                    expanded_indices.add(seq_idx + 1)
            
            # Get contexts in document order
            sorted_indices = sorted(expanded_indices)
            contexts = []
            for idx in sorted_indices:
                if idx in self.chunk_lookup:
                    contexts.append(self.chunk_lookup[idx]["content"])
            
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

Instructions:
1. Provide accurate, specific answers based only on the information in the context
2. Try to include relevant excerpts or quotes from the context that support your answer whenever possible
3. Reference specific parts of the context by including phrases like "According to the document..." or "As stated in the context..." when applicable
4. If information is not available in the context, state that clearly
5. Make your answers comprehensive by including the supporting context/evidence within the answer itself whenever possible

Return the response in this exact JSON format:
{{
    "dummy_key": [
        "Answer to question 1 with supporting context quotes",
        "Answer to question 2 with supporting context quotes",
        ...
    ]
}}

Example answer format: "According to the document, [specific quote from context]. This means that [your interpretation/answer]. The context also mentions that [additional supporting quote if relevant]."
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
        
        logger.info(f"SEQUENTIAL processing completed in {total_time:.2f} seconds (context: {context_time:.2f}s, LLM: {llm_time:.2f}s, parse: {parse_time:.2f}s)")
        
        # Store debugging information for audit
        self._last_qa_debug_info = {
            "processing_mode": "sequential",
            "unique_contexts": unique_contexts,
            "context_count": len(unique_contexts),
            "total_context_length": len(context_text),
            "prompt": prompt,
            "raw_llm_response": response,
            "cleaned_response": cleaned_response,
            "parsed_successfully": isinstance(answers, list) and len(answers) > 0,
            "neighbor_expansion_enabled": True,
            "timing": {
                "embed_generation": embed_time,
                "context_retrieval": context_time,
                "llm_generation": llm_time,
                "response_parsing": parse_time,
                "total": total_time
            }
        }
        
        return answers
        
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
            "neighbor_expansion_enabled": True,
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
        
    except ValueError as e:
        error_time = time.time() - start_time
        
        # Check if it's an unknown file type error
        if "unknown file type" in str(e).lower():
            logger.warning(f"[{request_id}] Unknown file type rejection after {error_time:.2f} seconds: {e}")
            
            # Save audit log for file type rejection
            timing_info = {'error_time': error_time}
            request_data = {
                "document_url": str(request.documents),
                "questions": request.questions,
                "num_questions": len(request.questions)
            }
            error_data = {
                "error": "unknown file type",
                "error_type": "FileTypeError"
            }
            
            save_audit_log(request_data, error_data, timing_info, request_id)
            
            raise HTTPException(status_code=400, detail="unknown file type")
        else:
            # Other ValueError - treat as server error
            logger.error(f"[{request_id}] ValueError after {error_time:.2f} seconds: {e}")
            
            timing_info = {'error_time': error_time}
            request_data = {
                "document_url": str(request.documents),
                "questions": request.questions,
                "num_questions": len(request.questions)
            }
            error_data = {
                "error": str(e),
                "error_type": "ValueError"
            }
            
            save_audit_log(request_data, error_data, timing_info, request_id)
            
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
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
        "description": "Process documents and answer questions using RAG pipeline with security features",
        "supported_file_types": {
            "pdf": "PDF documents (local files and URLs)",
            "docx": "Microsoft Word documents (local files and URLs)", 
            "txt": "Plain text files (local files and URLs)",
            "md": "Markdown files (local files and URLs)"
        },
        "features": {
            "no_size_limits": "Process large readable files without size barriers",
            "automatic_encoding": "Automatic text encoding detection for text files",
            "file_filtering": "Automatic rejection of any file type not in supported list",
            "fast_validation": "Quick file type validation for faster response times",
            "dual_embeddings": "Support for Google and Cohere embeddings",
            "security": "Rate limiting, authentication, and request monitoring"
        },
        "endpoints": {
            "POST /hackrx/run": "Process document and answer questions",
            "GET /health": "Health check",
            "GET /auth/info": "Authentication information",
            "POST /auth/validate": "Validate authentication token",
            "GET /security/status": "Security monitoring status",
            "GET /": "API information"
        }
    }


@app.get("/security/status")
async def get_security_status(token: str = Depends(verify_auth_token)):
    """Get security and monitoring statistics."""
    return {
        "status": "Security monitoring active",
        "timestamp": datetime.now().isoformat(),
        "statistics": get_security_stats(),
        "features": {
            "rate_limiting": "✅ 300 req/5min (60/min sustained) + 50 burst/min",
            "authentication": "✅ Bearer token required",
            "ip_monitoring": "✅ Suspicious IP detection & blocking",
            "request_logging": "✅ All requests logged with IP tracking",
            "security_headers": "✅ XSS, CSRF, content-type protection",
            "ssl_support": "✅ HTTPS/TLS ready (cert required)",
            "production_ready": "✅ High-throughput optimized"
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
    
    # Import production configuration
    import sys
    sys.path.append('g:/dem/hackrx')
    from production_config import run_production_server
    
    print("Starting HackRX API server...")
    run_production_server(app)