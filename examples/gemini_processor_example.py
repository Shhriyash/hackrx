"""
Example usage of the Gemini Flash 2.5 document processor.

This example demonstrates how to use the GeminiFlashProcessor for OCR processing
of PDF documents from URLs, including handling of tables and charts.
"""

import os
import json
from src.document_processing import GeminiFlashProcessor
from src.factory import RAGFactory
from config.settings import RAGConfig


def example_direct_processor_usage():
    """Example of using GeminiFlashProcessor directly."""
    print("=== Direct Gemini Flash Processor Usage ===")
    
    # Initialize processor
    processor = GeminiFlashProcessor(
        api_key=os.getenv("GOOGLE_API_KEY"),  # Or pass None to use env var
        chunk_size=800,
        chunk_overlap=100
    )
    
    # Example PDF URL
    doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"
    
    try:
        # Process document
        print(f"Processing document from: {doc_url}")
        chunks = processor.process_document(doc_url)
        
        print(f"Created {len(chunks)} chunks")
        
        # Display first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Has tables: {chunk['has_tables']}")
            print(f"Has charts: {chunk['has_charts']}")
            print(f"Content preview: {chunk['content'][:200]}...")
        
        # Save chunks to file
        processor.save_chunks_to_file(chunks, "gemini_chunks.json")
        print(f"\nSaved chunks to gemini_chunks.json")
        
    except Exception as e:
        print(f"Error processing document: {e}")


def example_raw_ocr_result():
    """Example of getting raw OCR result from Gemini."""
    print("\n=== Raw OCR Result Example ===")
    
    processor = GeminiFlashProcessor()
    doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"
    
    try:
        # Get raw OCR result
        print(f"Getting raw OCR result for: {doc_url}")
        raw_result = processor.save_raw_ocr_result(doc_url, "raw_ocr_result.json")
        
        print(f"Raw result contains {len(raw_result.get('pages', []))} pages")
        
        # Display structure of first page
        if raw_result.get('pages'):
            first_page = raw_result['pages'][0]
            print(f"\nFirst page structure:")
            print(f"- Page number: {first_page.get('page_number')}")
            print(f"- Content length: {len(first_page.get('content', ''))}")
            print(f"- Number of tables: {len(first_page.get('tables', []))}")
            print(f"- Number of charts: {len(first_page.get('charts', []))}")
            
            # Show table info if any
            for i, table in enumerate(first_page.get('tables', [])):
                print(f"  Table {i+1}: {len(table.get('markdown_table', ''))} chars")
            
            # Show chart info if any
            for i, chart in enumerate(first_page.get('charts', [])):
                print(f"  Chart {i+1}: {chart.get('description', '')[:100]}...")
        
        print(f"\nSaved raw result to raw_ocr_result.json")
        
    except Exception as e:
        print(f"Error getting raw OCR result: {e}")


def example_factory_usage():
    """Example of using GeminiFlashProcessor via factory."""
    print("\n=== Factory Usage Example ===")
    
    # Load config for Gemini processor
    config = RAGConfig.from_file("config/gemini_config.json")
    
    try:
        # Create pipeline with Gemini processor
        pipeline = RAGFactory.create_pipeline(config)
        
        print("Created RAG pipeline with Gemini Flash processor")
        print(f"Document processor type: {type(pipeline.document_processor).__name__}")
        
        # Example URL
        doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"
        
        # Process and index document
        print(f"\nProcessing and indexing document: {doc_url}")
        result = pipeline.process_and_index_document(doc_url)
        
        print(f"Indexed {result['chunks_created']} chunks")
        
        # Query the indexed content
        query = "What are the main findings?"
        print(f"\nQuerying: {query}")
        response = pipeline.query(query)
        
        print(f"Response: {response['answer'][:200]}...")
        print(f"Used {len(response['sources'])} source chunks")
        
    except Exception as e:
        print(f"Error with factory usage: {e}")


def example_batch_processing():
    """Example of processing multiple documents."""
    print("\n=== Batch Processing Example ===")
    
    processor = GeminiFlashProcessor()
    
    # Example URLs (replace with actual URLs)
    doc_urls = [
        "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf",
        # Add more URLs as needed
    ]
    
    all_chunks = []
    
    for i, url in enumerate(doc_urls):
        try:
            print(f"Processing document {i+1}/{len(doc_urls)}: {url}")
            chunks = processor.process_document(url)
            
            # Add document identifier to chunks
            for chunk in chunks:
                chunk['document_url'] = url
                chunk['document_id'] = f"doc_{i+1}"
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  Error processing {url}: {e}")
    
    print(f"\nTotal chunks from all documents: {len(all_chunks)}")
    
    # Save all chunks
    processor.save_chunks_to_file(all_chunks, "batch_processed_chunks.json")
    print("Saved all chunks to batch_processed_chunks.json")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY environment variable before running examples")
        print("export GOOGLE_API_KEY='your-api-key'")
        exit(1)
    
    # Run examples
    try:
        example_direct_processor_usage()
        example_raw_ocr_result()
        example_factory_usage()
        example_batch_processing()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
