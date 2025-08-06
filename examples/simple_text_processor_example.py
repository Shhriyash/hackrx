"""
Example script demonstrating the SimpleTextProcessor for direct text extraction.

This example shows how to use the SimpleTextProcessor to extract text
from PDF and DOCX files without OCR, supporting both local files and URLs.
"""

import logging
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing.processor import SimpleTextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Demonstrate SimpleTextProcessor usage with different document types.
    """
    print("=== SimpleTextProcessor Example ===\n")
    
    # Initialize the processor
    processor = SimpleTextProcessor(chunk_size=500, chunk_overlap=50)
    
    # Example 1: Process a PDF from URL
    print("1. Processing PDF from URL...")
    try:
        pdf_url = "https://example.com/sample.pdf"  # Replace with actual PDF URL
        print(f"Processing: {pdf_url}")
        
        chunks = processor.process_document(pdf_url)
        
        print(f"✅ Successfully processed PDF!")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - First chunk preview: {chunks[0]['content'][:100]}...")
        print(f"   - Processor used: {chunks[0]['processor']}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to process PDF: {e}")
        print()
    
    # Example 2: Process a local PDF file
    print("2. Processing local PDF file...")
    try:
        local_pdf = "path/to/your/document.pdf"  # Replace with actual local PDF path
        
        if os.path.exists(local_pdf):
            print(f"Processing: {local_pdf}")
            chunks = processor.process_document(local_pdf)
            
            print(f"✅ Successfully processed local PDF!")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - File type: {chunks[0]['file_type']}")
            print()
        else:
            print(f"⚠️  Local PDF file not found: {local_pdf}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to process local PDF: {e}")
        print()
    
    # Example 3: Process a DOCX from URL
    print("3. Processing DOCX from URL...")
    try:
        docx_url = "https://example.com/sample.docx"  # Replace with actual DOCX URL
        print(f"Processing: {docx_url}")
        
        chunks = processor.process_document(docx_url)
        
        print(f"✅ Successfully processed DOCX!")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Sample chunk content: {chunks[0]['content'][:100]}...")
        print()
        
    except Exception as e:
        print(f"❌ Failed to process DOCX: {e}")
        print()
    
    # Example 4: Show chunk details
    print("4. Chunk structure example:")
    if 'chunks' in locals() and chunks:
        sample_chunk = chunks[0]
        print("Sample chunk structure:")
        for key, value in sample_chunk.items():
            if key == 'content':
                print(f"   {key}: {str(value)[:100]}...")
            else:
                print(f"   {key}: {value}")
        print()
    
    print("=== Example completed ===")

def demonstrate_error_handling():
    """
    Demonstrate error handling for various scenarios.
    """
    print("\n=== Error Handling Demonstration ===\n")
    
    processor = SimpleTextProcessor()
    
    # Test 1: Invalid URL
    print("1. Testing invalid URL...")
    try:
        processor.process_document("https://invalid-url-that-does-not-exist.com/doc.pdf")
    except Exception as e:
        print(f"✅ Properly handled invalid URL: {e}")
    
    # Test 2: Unsupported file type
    print("\n2. Testing unsupported file type...")
    try:
        processor.process_document("document.txt")
    except Exception as e:
        print(f"✅ Properly handled unsupported file type: {e}")
    
    # Test 3: Non-existent local file
    print("\n3. Testing non-existent local file...")
    try:
        processor.process_document("non_existent_file.pdf")
    except Exception as e:
        print(f"✅ Properly handled non-existent file: {e}")
    
    print("\n=== Error handling demonstration completed ===")

if __name__ == "__main__":
    main()
    demonstrate_error_handling()
