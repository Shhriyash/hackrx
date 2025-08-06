"""
Basic example of using the RAG system.

This script demonstrates how to set up and use the RAG pipeline
for document indexing and question answering.
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from factory import RAGFactory
from config import RAGConfig


def main():
    """Main function demonstrating basic RAG usage."""
    
    # Configuration
    config = RAGConfig.default()
    
    # API keys are now loaded automatically from environment variables
    # Make sure you have set GEMINI_API_KEY in your environment or .env file
    
    # Create the RAG pipeline
    print("Creating RAG pipeline...")
    try:
        pipeline = RAGFactory.create_pipeline(config)
        print("✓ Pipeline created successfully")
    except Exception as e:
        print(f"✗ Error creating pipeline: {e}")
        if "environment variable" in str(e):
            print("\nMake sure to set your API keys:")
            print("  export GEMINI_API_KEY='your-gemini-api-key'")
            print("  Or create a .env file with: GEMINI_API_KEY=your-api-key")
        return
    
    # Check if documents are already indexed
    if not pipeline.is_indexed():
        # Index a document
        document_path = input("Enter path to PDF document to index: ").strip()
        if not os.path.exists(document_path):
            print(f"✗ Document not found: {document_path}")
            return
        
        print("Indexing document...")
        try:
            result = pipeline.index_document(document_path)
            if result["status"] == "success":
                print(f"✓ Successfully indexed {result['chunks_processed']} chunks")
            else:
                print(f"✗ Indexing failed: {result['message']}")
                return
        except Exception as e:
            print(f"✗ Error during indexing: {e}")
            return
    else:
        print("✓ Documents already indexed")
    
    # Interactive query loop
    print("\nRAG system ready! Enter your queries (type 'exit' to quit):")
    
    while True:
        query = input("\nYour Query: ").strip()
        
        if query.lower() == "exit":
            print("Exiting...")
            break
        
        if not query:
            continue
        
        try:
            print("Processing query...")
            result = pipeline.query(query)
            
            print("\n" + "="*50)
            print("ANSWER:")
            print(result["answer"])
            print(f"\nRetrieved {result['retrieved_chunks']} relevant chunks")
            print("="*50)
            
        except Exception as e:
            print(f"✗ Error processing query: {e}")


if __name__ == "__main__":
    main()
