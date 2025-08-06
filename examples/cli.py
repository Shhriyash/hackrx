"""
CLI interface for the RAG system.

This script provides a command-line interface for indexing documents
and running queries against the RAG system.
"""

import sys
import os
import argparse
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from factory import RAGFactory
from config import RAGConfig


def load_config(config_path: str) -> RAGConfig:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return RAGConfig.from_dict(config_dict)
    else:
        return RAGConfig.default()


def index_command(args):
    """Handle the index command."""
    config = load_config(args.config)
    
    if not config.language_model.api_key:
        config.language_model.api_key = input("Enter your Gemini API key: ").strip()
    
    pipeline = RAGFactory.create_pipeline(config)
    
    print(f"Indexing document: {args.document}")
    result = pipeline.index_document(args.document)
    
    if result["status"] == "success":
        print(f"✓ Successfully indexed {result['chunks_processed']} chunks")
    else:
        print(f"✗ Indexing failed: {result['message']}")


def query_command(args):
    """Handle the query command."""
    config = load_config(args.config)
    
    if not config.language_model.api_key:
        config.language_model.api_key = input("Enter your Gemini API key: ").strip()
    
    pipeline = RAGFactory.create_pipeline(config)
    
    if not pipeline.is_indexed():
        print("✗ No documents indexed. Please index a document first.")
        return
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'exit':
                break
            
            result = pipeline.query(query, args.n_results)
            print(f"\nAnswer: {result['answer']}")
    else:
        # Single query mode
        result = pipeline.query(args.query, args.n_results)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print(f"Answer: {result['answer']}")


def info_command(args):
    """Handle the info command."""
    config = load_config(args.config)
    
    # Don't require API key for info
    if not config.language_model.api_key:
        config.language_model.api_key = "dummy_key"
    
    try:
        pipeline = RAGFactory.create_pipeline(config)
        info = pipeline.get_pipeline_info()
        
        print("RAG Pipeline Information:")
        print(f"  Document Processor: {info['document_processor']}")
        print(f"  Embedding Provider: {info['embedding_provider']}")
        print(f"  Vector Store: {info['vector_store']}")
        print(f"  Language Model: {info['language_model']}")
        print(f"  Documents Indexed: {'Yes' if info['is_indexed'] else 'No'}")
        
    except Exception as e:
        print(f"✗ Error getting pipeline info: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("document", help="Path to the document to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the indexed documents")
    query_parser.add_argument("--query", "-q", help="Query text (required for non-interactive mode)")
    query_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    query_parser.add_argument("--n-results", "-n", type=int, default=10, help="Number of results to retrieve")
    query_parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show pipeline information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "index":
            index_command(args)
        elif args.command == "query":
            if not args.interactive and not args.query:
                print("✗ Query text is required for non-interactive mode")
                return
            query_command(args)
        elif args.command == "info":
            info_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
