"""
Prompt builder for RAG system.

This module provides utilities for building prompts with retrieved context
for the language model to generate appropriate responses.
"""

from typing import List, Dict, Any


class PromptBuilder:
    """
    Builder class for constructing prompts with retrieved context.
    
    This class provides methods to format retrieved chunks and queries
    into effective prompts for the language model.
    """
    
    def __init__(self, system_prompt: str = None):
        """
        Initialize the prompt builder.
        
        Args:
            system_prompt: Optional custom system prompt
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """
        Get the default system prompt for insurance queries.
        
        Returns:
            Default system prompt string
        """
        return """You are an expert advisor helping evaluate insurance queries based on official policy documents.

You have been given a user's question and several relevant document sections (with page numbers).

--- INSTRUCTIONS ---
Answer the query based strictly on the context. Return ONLY the answer as a JSON object.
The format must be:

{
  "answer": "<your direct answer here>"
}

Do NOT return anything else, and avoid mentioning decisions, justification or clause references."""
    
    def build_prompt(self, user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a complete prompt with context and query.
        
        Args:
            user_query: The user's question
            retrieved_chunks: List of retrieved document chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        context = self._format_context(retrieved_chunks)
        
        return f"""{self.system_prompt}

--- CONTEXT ---
{context}

--- QUERY ---
{user_query}"""
    
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            retrieved_chunks: List of chunks with content and metadata
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for chunk in retrieved_chunks:
            page_number = chunk.get("page_number", "Unknown")
            content = chunk.get("content", "")
            context_parts.append(f"[Page {page_number}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def build_simple_prompt(self, user_query: str, context: str) -> str:
        """
        Build a simple prompt with raw context.
        
        Args:
            user_query: The user's question
            context: Raw context string
            
        Returns:
            Formatted prompt string
        """
        return f"""{self.system_prompt}

--- CONTEXT ---
{context}

--- QUERY ---
{user_query}"""
    
    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt.
        
        Args:
            new_prompt: New system prompt to use
        """
        self.system_prompt = new_prompt
