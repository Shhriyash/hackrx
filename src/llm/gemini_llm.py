"""
Google Gemini language model implementation.

This module provides language model capabilities using Google's Gemini API
for generating responses in the RAG pipeline.
"""

from typing import Optional
from google import genai
from google.genai import types
from ..interfaces import LanguageModel


class GeminiLanguageModel(LanguageModel):
    """
    Language model implementation using Google Gemini.
    
    This class provides text generation capabilities using Google's Gemini API
    for answering questions based on retrieved context.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Gemini language model.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        
        # Configure the API
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response based on the given prompt.
        
        Args:
            prompt: Input prompt for the language model
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0,
                                                         include_thoughts=False)
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error generating response with Gemini: {str(e)}")
    
    def generate_structured_response(self, prompt: str, 
                                   response_format: str = "json") -> str:
        """
        Generate a structured response in the specified format.
        
        Args:
            prompt: Input prompt for the language model
            response_format: Desired response format (e.g., "json")
            
        Returns:
            Generated response in the specified format
        """
        structured_prompt = f"{prompt}\n\nPlease respond in {response_format} format."
        return self.generate_response(structured_prompt)
    
    def get_model_info(self) -> dict:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini",
            "api_configured": bool(self.api_key)
        }
