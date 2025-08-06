"""
Docling-based document processor implementation.

This module provides document processing capabilities using the Docling library
for parsing PDFs and extracting structured content with page information.
"""

import re
import json
import time
import logging
from itertools import accumulate
from typing import List, Dict, Any, Optional
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import types
import httpx

# For simple text extraction
import PyPDF2
from docx import Document
import io
from pathlib import Path

from ..interfaces import DocumentProcessor

# Setup logger for this module
logger = logging.getLogger(__name__)

class DoclingProcessor(DocumentProcessor):
    """
    Document processor using Docling for PDF parsing and text splitting.
    
    This processor converts PDF documents to markdown, preserves page information,
    and splits the content into manageable chunks with metadata.
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the Docling processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        logger.info("Initializing DoclingProcessor...")
        start_time = time.time()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize DocumentConverter
        converter_start = time.time()
        self.converter = DocumentConverter()
        converter_time = time.time() - converter_start
        logger.info(f"DocumentConverter initialized in {converter_time:.3f}s")
        
        # Initialize text splitter
        splitter_start = time.time()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        splitter_time = time.time() - splitter_start
        logger.info(f"Text splitter initialized in {splitter_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"DoclingProcessor fully initialized in {total_time:.3f}s")
    
    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF document and return chunks with metadata.
        
        Args:
            document_path: Path to the PDF document
            
        Returns:
            List of dictionaries containing chunk content and metadata
            
        Raises:
            ValueError: If no page markers are found in the processed document
        """
        logger.info(f"Starting document processing for: {document_path}")
        start_time = time.time()
        
        # Convert document to markdown with page markers
        markdown_start = time.time()
        markdown_content = self._convert_to_markdown_with_pages(document_path)
        markdown_time = time.time() - markdown_start
        logger.info(f"Markdown conversion completed in {markdown_time:.3f}s")
        
        # Split into pages and create chunks
        chunk_start = time.time()
        chunks = self._create_chunks_from_markdown(markdown_content)
        chunk_time = time.time() - chunk_start
        logger.info(f"Chunk creation completed in {chunk_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Document processing completed in {total_time:.3f}s - Created {len(chunks)} chunks")
        
        return chunks
    
    def _convert_to_markdown_with_pages(self, document_path: str) -> str:
        """
        Convert PDF to markdown with page break markers.
        
        Args:
            document_path: Path to the PDF document
            
        Returns:
            Markdown content with page markers
        """
        logger.info(f"Converting document to markdown: {document_path}")
        start_time = time.time()
        
        # Convert document
        convert_start = time.time()
        result = self.converter.convert(document_path)
        convert_time = time.time() - convert_start
        logger.info(f"Document conversion completed in {convert_time:.3f}s")
        
        # Calculate page element boundaries
        boundary_start = time.time()
        page_element_count = [len(p.assembled.elements) for p in result.pages]
        cutoffs = list(accumulate([0] + page_element_count))
        boundary_time = time.time() - boundary_start
        logger.info(f"Page boundary calculation completed in {boundary_time:.3f}s")
        
        # Export each page with page markers
        export_start = time.time()
        pages = [
            result.document.export_to_markdown(
                from_element=start,
                to_element=end,
                escape_underscores=False,
                included_content_layers={"body"},
            ).rstrip() + f"\n\n--- End of Page {i+1} ---"
            for i, (start, end) in enumerate(zip(cutoffs[:-1], cutoffs[1:]))
        ]
        export_time = time.time() - export_start
        logger.info(f"Page export completed in {export_time:.3f}s for {len(pages)} pages")
        
        total_time = time.time() - start_time
        logger.info(f"Markdown conversion fully completed in {total_time:.3f}s")
        
        return "\n\n\n".join(pages)
    
    def _create_chunks_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Create chunks from markdown content with page information.
        
        Args:
            markdown_content: Markdown content with page markers
            
        Returns:
            List of chunk dictionaries with metadata
            
        Raises:
            ValueError: If no page markers are found
        """
        logger.info("Creating chunks from markdown content")
        start_time = time.time()
        
        # Extract pages using regex
        regex_start = time.time()
        pattern = r"(.*?)--- End of Page (\d+) ---"
        matches = re.findall(pattern, markdown_content, re.DOTALL)
        regex_time = time.time() - regex_start
        logger.info(f"Regex extraction completed in {regex_time:.3f}s - Found {len(matches)} pages")
        
        if not matches:
            raise ValueError("No page markers like '--- End of Page X ---' found.")
        
        # Process chunks
        chunk_start = time.time()
        flat_chunks = []
        
        for page_text, page_number in matches:
            page_number = page_number.strip()
            chunks = self.text_splitter.split_text(page_text.strip())
            
            for i, chunk in enumerate(chunks):
                chunk_content = chunk.strip()
                if chunk_content:  # Only add non-empty chunks
                    flat_chunks.append({
                        "chunk_id": f"{page_number}_{i}",
                        "page_number": page_number,
                        "content": chunk_content
                    })
        
        chunk_time = time.time() - chunk_start
        total_time = time.time() - start_time
        logger.info(f"Chunk processing completed in {chunk_time:.3f}s")
        logger.info(f"Total chunk creation completed in {total_time:.3f}s - Created {len(flat_chunks)} chunks")
        
        return flat_chunks
    
    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], 
                           output_path: str) -> None:
        """
        Save processed chunks to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    def load_chunks_from_file(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load processed chunks from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            List of chunk dictionaries
        """
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


class GeminiFlashProcessor(DocumentProcessor):
    """
    Document processor using Gemini Flash 2.5 for PDF OCR processing.
    
    This processor downloads PDF documents from URLs, processes them with Gemini Flash 2.5
    for OCR (including tables and chart summaries), and returns page-wise content in markdown format.
    """
    
    def __init__(self, api_key: Optional[str] = None, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the Gemini Flash processor.
        
        Args:
            api_key: Google AI API key (if None, will use environment variable)
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        logger.info("Initializing GeminiFlashProcessor...")
        start_time = time.time()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Gemini client
        client_start = time.time()
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()  # Will use GOOGLE_API_KEY env var
        client_time = time.time() - client_start
        logger.info(f"Gemini client initialized in {client_time:.3f}s")
            
        # Initialize text splitter
        splitter_start = time.time()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n"],
            is_separator_regex=False
        )
        splitter_time = time.time() - splitter_start
        logger.info(f"Text splitter initialized in {splitter_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"GeminiFlashProcessor fully initialized in {total_time:.3f}s")
    
    def process_document(self, document_url: str) -> List[Dict[str, Any]]:
        """
        Process a PDF document from URL and return chunks with metadata.
        
        Args:
            document_url: URL to the PDF document
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        logger.info(f"Starting Gemini document processing for URL: {document_url}")
        start_time = time.time()
        
        # Download PDF document
        download_start = time.time()
        try:
            logger.info(f"Downloading PDF from: {document_url}")
            response = httpx.get(document_url, timeout=30.0)
            response.raise_for_status()
            doc_data = response.content
            download_time = time.time() - download_start
            logger.info(f"PDF download completed in {download_time:.3f}s - Size: {len(doc_data)} bytes")
        except httpx.HTTPError as e:
            download_time = time.time() - download_start
            logger.error(f"PDF download failed after {download_time:.3f}s: {e}")
            raise ValueError(f"Failed to download PDF from {document_url}: {e}")

        # Simple OCR prompt for faster processing
        ocr_prompt = """
Extract all text content from this PDF document. Return the content in a structured format with page information.
Focus on extracting:
1. All text content verbatim
2. Table content in readable format
3. Brief descriptions of any charts or figures

Return the text in a clean, readable format with clear page breaks indicated.
"""
        
        # Process with Gemini Flash 2.5
        gemini_start = time.time()
        try:
            logger.info("Processing PDF with Gemini Flash 2.5...")
            gemini_response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",  # Using Flash for faster processing
                contents=[
                    types.Part.from_bytes(
                        data=doc_data,
                        mime_type='application/pdf',
                    ),
                    ocr_prompt
                ]
            )
            
            extracted_text = gemini_response.text
            gemini_time = time.time() - gemini_start
            logger.info(f"Gemini processing completed in {gemini_time:.3f}s - Extracted {len(extracted_text)} characters")
        except Exception as e:
            gemini_time = time.time() - gemini_start
            logger.error(f"Gemini processing failed after {gemini_time:.3f}s: {e}")
            raise ValueError(f"Failed to process PDF with Gemini: {e}")
        
        # Split into chunks
        chunk_start = time.time()
        logger.info("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(extracted_text)
        chunk_time = time.time() - chunk_start
        logger.info(f"Text splitting completed in {chunk_time:.3f}s - Created {len(chunks)} raw chunks")
        
        # Create chunk objects
        process_start = time.time()
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                processed_chunks.append({
                    "chunk_id": f"chunk_{i}",
                    "page_number": str(i // 5 + 1),  # Approximate page mapping
                    "content": chunk.strip(),
                    "source_url": document_url,
                    "processor": "gemini_flash_lite"
                })
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        
        logger.info(f"Chunk object creation completed in {process_time:.3f}s")
        logger.info(f"Total Gemini document processing completed in {total_time:.3f}s")
        logger.info(f"Timing breakdown - Download: {download_time:.3f}s, Gemini: {gemini_time:.3f}s, Chunking: {chunk_time:.3f}s, Processing: {process_time:.3f}s")
        logger.info(f"Final result: {len(processed_chunks)} processed chunks")
        
        return processed_chunks


class SimpleTextProcessor(DocumentProcessor):
    """
    Simple document processor that extracts text directly from PDFs and DOCX files.
    
    This processor extracts text without OCR, relying on embedded text content.
    Supports both local files and URLs for PDF/DOCX documents.
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the Simple Text processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        logger.info("Initializing SimpleTextProcessor...")
        start_time = time.time()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        splitter_start = time.time()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "],
            is_separator_regex=False
        )
        splitter_time = time.time() - splitter_start
        logger.info(f"Text splitter initialized in {splitter_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"SimpleTextProcessor fully initialized in {total_time:.3f}s")
    
    def process_document(self, document_source: str) -> List[Dict[str, Any]]:
        """
        Process a PDF or DOCX document and return chunks with metadata.
        
        Args:
            document_source: Path to local file or URL to the document
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        logger.info(f"Starting simple text extraction for: {document_source}")
        start_time = time.time()
        
        # Determine if source is URL or local file
        is_url = document_source.startswith(('http://', 'https://'))
        
        if is_url:
            # Download document
            download_start = time.time()
            try:
                logger.info(f"Downloading document from: {document_source}")
                response = httpx.get(document_source, timeout=30.0)
                response.raise_for_status()
                doc_data = response.content
                download_time = time.time() - download_start
                logger.info(f"Document download completed in {download_time:.3f}s - Size: {len(doc_data)} bytes")
                
                # Determine file type from content-type or URL
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type or document_source.lower().endswith('.pdf'):
                    file_type = 'pdf'
                elif 'word' in content_type or document_source.lower().endswith(('.docx', '.doc')):
                    file_type = 'docx'
                else:
                    # Try to guess from URL extension
                    if document_source.lower().endswith('.pdf'):
                        file_type = 'pdf'
                    elif document_source.lower().endswith(('.docx', '.doc')):
                        file_type = 'docx'
                    else:
                        raise ValueError(f"Unsupported file type. Could not determine if PDF or DOCX from: {document_source}")
                
            except httpx.HTTPError as e:
                download_time = time.time() - download_start
                logger.error(f"Document download failed after {download_time:.3f}s: {e}")
                raise ValueError(f"Failed to download document from {document_source}: {e}")
        else:
            # Local file
            logger.info(f"Processing local file: {document_source}")
            file_path = Path(document_source)
            
            if not file_path.exists():
                raise ValueError(f"File not found: {document_source}")
            
            # Read file
            read_start = time.time()
            with open(file_path, 'rb') as f:
                doc_data = f.read()
            read_time = time.time() - read_start
            logger.info(f"File read completed in {read_time:.3f}s - Size: {len(doc_data)} bytes")
            
            # Determine file type from extension
            if file_path.suffix.lower() == '.pdf':
                file_type = 'pdf'
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                file_type = 'docx'
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}. Only PDF and DOCX are supported.")
            
            download_time = 0  # No download for local files
        
        # Extract text based on file type
        extract_start = time.time()
        if file_type == 'pdf':
            extracted_text = self._extract_pdf_text(doc_data)
        elif file_type == 'docx':
            extracted_text = self._extract_docx_text(doc_data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        extract_time = time.time() - extract_start
        logger.info(f"Text extraction completed in {extract_time:.3f}s - Extracted {len(extracted_text)} characters")
        
        # Split into chunks
        chunk_start = time.time()
        logger.info("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(extracted_text)
        chunk_time = time.time() - chunk_start
        logger.info(f"Text splitting completed in {chunk_time:.3f}s - Created {len(chunks)} raw chunks")
        
        # Create chunk objects
        process_start = time.time()
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                processed_chunks.append({
                    "chunk_id": f"chunk_{i}",
                    "page_number": str(i // 3 + 1),  # Approximate page mapping
                    "content": chunk.strip(),
                    "source": document_source,
                    "file_type": file_type,
                    "processor": "simple_text_extractor"
                })
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        
        logger.info(f"Chunk object creation completed in {process_time:.3f}s")
        logger.info(f"Total text extraction completed in {total_time:.3f}s")
        if is_url:
            logger.info(f"Timing breakdown - Download: {download_time:.3f}s, Extract: {extract_time:.3f}s, Chunking: {chunk_time:.3f}s, Processing: {process_time:.3f}s")
        else:
            logger.info(f"Timing breakdown - Read: {read_time:.3f}s, Extract: {extract_time:.3f}s, Chunking: {chunk_time:.3f}s, Processing: {process_time:.3f}s")
        logger.info(f"Final result: {len(processed_chunks)} processed chunks")
        
        return processed_chunks
    
    def _extract_pdf_text(self, pdf_data: bytes) -> str:
        """
        Extract text from PDF data using PyPDF2.
        
        Args:
            pdf_data: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        logger.info("Extracting text from PDF using PyPDF2...")
        start_time = time.time()
        
        try:
            # Create PDF reader from bytes
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            extracted_text = "\n\n".join(text_parts)
            extract_time = time.time() - start_time
            logger.info(f"PDF text extraction completed in {extract_time:.3f}s - {len(pdf_reader.pages)} pages processed")
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from PDF. The PDF might contain only images or scanned content.")
            
            return extracted_text
            
        except Exception as e:
            extract_time = time.time() - start_time
            logger.error(f"PDF text extraction failed after {extract_time:.3f}s: {e}")
            raise ValueError(f"Failed to extract text from PDF: {e}")
    
    def _extract_docx_text(self, docx_data: bytes) -> str:
        """
        Extract text from DOCX data using python-docx.
        
        Args:
            docx_data: DOCX file content as bytes
            
        Returns:
            Extracted text content
        """
        logger.info("Extracting text from DOCX using python-docx...")
        start_time = time.time()
        
        try:
            # Create document from bytes
            docx_file = io.BytesIO(docx_data)
            doc = Document(docx_file)
            
            # Extract text from all paragraphs
            text_parts = []
            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text.strip()
                if para_text:
                    text_parts.append(para_text)
            
            # Extract text from tables
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        table_text.append(" | ".join(row_text))
                
                if table_text:
                    text_parts.append("--- Table ---\n" + "\n".join(table_text))
            
            extracted_text = "\n\n".join(text_parts)
            extract_time = time.time() - start_time
            logger.info(f"DOCX text extraction completed in {extract_time:.3f}s - {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables processed")
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from DOCX document.")
            
            return extracted_text
            
        except Exception as e:
            extract_time = time.time() - start_time
            logger.error(f"DOCX text extraction failed after {extract_time:.3f}s: {e}")
            raise ValueError(f"Failed to extract text from DOCX: {e}")
