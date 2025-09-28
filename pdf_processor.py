"""
PDF Processing Module for Aadhaar Chat Agent

This module handles the extraction and processing of text content from PDF files
containing Aadhaar-related information. It uses PyPDF2 for text extraction and
provides methods for batch processing of multiple PDF files.

Key Features:
- Batch processing of all PDFs in a directory
- Text extraction with error handling
- Document metadata preservation
- Text chunking for vector database storage

Author: Avinav Mishra
Repository: https://github.com/avinav86/Aadhar_Agent
"""

import os
import PyPDF2
from typing import List, Dict
from pathlib import Path

class PDFProcessor:
    """
    Handles PDF text extraction and processing for the Aadhaar Chat Agent.
    
    This class provides functionality to:
    - Extract text from individual PDF files
    - Process all PDFs in a directory
    - Handle errors gracefully during extraction
    - Prepare documents for vector database storage
    
    Attributes:
        pdf_directory (Path): Path to the directory containing PDF files
    """
    
    def __init__(self, pdf_directory: str):
        """
        Initialize the PDF processor with a target directory.
        
        Args:
            pdf_directory (str): Path to the directory containing PDF files
        """
        # Convert string path to Path object for better path handling
        self.pdf_directory = Path(pdf_directory)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a single PDF file.
        
        This method uses PyPDF2 to read PDF files and extract text from all pages.
        It handles various PDF formats and provides error handling for corrupted
        or unreadable files.
        
        Args:
            pdf_path (str): Full path to the PDF file to process
            
        Returns:
            str: Extracted text content from all pages, or empty string if extraction fails
        """
        try:
            # Open PDF file in binary read mode
            with open(pdf_path, 'rb') as file:
                # Create PyPDF2 reader object
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Iterate through all pages in the PDF
                for page in pdf_reader.pages:
                    # Extract text from current page and add newline separator
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            # Log error and return empty string for failed extractions
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDF files in the configured directory.
        
        This method scans the pdf_directory for all PDF files and extracts
        text content from each one. It creates a list of document dictionaries
        containing the filename, content, and source path for each successfully
        processed PDF.
        
        Returns:
            List[Dict[str, str]]: List of document dictionaries with keys:
                - filename: Name of the PDF file
                - content: Extracted text content
                - source: Full path to the source file
        """
        documents = []
        
        # Use glob to find all PDF files in the directory
        for pdf_file in self.pdf_directory.glob("*.pdf"):
            # Display progress information
            print(f"Processing {pdf_file.name}...")
            
            # Extract text from the current PDF file
            text = self.extract_text_from_pdf(str(pdf_file))
            
            # Only add documents with non-empty content
            if text.strip():
                documents.append({
                    "filename": pdf_file.name,      # Just the filename
                    "content": text,                # Extracted text content
                    "source": str(pdf_file)        # Full path to source file
                })
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval performance.
        
        This method divides large text documents into smaller, manageable chunks
        with overlapping content. This approach improves vector search performance
        and ensures that related information isn't split across chunk boundaries.
        
        Args:
            text (str): The input text to be chunked
            chunk_size (int): Number of words per chunk (default: 1000)
            overlap (int): Number of words to overlap between chunks (default: 200)
            
        Returns:
            List[str]: List of text chunks with overlapping content
        """
        # Split text into individual words
        words = text.split()
        chunks = []
        
        # Create overlapping chunks by stepping through words
        for i in range(0, len(words), chunk_size - overlap):
            # Join words to create chunk text
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
