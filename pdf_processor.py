import os
import PyPDF2
from typing import List, Dict
from pathlib import Path

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs in the directory and return list of documents"""
        documents = []
        
        for pdf_file in self.pdf_directory.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            text = self.extract_text_from_pdf(str(pdf_file))
            if text.strip():
                documents.append({
                    "filename": pdf_file.name,
                    "content": text,
                    "source": str(pdf_file)
                })
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
