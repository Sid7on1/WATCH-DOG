#!/usr/bin/env python3
"""
PDF Text Extractor using local PDF processing libraries
Extracts text from PDFs and stores them in organized chunks
"""

import os
from pathlib import Path
import time

try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        print("Error: No PDF library found. Please install PyPDF2 or pdfplumber:")
        print("pip install PyPDF2")
        print("or")
        print("pip install pdfplumber")
        exit(1)

class PDFTextExtractor:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.pdfs_dir = self.artifacts_dir / "pdfs"
        self.texts_dir = self.artifacts_dir / "pdf-txts"
        
        # Ensure directories exist
        self.texts_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PDF Extractor initialized using {PDF_LIBRARY}")
        print(f"PDFs directory: {self.pdfs_dir}")
        print(f"Text output directory: {self.texts_dir}")
    
    def extract_text_with_pypdf2(self, pdf_path):
        """Extract text using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            return text.strip()
            
        except Exception as e:
            print(f"Error with PyPDF2 extraction: {e}")
            return None
    
    def extract_text_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            return text.strip()
            
        except Exception as e:
            print(f"Error with pdfplumber extraction: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using available library"""
        print(f"Processing: {pdf_path.name}")
        
        if PDF_LIBRARY == "PyPDF2":
            text = self.extract_text_with_pypdf2(pdf_path)
        else:  # pdfplumber
            text = self.extract_text_with_pdfplumber(pdf_path)
        
        if text and text.strip():
            print(f"Successfully extracted text from {pdf_path.name} ({len(text)} characters)")
            return text
        else:
            print(f"No text extracted from {pdf_path.name}")
            return None
    
    def chunk_text(self, text, chunk_size=120000):
        """Split text into chunks optimized for Gemini 2.0 Flash context window
        
        Gemini 2.0 Flash has 1.05M token context window
        Target: 2.5-4% = ~26K-42K tokens = ~105K-168K characters
        Using 120K characters (~30K tokens, ~3% of context window)
        """
        if not text:
            return []
        
        # Split by paragraphs first, then by sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def save_text_chunks(self, pdf_name, text):
        """Save extracted text as numbered chunks"""
        # Create directory for this PDF
        pdf_text_dir = self.texts_dir / pdf_name.replace('.pdf', '')
        pdf_text_dir.mkdir(parents=True, exist_ok=True)
        
        # Split text into chunks
        chunks = self.chunk_text(text)
        
        if not chunks:
            print(f"No text chunks to save for {pdf_name}")
            return 0
        
        # Save each chunk as a separate file
        for i, chunk in enumerate(chunks, 1):
            chunk_file = pdf_text_dir / f"chunk_{i}.txt"
            
            try:
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(f"PDF: {pdf_name}\n")
                    f.write(f"Chunk: {i}/{len(chunks)}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(chunk)
                
                print(f"Saved chunk {i}/{len(chunks)} for {pdf_name}")
                
            except Exception as e:
                print(f"Error saving chunk {i} for {pdf_name}: {e}")
        
        return len(chunks)
    
    def process_all_pdfs(self):
        """Process all PDFs in the artifacts/pdfs directory"""
        if not self.pdfs_dir.exists():
            print(f"PDFs directory not found: {self.pdfs_dir}")
            return
        
        # Get all PDF files
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in artifacts/pdfs directory")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("=" * 60)
        
        total_chunks = 0
        processed_count = 0
        
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file.name}...")
            
            # Extract text from PDF
            extracted_text = self.extract_text_from_pdf(pdf_file)
            
            if extracted_text:
                # Save text chunks
                chunk_count = self.save_text_chunks(pdf_file.name, extracted_text)
                total_chunks += chunk_count
                processed_count += 1
                
                print(f"Created {chunk_count} chunks for {pdf_file.name}")
            else:
                print(f"Failed to extract text from {pdf_file.name}")
        
        print("\n" + "=" * 60)
        print("PDF Text Extraction Summary:")
        print(f"Total PDFs processed: {processed_count}/{len(pdf_files)}")
        print(f"Total text chunks created: {total_chunks}")
        print(f"Text files saved to: {self.texts_dir}")
        print("=" * 60)


def main():
    """Main execution function"""
    try:
        extractor = PDFTextExtractor()
        extractor.process_all_pdfs()
        
    except Exception as e:
        print(f"Error initializing PDF extractor: {e}")


if __name__ == "__main__":
    main()
