#!/usr/bin/env python3
"""
Advanced Paper Content Extractor - Extracts actual content from research papers
using intelligent chunking and LLM analysis for paper-specific implementations.

Features:
- Robust PDF text extraction with multiple fallbacks
- Intelligent text chunking for LLM processing
- Free-tier friendly with rate limiting and delays
- Advanced content analysis using OpenRouter LLMs
- Comprehensive error handling and recovery
- Progress tracking and resumable processing
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import hashlib
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
import arxiv
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv
import tempfile
import shutil
import random


@dataclass
class ExtractionConfig:
    """Configuration for paper extraction process."""
    # OpenRouter API settings
    openrouter_api_key: str
    model: str = "anthropic/claude-3-haiku"  # Cheaper model for free tier
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Rate limiting for free tier
    requests_per_minute: int = 10  # Conservative for free tier
    delay_between_requests: float = 6.0  # 6 seconds between requests
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # Chunking settings
    max_chunk_size: int = 3000  # Tokens per chunk
    chunk_overlap: int = 200    # Overlap between chunks
    
    # Processing settings
    enable_caching: bool = True
    cache_dir: Path = Path("extraction_cache")
    resume_on_failure: bool = True
    
    # Output settings
    save_intermediate: bool = True
    verbose: bool = True


@dataclass
class TextChunk:
    """Represents a chunk of text for processing."""
    content: str
    chunk_id: int
    section: str
    start_page: int
    end_page: int
    token_count: int
    processed: bool = False
    analysis_result: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedPaperContent:
    """Comprehensive content extracted from a research paper."""
    # Basic information
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    publication_info: Dict[str, str] = field(default_factory=dict)
    
    # Full content
    full_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    
    # Technical content
    mathematical_formulations: List[Dict[str, str]] = field(default_factory=list)
    algorithms: List[Dict[str, str]] = field(default_factory=list)
    model_architectures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Implementation details
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    loss_functions: List[str] = field(default_factory=list)
    optimization_details: Dict[str, Any] = field(default_factory=dict)
    
    # Experimental details
    datasets_used: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    experimental_setup: str = ""
    results_summary: str = ""
    
    # References and related work
    key_references: List[str] = field(default_factory=list)
    related_methods: List[str] = field(default_factory=list)
    
    # Figures and tables
    figures: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, str]] = field(default_factory=list)
    
    # Code and implementation hints
    code_snippets: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_stats: Dict[str, Any] = field(default_factory=dict)


class AdvancedPaperExtractor:
    """Advanced paper content extractor with LLM analysis."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.request_times = []
        
        # Caching
        if self.config.enable_caching:
            self.config.cache_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        self.progress_file = Path("extraction_progress.json")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("PaperExtractor")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        # File handler
        log_file = Path("paper_extraction.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AdvancedPaperExtractor/1.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF with multiple fallback methods."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Extracting text from PDF: {pdf_path}")
        
        extraction_stats = {
            "file_size": pdf_path.stat().st_size,
            "extraction_method": None,
            "pages_processed": 0,
            "extraction_time": 0
        }
        
        start_time = time.time()
        
        # Method 1: PyMuPDF (fitz) - Best for most PDFs
        try:
            text, pages = self._extract_with_pymupdf(pdf_path)
            if text and len(text.strip()) > 100:
                extraction_stats["extraction_method"] = "PyMuPDF"
                extraction_stats["pages_processed"] = pages
                extraction_stats["extraction_time"] = time.time() - start_time
                self.logger.info(f"Successfully extracted {len(text)} characters using PyMuPDF")
                return text, extraction_stats
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: PyPDF2 - Fallback
        try:
            text, pages = self._extract_with_pypdf2(pdf_path)
            if text and len(text.strip()) > 100:
                extraction_stats["extraction_method"] = "PyPDF2"
                extraction_stats["pages_processed"] = pages
                extraction_stats["extraction_time"] = time.time() - start_time
                self.logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
                return text, extraction_stats
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 3: OCR fallback (if available)
        try:
            text, pages = self._extract_with_ocr(pdf_path)
            if text and len(text.strip()) > 100:
                extraction_stats["extraction_method"] = "OCR"
                extraction_stats["pages_processed"] = pages
                extraction_stats["extraction_time"] = time.time() - start_time
                self.logger.info(f"Successfully extracted {len(text)} characters using OCR")
                return text, extraction_stats
        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
        
        raise Exception("All text extraction methods failed")
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean up text
            text = self._clean_extracted_text(text)
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
        
        doc.close()
        return "\n".join(text_parts), len(doc)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text using PyPDF2."""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text = self._clean_extracted_text(text)
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
        
        return "\n".join(text_parts), len(pdf_reader.pages)
    
    def _extract_with_ocr(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text using OCR (requires pytesseract and pdf2image)."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(str(pdf_path))
            text_parts = []
            
            for page_num, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                text = self._clean_extracted_text(text)
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
            
            return "\n".join(text_parts), len(images)
            
        except ImportError:
            raise Exception("OCR dependencies not installed (pytesseract, pdf2image)")
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        # Remove page headers/footers (common patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely headers/footers
            if (len(line) < 5 or 
                re.match(r'^\d+$', line) or  # Page numbers
                re.match(r'^Page \d+', line, re.IGNORECASE) or
                line.lower() in ['abstract', 'introduction', 'conclusion', 'references']):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_text_chunks(self, text: str) -> List[TextChunk]:
        """Create intelligent text chunks for LLM processing."""
        self.logger.info("Creating text chunks for LLM processing")
        
        # First, split by sections if possible
        sections = self._identify_sections(text)
        chunks = []
        chunk_id = 0
        
        for section_name, section_text in sections.items():
            # Calculate tokens for this section
            tokens = len(self.tokenizer.encode(section_text))
            
            if tokens <= self.config.max_chunk_size:
                # Section fits in one chunk
                chunks.append(TextChunk(
                    content=section_text,
                    chunk_id=chunk_id,
                    section=section_name,
                    start_page=0,  # Would need page tracking
                    end_page=0,
                    token_count=tokens
                ))
                chunk_id += 1
            else:
                # Split section into smaller chunks
                section_chunks = self._split_text_by_tokens(
                    section_text, 
                    self.config.max_chunk_size,
                    self.config.chunk_overlap
                )
                
                for i, chunk_text in enumerate(section_chunks):
                    chunks.append(TextChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        section=f"{section_name}_part_{i+1}",
                        start_page=0,
                        end_page=0,
                        token_count=len(self.tokenizer.encode(chunk_text))
                    ))
                    chunk_id += 1
        
        self.logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract sections from the paper."""
        sections = {"full_text": text}  # Fallback
        
        # Common section headers
        section_patterns = [
            r'(?i)^\s*(abstract)\s*$',
            r'(?i)^\s*(\d+\.?\s*introduction)\s*$',
            r'(?i)^\s*(\d+\.?\s*related\s+work)\s*$',
            r'(?i)^\s*(\d+\.?\s*method(?:ology)?)\s*$',
            r'(?i)^\s*(\d+\.?\s*approach)\s*$',
            r'(?i)^\s*(\d+\.?\s*model)\s*$',
            r'(?i)^\s*(\d+\.?\s*experiment(?:s|al\s+setup)?)\s*$',
            r'(?i)^\s*(\d+\.?\s*results?)\s*$',
            r'(?i)^\s*(\d+\.?\s*evaluation)\s*$',
            r'(?i)^\s*(\d+\.?\s*discussion)\s*$',
            r'(?i)^\s*(\d+\.?\s*conclusion)\s*$',
            r'(?i)^\s*(references)\s*$',
        ]
        
        # Find section boundaries
        lines = text.split('\n')
        section_starts = []
        
        for i, line in enumerate(lines):
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    section_name = re.match(pattern, line.strip()).group(1).strip()
                    section_starts.append((i, section_name.lower()))
                    break
        
        # Extract sections
        if section_starts:
            sections = {}
            for i, (start_line, section_name) in enumerate(section_starts):
                end_line = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(lines)
                section_text = '\n'.join(lines[start_line:end_line])
                sections[section_name] = section_text
        
        return sections
    
    def _split_text_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Split text into chunks by token count with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        return chunks
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits for free tier."""
        current_time = time.time()
        
        # Remove old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.delay_between_requests:
            sleep_time = self.config.delay_between_requests - time_since_last
            self.logger.debug(f"Enforcing delay: sleeping for {sleep_time:.1f} seconds")
            await asyncio.sleep(sleep_time)
        
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    async def _call_llm(self, prompt: str, system_prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Call OpenRouter LLM with rate limiting and error handling."""
        await self._rate_limit_check()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return content, None
                    else:
                        error_text = await response.text()
                        self.logger.error(f"LLM API error {response.status}: {error_text}")
                        
            except Exception as e:
                self.logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.backoff_factor ** attempt
                    await asyncio.sleep(sleep_time)
        
        return None, f"Failed after {self.config.max_retries} attempts"
    
    async def analyze_chunk(self, chunk: TextChunk) -> Dict[str, Any]:
        """Analyze a text chunk using LLM."""
        cache_key = hashlib.md5(chunk.content.encode()).hexdigest()
        cache_file = self.config.cache_dir / f"chunk_{cache_key}.json"
        
        # Check cache first
        if self.config.enable_caching and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                self.logger.debug(f"Using cached analysis for chunk {chunk.chunk_id}")
                return cached_result
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        system_prompt = """You are an expert research paper analyzer. Extract specific technical details from the given text chunk.

Focus on:
1. Mathematical formulations and equations
2. Algorithm descriptions and pseudocode
3. Model architecture details
4. Hyperparameters and configuration values
5. Dataset information and preprocessing steps
6. Evaluation metrics and experimental setup
7. Implementation hints and code references

Return your analysis as a structured JSON object."""
        
        user_prompt = f"""Analyze this section from a research paper:

Section: {chunk.section}
Content:
{chunk.content}

Extract all technical details in JSON format:
{{
  "mathematical_formulations": [
    {{"equation": "formula", "description": "what it represents", "variables": {{"var": "meaning"}}}}
  ],
  "algorithms": [
    {{"name": "algorithm name", "steps": ["step 1", "step 2"], "complexity": "O(n)"}}
  ],
  "model_architecture": {{
    "layers": [], "parameters": {{}}, "architecture_type": ""
  }},
  "hyperparameters": {{"param": "value"}},
  "datasets": ["dataset names"],
  "evaluation_metrics": ["metric names"],
  "implementation_notes": ["specific implementation details"],
  "key_insights": ["important findings or contributions"]
}}

Only include information that is explicitly mentioned in the text."""
        
        self.logger.info(f"Analyzing chunk {chunk.chunk_id} ({chunk.section})")
        
        content, error = await self._call_llm(user_prompt, system_prompt)
        
        if error:
            self.logger.error(f"Failed to analyze chunk {chunk.chunk_id}: {error}")
            return {"error": error}
        
        try:
            # Try to parse as JSON
            analysis = json.loads(content)
            
            # Cache the result
            if self.config.enable_caching:
                with open(cache_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return raw content as fallback
            return {"raw_content": content, "parse_error": str(e)}
    
    def consolidate_analyses(self, chunk_analyses: List[Dict[str, Any]]) -> ExtractedPaperContent:
        """Consolidate analyses from all chunks into final result."""
        self.logger.info("Consolidating analyses from all chunks")
        
        consolidated = ExtractedPaperContent()
        
        # Aggregate all findings
        all_math_formulations = []
        all_algorithms = []
        all_hyperparams = {}
        all_datasets = set()
        all_metrics = set()
        all_implementation_notes = []
        all_insights = []
        
        for analysis in chunk_analyses:
            if "error" in analysis:
                continue
            
            # Mathematical formulations
            if "mathematical_formulations" in analysis:
                all_math_formulations.extend(analysis["mathematical_formulations"])
            
            # Algorithms
            if "algorithms" in analysis:
                all_algorithms.extend(analysis["algorithms"])
            
            # Hyperparameters
            if "hyperparameters" in analysis:
                all_hyperparams.update(analysis["hyperparameters"])
            
            # Datasets
            if "datasets" in analysis:
                all_datasets.update(analysis["datasets"])
            
            # Metrics
            if "evaluation_metrics" in analysis:
                all_metrics.update(analysis["evaluation_metrics"])
            
            # Implementation notes
            if "implementation_notes" in analysis:
                all_implementation_notes.extend(analysis["implementation_notes"])
            
            # Key insights
            if "key_insights" in analysis:
                all_insights.extend(analysis["key_insights"])
        
        # Populate consolidated result
        consolidated.mathematical_formulations = all_math_formulations
        consolidated.algorithms = all_algorithms
        consolidated.hyperparameters = all_hyperparams
        consolidated.datasets_used = list(all_datasets)
        consolidated.evaluation_metrics = list(all_metrics)
        consolidated.implementation_notes = all_implementation_notes
        
        # Processing stats
        consolidated.processing_stats = {
            "total_chunks_processed": len(chunk_analyses),
            "successful_analyses": len([a for a in chunk_analyses if "error" not in a]),
            "failed_analyses": len([a for a in chunk_analyses if "error" in a]),
            "total_formulations": len(all_math_formulations),
            "total_algorithms": len(all_algorithms),
            "total_hyperparameters": len(all_hyperparams),
            "total_datasets": len(all_datasets),
            "total_metrics": len(all_metrics)
        }
        
        return consolidated
    
    async def extract_from_pdf(self, pdf_path: Union[str, Path]) -> ExtractedPaperContent:
        """Main method to extract content from PDF."""
        pdf_path = Path(pdf_path)
        self.logger.info(f"Starting advanced extraction from: {pdf_path}")
        
        # Check for existing progress
        progress_file = pdf_path.parent / f"{pdf_path.stem}_progress.json"
        if self.config.resume_on_failure and progress_file.exists():
            self.logger.info("Found existing progress, attempting to resume...")
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                # TODO: Implement resume logic
            except Exception as e:
                self.logger.warning(f"Failed to load progress: {e}")
        
        try:
            # Step 1: Extract text from PDF
            full_text, extraction_stats = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Create text chunks
            chunks = self.create_text_chunks(full_text)
            
            # Step 3: Analyze chunks with progress bar
            chunk_analyses = []
            
            with tqdm(total=len(chunks), desc="Analyzing chunks") as pbar:
                for chunk in chunks:
                    analysis = await self.analyze_chunk(chunk)
                    chunk_analyses.append(analysis)
                    chunk.processed = True
                    chunk.analysis_result = analysis
                    pbar.update(1)
                    
                    # Save intermediate progress
                    if self.config.save_intermediate:
                        progress_data = {
                            "processed_chunks": len([c for c in chunks if c.processed]),
                            "total_chunks": len(chunks),
                            "last_processed": chunk.chunk_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f, indent=2)
            
            # Step 4: Consolidate results
            result = self.consolidate_analyses(chunk_analyses)
            
            # Add basic metadata
            result.full_text = full_text
            result.processing_stats.update(extraction_stats)
            
            # Clean up progress file
            if progress_file.exists():
                progress_file.unlink()
            
            self.logger.info("Extraction completed successfully!")
            return result
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise


async def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Paper Content Extractor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file", default="extracted_content.json")
    parser.add_argument("--model", help="OpenRouter model to use", default="anthropic/claude-3-haiku")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Create config
    config = ExtractionConfig(
        openrouter_api_key=api_key,
        model=args.model,
        verbose=args.verbose,
        enable_caching=not args.no_cache,
        resume_on_failure=args.resume
    )
    
    # Extract content
    async with AdvancedPaperExtractor(config) as extractor:
        try:
            result = await extractor.extract_from_pdf(args.pdf_path)
            
            # Save result
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Extraction completed successfully!")
            print(f"📄 Output saved to: {output_path}")
            print(f"📊 Processing stats:")
            for key, value in result.processing_stats.items():
                print(f"   {key}: {value}")
            
            print(f"\n🔍 Extracted content summary:")
            print(f"   Mathematical formulations: {len(result.mathematical_formulations)}")
            print(f"   Algorithms: {len(result.algorithms)}")
            print(f"   Hyperparameters: {len(result.hyperparameters)}")
            print(f"   Datasets: {len(result.datasets_used)}")
            print(f"   Evaluation metrics: {len(result.evaluation_metrics)}")
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
