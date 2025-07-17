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
from datetime import datetime, timedelta
import arxiv
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv
import tempfile
import shutil
import random


@dataclass
class ExtractionConfig:
    """Configuration for complete paper extraction and filtering process."""
    # OpenRouter API settings
    openrouter_api_key: str
    model: str = "moonshotai/kimi-k2:free"  # Free model
    temperature: float = 0.1
    max_tokens: int = 1000  # Reduced from 4000 to save credits
    
    # Rate limiting for free tier
    requests_per_minute: int = 8  # Conservative for free tier
    delay_between_requests: float = 8.0  # 8 seconds between requests
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # ArXiv API settings
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    max_papers_per_query: int = 50
    days_back: int = 7  # Look back 7 days for new papers
    
    # Research domains and keywords
    target_domains: List[str] = field(default_factory=lambda: [
        "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"
    ])
    relevance_keywords: List[str] = field(default_factory=lambda: [
        "machine learning", "deep learning", "neural network", "transformer",
        "attention", "computer vision", "natural language processing",
        "reinforcement learning", "generative", "diffusion", "LLM"
    ])
    
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
    
    # Directories
    pdf_output_dir: Path = Path("relevant_pdfs")
    json_output_dir: Path = Path("relevant_json")
    
    # Relevance filtering
    relevance_threshold: float = 0.7  # Minimum relevance score (0-1)
    enable_relevance_filtering: bool = True


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
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            if content and content.strip():
                                return content, None
                            else:
                                self.logger.warning("LLM returned empty content")
                                return None, "Empty response from LLM"
                        else:
                            self.logger.error(f"Unexpected API response format: {data}")
                            return None, "Invalid API response format"
                    else:
                        error_text = await response.text()
                        self.logger.error(f"LLM API error {response.status}: {error_text}")
                        if response.status == 401:
                            return None, "Invalid API key"
                        elif response.status == 402:
                            return None, "Insufficient credits"
                        elif response.status == 404:
                            return None, f"Model {self.config.model} not found"
                        
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
        
        system_prompt = """You are a research paper analyzer. Extract key technical information from the given text chunk in a simple, readable format."""
        
        user_prompt = f"""Analyze this section from a research paper and extract key technical details:

Section: {chunk.section}
Content:
{chunk.content}

Provide a concise summary focusing on:
- Key algorithms or methods
- Important equations or formulas  
- Model architecture details
- Hyperparameters and settings
- Datasets and evaluation metrics
- Implementation details

Keep the response concise and technical."""
        
        self.logger.info(f"Analyzing chunk {chunk.chunk_id} ({chunk.section})")
        
        content, error = await self._call_llm(user_prompt, system_prompt)
        
        if error:
            self.logger.error(f"Failed to analyze chunk {chunk.chunk_id}: {error}")
            return {"error": error}
        
        # Return the raw text content - no JSON parsing needed
        analysis = {
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "analysis": content,
            "token_count": chunk.token_count
        }
        
        # Cache the result
        if self.config.enable_caching:
            with open(cache_file, 'w') as f:
                json.dump(analysis, f, indent=2)
        
        return analysis
    
    def consolidate_analyses(self, chunk_analyses: List[Dict[str, Any]], title: str = "", abstract: str = "") -> ExtractedPaperContent:
        """Consolidate analyses from all chunks into final result."""
        self.logger.info("Consolidating analyses from all chunks")
        
        consolidated = ExtractedPaperContent(title=title, abstract=abstract)
        
        # Combine all analysis text into a comprehensive summary
        all_analyses = []
        successful_chunks = 0
        failed_chunks = 0
        
        for analysis in chunk_analyses:
            if "error" in analysis:
                failed_chunks += 1
                continue
            
            successful_chunks += 1
            if "analysis" in analysis:
                section_header = f"\n=== {analysis.get('section', 'Unknown Section')} ===\n"
                all_analyses.append(section_header + analysis["analysis"])
        
        # Store the combined analysis as implementation notes
        if all_analyses:
            consolidated.implementation_notes = all_analyses
            consolidated.experimental_setup = "\n\n".join(all_analyses)
        
        # Processing stats
        consolidated.processing_stats = {
            "total_chunks_processed": len(chunk_analyses),
            "successful_analyses": successful_chunks,
            "failed_analyses": failed_chunks,
            "extraction_method": "simplified_text_analysis"
        }
        
        return consolidated
    
    async def fetch_papers_from_arxiv(self, domains: List[str] = None, max_papers: int = None) -> List[Dict[str, Any]]:
        """Fetch recent papers from ArXiv API for specified domains."""
        if domains is None:
            domains = self.config.target_domains
        if max_papers is None:
            max_papers = self.config.max_papers_per_query
        
        self.logger.info(f"Fetching papers from ArXiv for domains: {domains}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.days_back)
        
        papers = []
        
        for domain in domains:
            try:
                # Create ArXiv search query
                search = arxiv.Search(
                    query=f"cat:{domain}",
                    max_results=max_papers // len(domains),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                self.logger.info(f"Searching ArXiv for domain: {domain}")
                
                for paper in search.results():
                    # Filter by date
                    if paper.published.replace(tzinfo=None) < start_date:
                        continue
                    
                    paper_info = {
                        "id": paper.entry_id.split('/')[-1],
                        "title": paper.title,
                        "abstract": paper.summary,
                        "authors": [str(author) for author in paper.authors],
                        "published": paper.published.isoformat(),
                        "pdf_url": paper.pdf_url,
                        "domain": domain,
                        "categories": paper.categories
                    }
                    papers.append(paper_info)
                    
                    if len(papers) >= max_papers:
                        break
                
                # Rate limiting for ArXiv API
                await asyncio.sleep(3)  # ArXiv recommends 3 second delays
                
            except Exception as e:
                self.logger.error(f"Failed to fetch papers for domain {domain}: {e}")
                continue
        
        self.logger.info(f"Fetched {len(papers)} papers from ArXiv")
        return papers
    
    async def download_pdf(self, paper_info: Dict[str, Any]) -> Optional[Path]:
        """Download PDF from ArXiv."""
        pdf_url = paper_info["pdf_url"]
        paper_id = paper_info["id"]
        
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', paper_info["title"])
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:100]  # Limit length
        filename = f"{paper_id}_{safe_title}.pdf"
        
        pdf_path = self.config.pdf_output_dir / filename
        
        # Skip if already exists
        if pdf_path.exists():
            self.logger.info(f"PDF already exists: {filename}")
            return pdf_path
        
        try:
            self.logger.info(f"Downloading PDF: {filename}")
            
            # Create output directory
            self.config.pdf_output_dir.mkdir(exist_ok=True)
            
            # Download with requests (simpler than aiohttp for file downloads)
            import requests
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            # Save PDF
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Successfully downloaded: {filename}")
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"Failed to download PDF {filename}: {e}")
            return None
    
    async def check_paper_relevance(self, paper_info: Dict[str, Any], full_text: str = None) -> Tuple[bool, float, str]:
        """Check if paper is relevant using LLM analysis."""
        if not self.config.enable_relevance_filtering:
            return True, 1.0, "Relevance filtering disabled"
        
        # Use abstract if full text not available
        content_to_analyze = full_text if full_text else paper_info["abstract"]
        
        system_prompt = f"""You are an expert AI researcher. Analyze the given research paper content and determine if it's relevant for implementation.

Target domains: {', '.join(self.config.target_domains)}
Target keywords: {', '.join(self.config.relevance_keywords)}

A paper is relevant if it:
1. Presents novel algorithms, models, or techniques
2. Has clear implementation potential
3. Includes technical details and methodologies
4. Is related to machine learning, AI, or computer science
5. Has practical applications

Rate relevance from 0.0 (not relevant) to 1.0 (highly relevant)."""

        user_prompt = f"""Analyze this research paper for implementation relevance:

Title: {paper_info['title']}
Abstract: {paper_info['abstract']}
Categories: {', '.join(paper_info.get('categories', []))}

{f"Full content preview: {content_to_analyze[:2000]}..." if full_text else ""}

Respond with JSON:
{{
  "relevant": true/false,
  "relevance_score": 0.0-1.0,
  "reasoning": "explanation of why relevant or not",
  "key_contributions": ["list of main contributions"],
  "implementation_complexity": "low/medium/high",
  "recommended_for_implementation": true/false
}}"""

        content, error = await self._call_llm(user_prompt, system_prompt)
        
        if error:
            self.logger.warning(f"Relevance check failed: {error}")
            return True, 0.5, "Failed to check relevance, assuming relevant"
        
        try:
            analysis = json.loads(content)
            is_relevant = analysis.get("relevant", False)
            score = analysis.get("relevance_score", 0.0)
            reasoning = analysis.get("reasoning", "No reasoning provided")
            
            # Apply threshold
            is_relevant = is_relevant and score >= self.config.relevance_threshold
            
            return is_relevant, score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse relevance analysis: {e}")
            return True, 0.5, "Failed to parse relevance analysis"
    
    async def process_arxiv_papers(self, domains: List[str] = None, max_papers: int = None) -> Dict[str, Any]:
        """Complete pipeline: fetch, download, extract, and filter papers from ArXiv."""
        self.logger.info("Starting complete ArXiv paper processing pipeline")
        
        # Create output directories
        self.config.pdf_output_dir.mkdir(exist_ok=True)
        self.config.json_output_dir.mkdir(exist_ok=True)
        
        # Step 1: Fetch papers from ArXiv
        papers = await self.fetch_papers_from_arxiv(domains, max_papers)
        
        if not papers:
            self.logger.warning("No papers fetched from ArXiv")
            return {"processed": 0, "relevant": 0, "downloaded": 0}
        
        # Processing stats
        stats = {
            "total_fetched": len(papers),
            "downloaded": 0,
            "processed": 0,
            "relevant": 0,
            "failed": 0,
            "relevant_papers": []
        }
        
        # Step 2: Process each paper
        with tqdm(total=len(papers), desc="Processing papers") as pbar:
            for paper_info in papers:
                try:
                    pbar.set_description(f"Processing: {paper_info['title'][:50]}...")
                    
                    # Step 2a: Download PDF
                    pdf_path = await self.download_pdf(paper_info)
                    if not pdf_path:
                        stats["failed"] += 1
                        pbar.update(1)
                        continue
                    
                    stats["downloaded"] += 1
                    
                    # Step 2b: Extract text from PDF
                    try:
                        full_text, extraction_stats = self.extract_text_from_pdf(pdf_path)
                    except Exception as e:
                        self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
                        stats["failed"] += 1
                        pbar.update(1)
                        continue
                    
                    # Step 2c: Check relevance
                    is_relevant, relevance_score, reasoning = await self.check_paper_relevance(
                        paper_info, full_text
                    )
                    
                    self.logger.info(f"Paper '{paper_info['title'][:50]}...' - Relevant: {is_relevant} (Score: {relevance_score:.2f})")
                    
                    if is_relevant:
                        # Step 2d: Full content extraction and analysis
                        chunks = self.create_text_chunks(full_text)
                        chunk_analyses = []
                        
                        for chunk in chunks:
                            analysis = await self.analyze_chunk(chunk)
                            chunk_analyses.append(analysis)
                        
                        # Consolidate results
                        extracted_content = self.consolidate_analyses(chunk_analyses, paper_info["title"], paper_info["abstract"])
                        
                        # Add paper metadata
                        extracted_content.title = paper_info["title"]
                        extracted_content.abstract = paper_info["abstract"]
                        extracted_content.authors = paper_info["authors"]
                        extracted_content.publication_info = {
                            "arxiv_id": paper_info["id"],
                            "published": paper_info["published"],
                            "categories": paper_info.get("categories", []),
                            "pdf_url": paper_info["pdf_url"]
                        }
                        extracted_content.full_text = full_text
                        extracted_content.processing_stats.update(extraction_stats)
                        extracted_content.processing_stats["relevance_score"] = relevance_score
                        extracted_content.processing_stats["relevance_reasoning"] = reasoning
                        
                        # Save to JSON
                        clean_title = re.sub(r'[^\w\s-]', '', paper_info['title'])[:50]
                        json_filename = f"{paper_info['id']}_{clean_title}.json"
                        json_filename = re.sub(r'[-\s]+', '_', json_filename)
                        json_path = self.config.json_output_dir / json_filename
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(asdict(extracted_content), f, indent=2, ensure_ascii=False, default=str)
                        
                        stats["relevant"] += 1
                        stats["relevant_papers"].append({
                            "title": paper_info["title"],
                            "arxiv_id": paper_info["id"],
                            "relevance_score": relevance_score,
                            "json_file": str(json_path),
                            "pdf_file": str(pdf_path)
                        })
                        
                        self.logger.info(f"✅ Saved relevant paper: {json_filename}")
                    else:
                        self.logger.info(f"⏭️  Skipped irrelevant paper: {paper_info['title'][:50]}...")
                        # Optionally remove downloaded PDF if not relevant
                        if pdf_path and pdf_path.exists():
                            pdf_path.unlink()
                    
                    stats["processed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process paper {paper_info['title']}: {e}")
                    stats["failed"] += 1
                
                pbar.update(1)
                
                # Rate limiting between papers
                await asyncio.sleep(2)
        
        # Save processing report
        report_path = Path(f"arxiv_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"📊 Processing complete! Report saved to: {report_path}")
        self.logger.info(f"📈 Stats: {stats['relevant']}/{stats['processed']} relevant papers found")
        
        return stats

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
            result = self.consolidate_analyses(chunk_analyses, pdf_path.stem, "")
            
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
    """Main function for command-line usage - Complete ArXiv Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Paper Content Extractor - Complete ArXiv Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete ArXiv pipeline (fetch + extract + filter)
  python advanced_paper_extractor.py --arxiv --domains cs.AI cs.LG --max-papers 10
  
  # Extract from existing PDF
  python advanced_paper_extractor.py paper.pdf --output extracted.json
  
  # Fetch papers for specific domains
  python advanced_paper_extractor.py --arxiv --domains cs.CV cs.CL --days-back 3
  
  # Free tier friendly settings
  python advanced_paper_extractor.py --arxiv --model anthropic/claude-3-haiku --delay 10
        """
    )
    
    # Mode selection
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file (for single PDF mode)")
    parser.add_argument("--arxiv", action="store_true", help="Fetch papers from ArXiv (replaces v3.py)")
    
    # ArXiv fetching options
    parser.add_argument("--domains", nargs="+", default=["cs.AI", "cs.LG", "cs.CV", "cs.CL"], 
                       help="ArXiv domains to search")
    parser.add_argument("--max-papers", type=int, default=20, help="Maximum papers to fetch")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for papers")
    parser.add_argument("--keywords", nargs="+", help="Relevance keywords")
    
    # Processing options
    parser.add_argument("--output", "-o", help="Output JSON file", default="extracted_content.json")
    parser.add_argument("--model", help="OpenRouter model", default="anthropic/claude-3-haiku")
    parser.add_argument("--delay", type=float, default=8.0, help="Delay between requests (seconds)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Relevance threshold (0-1)")
    
    # Control options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-filter", action="store_true", help="Disable relevance filtering")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key in .env file:")
        print("OPENROUTER_API_KEY=your_key_here")
        return 1
    
    # Create config
    config = ExtractionConfig(
        openrouter_api_key=api_key,
        model=args.model,
        delay_between_requests=args.delay,
        relevance_threshold=args.threshold,
        verbose=args.verbose,
        enable_caching=not args.no_cache,
        enable_relevance_filtering=not args.no_filter,
        target_domains=args.domains,
        max_papers_per_query=args.max_papers,
        days_back=args.days_back
    )
    
    if args.keywords:
        config.relevance_keywords = args.keywords
    
    print("🚀 Advanced Paper Content Extractor")
    print("=" * 50)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be created")
    
    print(f"📋 Configuration:")
    print(f"   Model: {config.model}")
    print(f"   Rate limit: {config.requests_per_minute} req/min")
    print(f"   Delay: {config.delay_between_requests}s")
    print(f"   Relevance threshold: {config.relevance_threshold}")
    print(f"   Caching: {config.enable_caching}")
    print(f"   Filtering: {config.enable_relevance_filtering}")
    
    async with AdvancedPaperExtractor(config) as extractor:
        try:
            if args.arxiv:
                # Complete ArXiv pipeline (replaces v3.py)
                print(f"\n📚 ArXiv Pipeline Mode")
                print(f"   Domains: {', '.join(config.target_domains)}")
                print(f"   Max papers: {config.max_papers_per_query}")
                print(f"   Days back: {config.days_back}")
                print(f"   Keywords: {', '.join(config.relevance_keywords[:5])}...")
                
                if args.dry_run:
                    # Just fetch and show what would be processed
                    papers = await extractor.fetch_papers_from_arxiv()
                    print(f"\n🔍 Would process {len(papers)} papers:")
                    for i, paper in enumerate(papers[:10]):
                        print(f"   {i+1}. {paper['title'][:60]}...")
                    if len(papers) > 10:
                        print(f"   ... and {len(papers) - 10} more")
                    return 0
                
                # Run complete pipeline
                stats = await extractor.process_arxiv_papers()
                
                print(f"\n🎉 ArXiv Pipeline Complete!")
                print(f"📊 Final Statistics:")
                print(f"   📄 Papers fetched: {stats['total_fetched']}")
                print(f"   ⬇️  PDFs downloaded: {stats['downloaded']}")
                print(f"   🔬 Papers processed: {stats['processed']}")
                print(f"   ✅ Relevant papers: {stats['relevant']}")
                print(f"   ❌ Failed: {stats['failed']}")
                
                if stats['relevant'] > 0:
                    success_rate = (stats['relevant'] / stats['processed']) * 100
                    print(f"   📈 Success rate: {success_rate:.1f}%")
                    
                    print(f"\n✅ Relevant Papers Saved:")
                    for paper in stats['relevant_papers']:
                        print(f"   • {paper['title'][:60]}... (Score: {paper['relevance_score']:.2f})")
                        print(f"     JSON: {paper['json_file']}")
                
            elif args.pdf_path:
                # Single PDF mode
                print(f"\n📄 Single PDF Mode")
                print(f"   Input: {args.pdf_path}")
                print(f"   Output: {args.output}")
                
                if args.dry_run:
                    print(f"🔍 Would extract content from: {args.pdf_path}")
                    return 0
                
                result = await extractor.extract_from_pdf(args.pdf_path)
                
                # Save result
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(result), f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ Extraction completed!")
                print(f"📄 Output: {output_path}")
                print(f"🔍 Content summary:")
                print(f"   Mathematical formulations: {len(result.mathematical_formulations)}")
                print(f"   Algorithms: {len(result.algorithms)}")
                print(f"   Hyperparameters: {len(result.hyperparameters)}")
                print(f"   Datasets: {len(result.datasets_used)}")
                print(f"   Evaluation metrics: {len(result.evaluation_metrics)}")
                
            else:
                print("❌ Error: Must specify either --arxiv or provide a PDF path")
                print("Use --help for usage examples")
                return 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Process interrupted by user")
            return 0
        except Exception as e:
            print(f"\n❌ Process failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
